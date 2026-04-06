import argparse
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None

from dataset import create_dataloaders
from model import VanillaPointNet
from utils.common import accuracy_from_logits, resolve_device, save_checkpoint, save_metrics_json, set_seed


def parse_args():
    default_data_root = str(
        Path(__file__).resolve().parent.parent / "ModelNet-10-20260404T105413Z-3-001" / "ModelNet-10"
    )
    parser = argparse.ArgumentParser(description="Train Vanilla PointNet on ModelNet-10")
    parser.add_argument("--data-root", type=str, default=default_data_root)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train_mode: bool):
    if train_mode:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for points, labels in tqdm(loader, leave=False):
            points = points.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            logits = model(points)
            loss = criterion(logits, labels)

            if train_mode:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            correct, total = accuracy_from_logits(logits, labels)
            running_correct += correct
            running_total += total

    epoch_loss = running_loss / max(1, running_total)
    epoch_acc = running_correct / max(1, running_total)
    return epoch_loss, epoch_acc


def plot_curves(metrics: Dict[str, List[float]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(metrics["train_loss"]) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(epochs, metrics["train_loss"], color="tab:blue")
    axes[0, 0].set_title("Train Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].plot(epochs, metrics["val_loss"], color="tab:orange")
    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")

    axes[1, 0].plot(epochs, metrics["train_acc"], color="tab:green")
    axes[1, 0].set_title("Train Accuracy")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")

    axes[1, 1].plot(epochs, metrics["val_acc"], color="tab:red")
    axes[1, 1].set_title("Validation Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(prefer_cuda=not args.force_cpu)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    train_ds, test_ds, train_loader, test_loader = create_dataloaders(
        root_dir=args.data_root,
        num_points=args.num_points,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
    )

    model = VanillaPointNet(num_classes=len(train_ds.class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    start_time = time.time()

    use_wandb = (not args.disable_wandb) and (args.wandb_project is not None)
    if use_wandb and wandb is None:
        print("wandb is not installed. Continuing without wandb logging.")
        use_wandb = False

    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "num_points": args.num_points,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "device": str(device),
            },
        )

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train_mode=True)
        val_loss, val_acc = run_epoch(model, test_loader, criterion, optimizer, device, train_mode=False)

        metrics["train_loss"].append(train_loss)
        metrics["val_loss"].append(val_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed / 60.0:.2f} minutes")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hyperparams = {
        "num_points": args.num_points,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "device": str(device),
    }

    ckpt_path = out_dir / "pointnet_model.pt"
    save_checkpoint(str(ckpt_path), model, train_ds.class_names, hyperparams)
    save_metrics_json(metrics, str(out_dir / "metrics.json"))
    plot_curves(metrics, out_dir)

    print(f"Saved checkpoint: {ckpt_path}")
    print(f"Saved metrics json: {out_dir / 'metrics.json'}")
    print(f"Saved plot: {out_dir / 'training_curves.png'}")

    if use_wandb:
        wandb.log({"training_minutes": elapsed / 60.0})
        wandb.finish()


if __name__ == "__main__":
    main()
