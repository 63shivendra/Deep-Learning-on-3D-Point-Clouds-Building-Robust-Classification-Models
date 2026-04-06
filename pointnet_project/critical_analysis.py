import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataset import PointCloudDataset, create_dataloaders
from model import VanillaPointNet
from utils.common import accuracy_from_logits, load_checkpoint, resolve_device, set_seed


def parse_args():
    default_data_root = str(
        Path(__file__).resolve().parent.parent / "ModelNet-10-20260404T105413Z-3-001" / "ModelNet-10"
    )
    parser = argparse.ArgumentParser(description="Critical point analysis for PointNet")
    parser.add_argument("--data-root", type=str, default=default_data_root)
    parser.add_argument("--ckpt", type=str, default="outputs/pointnet_model.pt")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-visuals", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--out-dir", type=str, default="outputs/critical_analysis")
    return parser.parse_args()


def set_axes_equal(ax, points: np.ndarray):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = (maxs - mins).max() / 2.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def extract_critical_indices(model: VanillaPointNet, points_bcn: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        _, point_features = model.forward_with_point_features(points_bcn)
        max_indices = torch.argmax(point_features, dim=2)[0]
        unique_indices = torch.unique(max_indices).detach().cpu().numpy()
    return np.sort(unique_indices)


def evaluate_original_accuracy(model, test_loader, device):
    model.eval()
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for points, labels in tqdm(test_loader, leave=False, desc="Original accuracy"):
            points = points.to(device)
            labels = labels.to(device)
            logits = model(points)
            correct, total = accuracy_from_logits(logits, labels)
            total_correct += correct
            total_seen += total
    return total_correct / max(1, total_seen)


def evaluate_sparse_critical_accuracy(model, test_ds: PointCloudDataset, device):
    model.eval()
    total_correct = 0
    total_seen = 0

    for idx in tqdm(range(len(test_ds)), desc="Sparse critical accuracy"):
        points_np, label, _ = test_ds.get_raw_item(idx)
        points_t = torch.from_numpy(points_np).float().transpose(0, 1).unsqueeze(0).to(device)
        crit_idx = extract_critical_indices(model, points_t)
        sparse_points = points_np[crit_idx]

        sparse_t = torch.from_numpy(sparse_points).float().transpose(0, 1).unsqueeze(0).to(device)
        label_t = torch.tensor([label], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(sparse_t)
        correct, total = accuracy_from_logits(logits, label_t)
        total_correct += correct
        total_seen += total

    return total_correct / max(1, total_seen)


def visualize_critical_points(model, test_ds: PointCloudDataset, out_dir: Path, num_visuals: int, device):
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_indices = np.linspace(0, len(test_ds) - 1, num=min(num_visuals, len(test_ds)), dtype=int)
    for i, sample_idx in enumerate(sample_indices):
        points_np, label, fpath = test_ds.get_raw_item(int(sample_idx))
        points_t = torch.from_numpy(points_np).float().transpose(0, 1).unsqueeze(0).to(device)
        crit_idx = extract_critical_indices(model, points_t)

        fig = plt.figure(figsize=(12, 5))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")

        ax1.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=3, c="steelblue", alpha=0.9)
        ax1.set_title("Original Point Cloud")

        ax2.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], s=2, c="gray", alpha=0.15)
        ax2.scatter(
            points_np[crit_idx, 0],
            points_np[crit_idx, 1],
            points_np[crit_idx, 2],
            s=12,
            c="crimson",
            alpha=0.95,
        )
        ax2.set_title(f"Critical Points ({len(crit_idx)} unique)")

        for ax in (ax1, ax2):
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            set_axes_equal(ax, points_np)
            ax.view_init(elev=25, azim=45)

        class_name = test_ds.class_names[label]
        fig.suptitle(f"Sample: {fpath.name} | Class: {class_name}")
        fig.tight_layout()
        save_path = out_dir / f"critical_vis_{i + 1:02d}.png"
        fig.savefig(save_path, dpi=220)
        plt.close(fig)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(prefer_cuda=not args.force_cpu)
    print(f"Using device: {device}")

    _, test_ds, _, test_loader = create_dataloaders(
        root_dir=args.data_root,
        num_points=args.num_points,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        max_train_samples=1,
        max_test_samples=args.max_test_samples,
    )

    payload = load_checkpoint(args.ckpt, device=device)
    class_names = payload["class_names"]

    model = VanillaPointNet(num_classes=len(class_names)).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    out_dir = Path(args.out_dir)

    print("Extracting and visualizing critical points...")
    visualize_critical_points(model, test_ds, out_dir=out_dir, num_visuals=args.num_visuals, device=device)

    original_acc = evaluate_original_accuracy(model, test_loader, device)
    sparse_acc = evaluate_sparse_critical_accuracy(model, test_ds, device)

    print(f"Original test accuracy: {original_acc * 100.0:.2f}%")
    print(f"Sparse critical-points accuracy: {sparse_acc * 100.0:.2f}%")
    print(f"Accuracy drop: {(original_acc - sparse_acc) * 100.0:.2f}%")
    print(
        "Explanation: PointNet global features come from max pooling over per-point activations. "
        "Critical points are those that win these maxima, so they retain most discriminative information."
    )
    print(f"Saved visualizations in: {out_dir}")


if __name__ == "__main__":
    main()
