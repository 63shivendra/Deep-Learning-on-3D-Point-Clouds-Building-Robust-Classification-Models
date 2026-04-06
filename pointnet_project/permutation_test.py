import argparse
from pathlib import Path

import torch

from dataset import create_dataloaders
from model import VanillaPointNet
from utils.common import load_checkpoint, resolve_device, set_seed


def parse_args():
    default_data_root = str(
        Path(__file__).resolve().parent.parent / "ModelNet-10-20260404T105413Z-3-001" / "ModelNet-10"
    )
    parser = argparse.ArgumentParser(description="Permutation invariance test for PointNet")
    parser.add_argument("--data-root", type=str, default=default_data_root)
    parser.add_argument("--ckpt", type=str, default="outputs/pointnet_model.pt")
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--max-test-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(prefer_cuda=not args.force_cpu)

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

    changed = 0
    total = 0

    with torch.no_grad():
        for points, _ in test_loader:
            points = points.to(device)
            original_logits = model(points)
            original_pred = original_logits.argmax(dim=1)

            permuted_points = points.clone()
            bsz, _, num_points = permuted_points.shape
            for i in range(bsz):
                perm_idx = torch.randperm(num_points, device=device)
                permuted_points[i] = permuted_points[i, :, perm_idx]

            permuted_logits = model(permuted_points)
            permuted_pred = permuted_logits.argmax(dim=1)

            changed += (original_pred != permuted_pred).sum().item()
            total += points.size(0)

    pct_changed = 100.0 * changed / max(1, total)

    print(f"Total samples tested: {total}")
    print(f"Predictions changed after permutation: {changed}")
    print(f"Percentage changed: {pct_changed:.4f}%")
    print(
        "Expected behavior: this should be near 0% because PointNet uses shared MLPs and global max pooling, "
        "which are permutation-invariant over point order."
    )


if __name__ == "__main__":
    main()
