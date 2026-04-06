from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import DataLoader, Dataset


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _read_ply_xyz(file_path: Path) -> np.ndarray:
    ply = PlyData.read(str(file_path))
    vertex = ply["vertex"]
    x = np.asarray(vertex["x"], dtype=np.float32)
    y = np.asarray(vertex["y"], dtype=np.float32)
    z = np.asarray(vertex["z"], dtype=np.float32)
    return np.stack((x, y, z), axis=1)


def _read_off_xyz(file_path: Path) -> np.ndarray:
    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline().strip()
        if header != "OFF":
            raise ValueError(f"Invalid OFF header in {file_path}")
        counts = f.readline().strip().split()
        num_vertices = int(counts[0])
        vertices = []
        for _ in range(num_vertices):
            parts = f.readline().strip().split()
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.asarray(vertices, dtype=np.float32)


def load_point_cloud(file_path: Path) -> np.ndarray:
    suffix = file_path.suffix.lower()
    if suffix == ".ply":
        return _read_ply_xyz(file_path)
    if suffix == ".off":
        return _read_off_xyz(file_path)
    raise ValueError(f"Unsupported point-cloud format: {file_path}")


def center_and_normalize(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32, copy=True)
    centroid = points.mean(axis=0, keepdims=True)
    points -= centroid
    max_dist = np.linalg.norm(points, axis=1).max()
    if max_dist > 0:
        points /= max_dist
    return points


def subsample_or_pad(points: np.ndarray, num_points: int) -> np.ndarray:
    n = points.shape[0]
    if n == num_points:
        return points
    if n > num_points:
        idx = np.random.choice(n, num_points, replace=False)
        return points[idx]
    pad_idx = np.random.choice(n, num_points - n, replace=True)
    padded = np.concatenate([points, points[pad_idx]], axis=0)
    return padded


def random_rotate_z(points: np.ndarray) -> np.ndarray:
    theta = np.random.uniform(0, 2 * np.pi)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return points @ rot.T


class PointCloudDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str,
        num_points: int = 1024,
        class_names: Optional[List[str]] = None,
        augment: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")

        self.root_dir = Path(root_dir)
        self.split = split
        self.num_points = num_points
        self.augment = augment

        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        if class_names is None:
            class_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        self.class_names = class_names
        self.class_to_idx: Dict[str, int] = {name: i for i, name in enumerate(self.class_names)}

        self.samples: List[Tuple[Path, int]] = []
        for class_name in self.class_names:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            for ext in ("*.ply", "*.off"):
                for fpath in sorted(class_dir.glob(ext)):
                    self.samples.append((fpath, self.class_to_idx[class_name]))

        if not self.samples:
            raise RuntimeError(f"No .ply or .off files found in {split_dir}")

        if max_samples is not None:
            self.samples = self.samples[: max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fpath, label = self.samples[idx]
        points = load_point_cloud(fpath)
        points = center_and_normalize(points)
        if self.augment:
            points = random_rotate_z(points)
        points = subsample_or_pad(points, self.num_points)
        points_tensor = torch.from_numpy(points).float().transpose(0, 1)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return points_tensor, label_tensor

    def get_raw_item(self, idx: int) -> Tuple[np.ndarray, int, Path]:
        fpath, label = self.samples[idx]
        points = load_point_cloud(fpath)
        points = center_and_normalize(points)
        return points, label, fpath


def create_dataloaders(
    root_dir: str,
    num_points: int = 1024,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
):
    set_global_seed(seed)
    train_ds = PointCloudDataset(
        root_dir=root_dir,
        split="train",
        num_points=num_points,
        augment=True,
        max_samples=max_train_samples,
    )
    test_ds = PointCloudDataset(
        root_dir=root_dir,
        split="test",
        num_points=num_points,
        class_names=train_ds.class_names,
        augment=False,
        max_samples=max_test_samples,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_ds, test_ds, train_loader, test_loader
