import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(
    ckpt_path: str,
    model: torch.nn.Module,
    class_names: List[str],
    hyperparams: Dict,
) -> None:
    out_path = Path(ckpt_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "hyperparams": hyperparams,
    }
    torch.save(payload, out_path)


def load_checkpoint(ckpt_path: str, device: torch.device):
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    return payload


def save_metrics_json(metrics: Dict, out_path: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Tuple[int, int]:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct, total
