from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaPointNet(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(p=dropout)

        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(p=dropout)

        self.fc3 = nn.Linear(256, num_classes)

    def extract_point_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    def global_feature(self, x: torch.Tensor) -> torch.Tensor:
        point_features = self.extract_point_features(x)
        global_feat = torch.max(point_features, dim=2)[0]
        return global_feat

    def classify_global_feature(self, global_feat: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn4(self.fc1(global_feat)))
        x = self.drop1(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.drop2(x)
        logits = self.fc3(x)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_feat = self.global_feature(x)
        logits = self.classify_global_feature(global_feat)
        return logits

    def forward_with_point_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        point_features = self.extract_point_features(x)
        global_feat = torch.max(point_features, dim=2)[0]
        logits = self.classify_global_feature(global_feat)
        return logits, point_features
