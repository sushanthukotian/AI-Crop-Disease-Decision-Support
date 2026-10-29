
import torch
import torch.nn as nn
from torchvision import models


class SpatialAttention(nn.Module):
    """Lightweight spatial attention for highlighting useful leaf regions."""

    def __init__(self, kernel_size=7):
        super().__init__()

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.max(x, dim=1, keepdim=True)[0]

        attention_map = torch.cat([avg_map, max_map], dim=1)
        attention_map = self.sigmoid(self.conv(attention_map))

        return x * attention_map


class CropDiseaseModel(nn.Module):
    """
    Memory-efficient MobileNetV2 with spatial attention
    for 38-class crop disease classification.

    Note: Train this architecture before using its checkpoint.
    """

    def __init__(self, num_classes=38):
        super().__init__()

        # No pretrained weights are downloaded during initialization.
        self.backbone = models.mobilenet_v2(weights=None)

        # Use MobileNetV2 as the only feature extractor.
        self.features = self.backbone.features

        # Lightweight spatial attention.
        self.attention = SpatialAttention(kernel_size=7)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Small classifier instead of the previous 26,368 -> 4,096 layer.
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.backbone.last_channel, num_classes)
        )

        # The original classifier is no longer required.
        del self.backbone.classifier
        del self.backbone

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
