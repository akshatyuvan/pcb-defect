"""
PCBNet: a small CNN trained from scratch (locked decision 5).

Why from scratch and not a pretrained ResNet: ImageNet features are learned on
natural colour photographs. These inputs are binarised single-channel copper
traces. The transfer is weak, and more importantly the point is to show I can
specify and train an architecture, not that I can call a constructor.

Why GAP instead of Flatten -> Dense: a flatten head would need a 4096-unit dense
layer, would dominate the parameter count, and would destroy the spatial
correspondence Grad-CAM depends on. GAP keeps the final conv map spatially
meaningful, which is what makes Day 4's pointing-game score interpretable.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv-BN-ReLU twice, then MaxPool. Halves spatial resolution."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.body = nn.Sequential(
            # bias=False because the BatchNorm immediately after has its own
            # affine shift, so a conv bias is redundant parameters that never
            # learn anything independent of it.
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.pool(self.body(x))


class PCBNet(nn.Module):
    def __init__(self, num_classes: int = 7, in_ch: int = 1,
                 widths: tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        blocks = []
        c = in_ch
        for w in widths:
            blocks.append(ConvBlock(c, w))
            c = w
        # Exposed as .features (not folded into forward) because Day 4's Grad-CAM
        # hooks the output of the last block. A named submodule keeps the hook
        # target stable rather than depending on module iteration order.
        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(c, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 64, 64) -> feature map (B, 256, 4, 4) after four /2 pools
        fmap = self.features(x)
        pooled = self.gap(fmap).flatten(1)      # (B, 256)
        return self.classifier(pooled)          # (B, 7)

    def forward_with_features(self, x: torch.Tensor):
        """Day 4 convenience: logits plus the pre-GAP map, no second forward pass."""
        fmap = self.features(x)
        logits = self.classifier(self.gap(fmap).flatten(1))
        return logits, fmap


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    m = PCBNet()
    x = torch.randn(4, 1, 64, 64)
    logits, fmap = m.forward_with_features(x)
    print("logits:", tuple(logits.shape))     # (4, 7)
    print("feature map:", tuple(fmap.shape))  # (4, 256, 4, 4)
    print("params:", count_params(m))         # 1174439