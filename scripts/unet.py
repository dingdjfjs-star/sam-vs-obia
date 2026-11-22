# -*- coding: utf-8 -*-
"""手写标准 U-Net（encoder-decoder + 跳连），v2 第三方案。

设计说明：
* 经典 4 级下采样 U 形结构（DoubleConv=Conv3x3+BN+ReLU ×2），宽度基 32
  （标准 64 的减半版——RTX 4060 8GB 下 512² batch 8 可稳定训练，架构不变）；
* 输出 head 1×1 conv → n_classes logits，交叉熵（ignore_index 可配）；
* 无任何预训练权重——"训练类方案"的证据就是它在这份数据的 train split 上
  从零学出来的，训练曲线全程落盘。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """base=32 → 通道 32/64/128/256/512；标准 U-Net 拓扑，无预训练。"""

    def __init__(self, in_ch=3, n_classes=7, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.enc1 = DoubleConv(in_ch, c[0])
        self.enc2 = DoubleConv(c[0], c[1])
        self.enc3 = DoubleConv(c[1], c[2])
        self.enc4 = DoubleConv(c[2], c[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c[3], c[4])
        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.dec4 = DoubleConv(c[4], c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = DoubleConv(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = DoubleConv(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = DoubleConv(c[1], c[0])
        self.head = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)          # (B, n_classes, H, W) logits
