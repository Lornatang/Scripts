# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""文件说明: 实现模型定义功能."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

__all__ = [
    "SqueezeExcitationModule", "DenselyAttentionConv",
    "LightAttentionConvBlock", "DenselyAttentionConvBlock",
    "Generator",
]


class SqueezeExcitationModule(nn.Module):
    """注意力卷积模块. 自动提取一幅图像中感兴趣区域, 在这里实现的是一种软注意力方法,
    通过反向传播更新注意力卷积模块机制内部的权重.

    Attributes:
        se_module (nn.Sequential): 定义注意力卷积方法.

    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        """

        Args:
            channels (int): 输入图像的通道数.
            reduction (optional, int): 通道数降维因子.

        """
        super(SqueezeExcitationModule, self).__init__()
        hidden_channels = channels // reduction

        self.se_module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden_channels, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(hidden_channels, channels, (1, 1), (1, 1), (0, 0)),
            nn.Hardsigmoid(True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): Tensor(NCHW)格式图像数据.

        Returns:
            torch.Tensor: 注意力卷积处理后Tensor(NCHW)格式图像数据.

        """
        out = self.se_module(x)
        out = torch.mul(out, x)

        return out


class DenselyAttentionConv(nn.Module):
    def __init__(self, channels: int, reduction: int = 2) -> None:
        super(DenselyAttentionConv, self).__init__()
        growths = channels // reduction

        self.conv1 = nn.Sequential(
            nn.Conv2d(channels + growths * 0, growths, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels + growths * 1, growths, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels + growths * 2, growths, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels + growths * 3, growths, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels + growths * 4, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        self.squeeze_excitation = SqueezeExcitationModule(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.conv1(x)
        out2 = self.conv2(torch.cat((x, out1), 1))
        out3 = self.conv3(torch.cat((x, out1, out2), 1))
        out4 = self.conv4(torch.cat((x, out1, out2, out3), 1))
        out5 = self.conv5(torch.cat((x, out1, out2, out3, out4), 1))

        out = self.squeeze_excitation(out5)
        out = torch.add(out, identity)

        return out


class LightAttentionConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(LightAttentionConvBlock, self).__init__()
        self.lac_block = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))
        )

        self.squeeze_excitation = SqueezeExcitationModule(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.lac_block(x)

        out = self.squeeze_excitation(out)
        out = torch.add(out, identity)

        return out


class DenselyAttentionConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(DenselyAttentionConvBlock, self).__init__()
        self.dac_block = nn.Sequential(
            DenselyAttentionConv(channels),
            DenselyAttentionConv(channels),
            DenselyAttentionConv(channels),
            DenselyAttentionConv(channels),
            DenselyAttentionConv(channels)
        )

        self.squeeze_excitation = SqueezeExcitationModule(channels)

    def forward(self, x):
        identity = x

        out = self.dac_block(x)

        out = self.squeeze_excitation(out)
        out = torch.add(out, identity)

        return out


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        self.conv_block1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))

        trunk = []
        for _ in range(2):
            trunk.append(LightAttentionConvBlock(64))
        for _ in range(2):
            trunk.append(DenselyAttentionConvBlock(64))
        for _ in range(2):
            trunk.append(LightAttentionConvBlock(64))
        self.trunk = nn.Sequential(*trunk)

        self.conv_block2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_block4 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))

        # 初始化模型权重.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # 支持Torch.script方法.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv_block1(x)
        out = self.trunk(out1)
        out2 = self.conv_block2(out)
        out = out1 + out2
        out = self.upsampling(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.conv_block3(out)
        out = self.conv_block4(out)

        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                m.weight.data *= 0.1
