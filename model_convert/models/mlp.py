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

# ==============================================================================
# 文件说明: 实现模型定义功能.
# ==============================================================================
import torch
import torch.nn as nn
from torch import Tensor

__all__ = [
    "SqueezeExcitation", "DenselyConnected",
    "LightAttentionConvBlock", "DenselyAttentionConvBlock",
    "Model",
]


class SqueezeExcitation(nn.Module):
    """该模块是嵌入至生成网络中的Trunk_a/b/c/d当中的,它主要功能是使得网络自动学习特征之间的重要程度.

    `Squeeze-and-Excitation Networks <https://arxiv.org/pdf/1709.01507v4.pdf>` paper.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        """

        Args:
            channels (int): 输入图像中的通道数.
            reduction (optional, int): 特征通道压缩因子. (Default: 4)
        """
        super(SqueezeExcitation, self).__init__()
        self.se_block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels // reduction, channels, (1, 1), (1, 1), (0, 0)),
            nn.Hardsigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.se_block(x)
        out = out * x

        return out


class DenselyConnected(nn.Module):
    """该模块是嵌入至生成网络中的Trunk_c当中的, 它主要功能是增强网络特征信息相关度.

    `Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.
    """

    def __init__(self, channels: int) -> None:
        """

        Args:
            channels (int): 输入图像中的通道数.
        """
        super(DenselyConnected, self).__init__()
        growths = channels // 2
        self.conv1 = nn.Conv2d(channels + growths * 0, growths,  (3, 3), (1, 1), (1, 1))
        self.conv2 = nn.Conv2d(channels + growths * 1, growths,  (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(channels + growths * 2, growths,  (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(channels + growths * 3, growths,  (3, 3), (1, 1), (1, 1))
        self.conv5 = nn.Conv2d(channels + growths * 4, channels, (3, 3), (1, 1), (1, 1))

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat((x, out1), 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat((x, out1, out2), 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat((x, out1, out2, out3), 1)))
        out5 = self.identity(self.conv5(torch.cat((x, out1, out2, out3, out4), 1)))
        out = out5 + identity

        return out


class LightAttentionConvBlock(nn.Module):
    """Trunk_a/c模块中的轻量级注意力卷积实现.

    `Squeeze-and-Excitation Networks <https://arxiv.org/pdf/1709.01507v4.pdf>` paper.
    """

    def __init__(self, channels: int) -> None:
        """

        Args:
            channels (int): 输入图像中的通道数.
        """
        super(LightAttentionConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1))
        )
        self.squeeze_excitation = SqueezeExcitation(channels)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv_block(x)
        out = self.squeeze_excitation(out)
        out = out + identity
        out = self.leaky_relu(out)

        return out


class DenselyAttentionConvBlock(nn.Module):
    """Trunk_b模块中的密集注意力卷积实现.

    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.
    """

    def __init__(self, channels: int) -> None:
        """

        Args:
            channels (int): 输入图像中的通道数.
        """
        super(DenselyAttentionConvBlock, self).__init__()
        self.dc_block = nn.Sequential(
            DenselyConnected(channels),
            DenselyConnected(channels),
            DenselyConnected(channels),
            DenselyConnected(channels),
            DenselyConnected(channels)
        )
        self.squeeze_excitation = SqueezeExcitation(channels)
        self.leaky_relu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        identity = x

        out = self.dc_block(x)
        out = self.squeeze_excitation(out)
        out = out + identity
        out = self.leaky_relu(out)

        return out


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        # 第一层卷积层.
        self.conv_block1 = nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1))

        # 四个特征提取主干网络.
        trunk = []
        for _ in range(32):
            trunk.append(LightAttentionConvBlock(64))
        for _ in range(16):
            trunk.append(DenselyAttentionConvBlock(64))
        for _ in range(32):
            trunk.append(LightAttentionConvBlock(64))
        self.trunk = nn.Sequential(*trunk)

        # 特征提取网络后重新接一层对称卷积块.
        self.conv_block2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))

        # 上采样卷积层.
        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 256, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor=2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 256, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor=2),
            nn.LeakyReLU(0.2, True)
        )

        # 上采样块后卷积层.
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True)
        )

        # 输出层.
        self.conv_block4 = nn.Conv2d(64, 3, (3, 3), (1, 1), (1, 1))

        # 初始化模型权重.
        self._initialize_weights()

        self.identity = nn.Identity();

    # 使用Script模型跟踪方法必须这样定义.
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.identity(x)
        # out = self.trunk(out1)
        # out2 = self.conv_block2(out)
        # out = out1 + out2
        # out = self.upsampling(out)
        # out = self.conv_block3(out)
        # out = self.conv_block4(out)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                m.weight.data *= 0.1
