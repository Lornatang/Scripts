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
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch import Tensor

__all__ = [
    "SqueezeExcitation", "DenselyConnected",
    "LightAttentionConvBlock", "DenselyAttentionConvBlock",
    "Discriminator", "Generator",
    "PerceptualLoss"
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
        self.identity   = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat((x, out1), 1)))
        out3 = self.leaky_relu(self.conv3(torch.cat((x, out1, out2), 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat((x, out1, out2, out3), 1)))
        out5 = self.identity(  self.conv5(torch.cat((x, out1, out2, out3, out4), 1)))
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


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 128 x 128
            nn.Conv2d(3,  64,   (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 64 x 64
            nn.Conv2d(64, 64,   (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128,  (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 32 x 32
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 16 x 16
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 8 x 8
            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 4 x 4
            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
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

    # Torch.script方法下模型应该如此定义.
    def _forward_impl(self, x: Tensor) -> Tensor:
        out1 = self.conv_block1(x)
        out  = self.trunk(out1)
        out2 = self.conv_block2(out)
        out  = out1 + out2
        out  = self.upsampling(out)
        out  = self.conv_block3(out)
        out  = self.conv_block4(out)

        return out

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
                m.weight.data *= 0.1


class PerceptualLoss(nn.Module):
    """构建了基于VGG19网络的感知损失函数.
    使用来自前几层的低级别的特征映射层会更专注于图像的纹理和色彩内容.

    论文参考列表:
        - `Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network <https://arxiv.org/pdf/1609.04802.pdf>` paper.
        - `ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks                    <https://arxiv.org/pdf/1809.00219.pdf>` paper.
        - `Perceptual Extreme Super Resolution Network with Receptive Field Block               <https://arxiv.org/pdf/2005.12597.pdf>` paper.
    """

    def __init__(self) -> None:
        super(PerceptualLoss, self).__init__()
        # 加载基于BreaKHis数据集训练的VGG19模型.
        vgg19 = models.vgg19(pretrained=False, num_classes=2).eval()
        vgg19.load_state_dict(torch.load("vgg19-bb7111ed.pth", map_location=lambda storage, loc: storage))
        # 提取VGG19模型中第35层输出作为内容损失.
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:35])
        # 冻结模型参数.
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

        # 对输入数据的预处理方式. 这是BreaKHis数据集预处理方式.
        self.register_buffer("mean", torch.Tensor([0.787, 0.626, 0.764]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.106, 0.139, 0.091]).view(1, 3, 1, 1))
        self.resize = transforms.Resize([224, 224])

    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        # 标准化操作.
        sr = (sr - self.mean) / self.std
        hr = (hr - self.mean) / self.std

        # 缩放图像至VGG19模型输入大小.
        sr = self.resize(sr)
        hr = self.resize(hr)

        # 求两张图像之间的特征图差异.
        loss = F.l1_loss(self.feature_extractor(sr), self.feature_extractor(hr))

        return loss
