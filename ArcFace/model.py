import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block: 通道注意力机制
    """

    def __init__(self, channel, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    空间注意力模块：让模型聚焦人脸关键区域（眼睛、鼻子、嘴巴）
    避免模型被背景或非关键区域干扰
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class IRBlock(nn.Module):
    """
    Inverted Residual Block: ArcFace 官方推荐的基础单元，集成 SE 通道注意力与空间注意力
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, use_sa=False):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.use_se = use_se
        self.use_sa = use_sa
        if self.use_se:
            self.se = SEBlock(planes, reduction=8)
        if self.use_sa:
            self.sa = SpatialAttention()

    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        if self.use_sa:
            sa_map = self.sa(out)
            out = out * sa_map
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return out


class FaceEncoder(nn.Module):
    """
    人脸识别骨干网络：基于 IR-ResNet 架构，输出 512 维 L2 归一化特征向量。
    使用 1×1 Conv + GAP 替代 Flatten+FC，大幅减少参数量以抑制过拟合。
    """

    def __init__(self, block=IRBlock, layers=[3, 4, 14, 3], embedding_dim=512, use_se=True, use_sa=True):
        super(FaceEncoder, self).__init__()
        self.inplanes = 64
        self.use_se = use_se
        self.use_sa = use_sa
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()

        # 构建残差层
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2, use_se=use_se, use_sa=use_sa)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_se=use_se, use_sa=use_sa)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_se=use_se, use_sa=use_sa)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_se=use_se, use_sa=use_sa)

        # 1×1 Conv 降维 → GAP → Dropout → L2 Norm
        self.bn2 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(embedding_dim)
        self.dropout = nn.Dropout(p=0.4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, use_sa=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se, use_sa=use_sa))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=use_se, use_sa=use_sa))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.conv5(x)
        x = self.bn3(x)
        x = self.prelu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        # ArcFace 需要 L2 归一化的特征向量
        x = F.normalize(x, p=2, dim=1)
        return x


class ArcMarginProduct(nn.Module):
    """
    ArcFace Loss 实现：通过加法角度间隔惩罚，增强类内紧凑性和类间差异性。
    默认 s=30.0, m=0.50 — 小数据集优化配置（原始 ArcFace 论文 s=64 适用于百万级数据）
    """

    def __init__(self, in_features=512, out_features=1000, s=64.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output