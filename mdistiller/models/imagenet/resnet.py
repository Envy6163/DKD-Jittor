import jittor as jt
import jittor.nn as nn
import math

__all__ = ["ResNet", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]

# 预训练模型链接保留为字典，仅用于参考
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv(
        in_planes, out_planes,
        kernel_size=3, stride=stride, padding=1, bias=False
    )

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        x = nn.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(planes)
        self.conv3 = nn.Conv(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm(planes * 4)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        x = nn.relu(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.Pool(kernel_size=3, stride=2, padding=1, op='maximum')

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.Pool(kernel_size=7, stride=1, op='mean')
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv):
                n = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
                std = math.sqrt(2.0 / n)
                m.weight.init_gauss(0, std)
            elif isinstance(m, nn.BatchNorm):
                m.weight.init_one()
                m.bias.init_zero()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
            bn4 = self.layer4[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
            bn4 = self.layer4[-1].bn2
        else:
            raise RuntimeError("Unknown block type in ResNet")

        return [bn2, bn3, bn4]

    def get_stage_channels(self):
        return [256, 512, 1024, 2048]

    def execute(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        stem = x
        x = self.relu(x)
        x = self.maxpool(x)

        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)

        x = self.avgpool(nn.relu(feat4))
        x = x.reshape(x.shape[0], -1)
        avg = x
        out = self.fc(x)

        feats = {
            "pooled_feat": avg,
            "feats": [
                nn.relu(stem),
                nn.relu(feat1),
                nn.relu(feat2),
                nn.relu(feat3),
                nn.relu(feat4)
            ],
            "preact_feats": [stem, feat1, feat2, feat3, feat4]
        }
        return out, feats

def resnet18(pretrained=False, **kwargs):
    """
    Constructs a ResNet-18 model.
    Args:
        pretrained (bool): Ignored. Pretrained weights not supported in Jittor.
    """
    if pretrained:
        print("Warning: pretrained weights not supported in Jittor version.")
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet34(pretrained=False, **kwargs):
    """
    Constructs a ResNet-34 model.
    """
    if pretrained:
        print("Warning: pretrained weights not supported in Jittor version.")
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def resnet50(pretrained=False, **kwargs):
    """
    Constructs a ResNet-50 model.
    """
    if pretrained:
        print("Warning: pretrained weights not supported in Jittor version.")
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def resnet101(pretrained=False, **kwargs):
    """
    Constructs a ResNet-101 model.
    """
    if pretrained:
        print("Warning: pretrained weights not supported in Jittor version.")
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def resnet152(pretrained=False, **kwargs):
    """
    Constructs a ResNet-152 model.
    """
    if pretrained:
        print("Warning: pretrained weights not supported in Jittor version.")
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

if __name__ == "__main__":
    model = resnet18(num_classes=1000)
    x = jt.randn(1, 3, 224, 224)
    logits, feats = model(x)
    print("Logits:", logits.shape)
    for i, f in enumerate(feats["feats"]):
        print(f"feat[{i}]:", f.shape)

