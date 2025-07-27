from __future__ import absolute_import
import jittor as jt
from jittor import nn

__all__ = ["resnet"]

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv(
        in_planes, out_planes,
        kernel_size=3, stride=stride, padding=1, bias=False
    )

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm(planes)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        preact = out
        out = self.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, is_last=False):
        super().__init__()
        self.is_last = is_last
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
        preact = out
        out = self.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out

class ResNet(nn.Module):
    def __init__(self, depth, num_filters, block_name="BasicBlock", num_classes=10):
        super().__init__()

        if block_name.lower() == "basicblock":
            assert (depth - 2) % 6 == 0, "When use basicblock, depth should be 6n+2"
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == "bottleneck":
            assert (depth - 2) % 9 == 0, "When use bottleneck, depth should be 9n+2"
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError("block_name should be Basicblock or Bottleneck")

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv(3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(num_filters[0])
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        
        #self.avgpool = nn.Pool(8, stride=1, op="mean")  # 等效于 AvgPool2d(8)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 修改为自适应池化
        
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)
        self.stage_channels = num_filters

        for m in self.modules():
            if isinstance(m, nn.Conv):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, is_last=(blocks == 1)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=(i == blocks - 1)))

        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([
            self.conv1,
            self.bn1,
            self.relu,
            self.layer1,
            self.layer2,
            self.layer3,
        ])
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            raise NotImplementedError("ResNet unknown block error !!!")

        return [bn1, bn2, bn3]

    def get_stage_channels(self):
        return self.stage_channels

    def execute(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        f0 = x

        x, f1_pre = self.layer1(x)
        f1 = x
        x, f2_pre = self.layer2(x)
        f2 = x
        x, f3_pre = self.layer3(x)
        f3 = x

        x = self.avgpool(x)
        avg = x.reshape((x.shape[0], -1))
        out = self.fc(avg)

        feats = {
            "feats": [f0, f1, f2, f3],
            "preact_feats": [f0, f1_pre, f2_pre, f3_pre],
            "pooled_feat": avg,
        }

        return out, feats



def resnet8(**kwargs):
    return ResNet(8, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet14(**kwargs):
    return ResNet(14, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet20(**kwargs):
    return ResNet(20, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet32(**kwargs):
    return ResNet(32, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet44(**kwargs):
    return ResNet(44, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet56(**kwargs):
    return ResNet(56, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet110(**kwargs):
    return ResNet(110, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet8x4(**kwargs):
    return ResNet(8, [32, 64, 128, 256], "basicblock", **kwargs)


def resnet32x4(**kwargs):
    return ResNet(32, [32, 64, 128, 256], "basicblock", **kwargs)


if __name__ == "__main__":
    #import jittor as jt
    jt.flags.use_cuda = 1

    x = jt.randn(2, 3, 32, 32)
    net = resnet8x4(num_classes=20)
    logit, feats = net(x)

    for f in feats["feats"]:
        print(f.shape, float(f.min()))
    print(logit.shape)

