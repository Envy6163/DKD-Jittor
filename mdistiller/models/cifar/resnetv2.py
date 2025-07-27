import jittor as jt
import jittor.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm(self.expansion * planes),
            )

    def execute(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = nn.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super().__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(planes)
        self.conv2 = nn.Conv(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm(planes)
        self.conv3 = nn.Conv(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm(self.expansion * planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm(self.expansion * planes),
            )

    def execute(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        out = nn.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = nn.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, zero_init_residual=False):
        super().__init__()
        self.in_planes = 64
        self.block = block

        self.conv1 = nn.Conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        self.stage_channels = [256, 512, 1024, 2048]

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    m.bn3.weight.data.fill_(0)
                elif isinstance(m, BasicBlock):
                    m.bn2.weight.data.fill_(0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride, i == num_blocks - 1))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_feat_modules(self):
        return nn.ModuleList([
            self.conv1,
            self.bn1,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4
        ])

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            return [
                self.layer1[-1].bn3,
                self.layer2[-1].bn3,
                self.layer3[-1].bn3,
                self.layer4[-1].bn3,
            ]
        elif isinstance(self.layer1[0], BasicBlock):
            return [
                self.layer1[-1].bn2,
                self.layer2[-1].bn2,
                self.layer3[-1].bn2,
                self.layer4[-1].bn2,
            ]
        else:
            raise NotImplementedError("ResNet unknown block error !!!")

    def get_stage_channels(self):
        return self.stage_channels

    def encode(self, x, idx, preact=False):
        if idx == -1:
            out, pre = self.layer4(nn.relu(x))
        elif idx == -2:
            out, pre = self.layer3(nn.relu(x))
        elif idx == -3:
            out, pre = self.layer2(nn.relu(x))
        else:
            raise NotImplementedError()
        return pre

    def execute(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        f0 = out
        out, f1_pre = self.layer1(out)
        f1 = out
        out, f2_pre = self.layer2(out)
        f2 = out
        out, f3_pre = self.layer3(out)
        f3 = out
        out, f4_pre = self.layer4(out)
        f4 = out
        out = self.avgpool(out)
        avg = out.reshape(out.shape[0], -1)
        out = self.linear(avg)

        feats = {
            "feats": [f0, f1, f2, f3, f4],
            "preact_feats": [f0, f1_pre, f2_pre, f3_pre, f4_pre],
            "pooled_feat": avg
        }
        return out, feats

def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

if __name__ == "__main__":
    jt.flags.use_cuda = 1

    net = ResNet18(num_classes=100)
    x = jt.randn(2, 3, 32, 32)
    logit, feats = net(x)

    for f in feats["feats"]:
        print(f.shape, f.min().item())
    print(logit.shape)

