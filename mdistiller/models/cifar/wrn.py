import math
import jittor as jt
from jittor import nn

__all__ = ["wrn"]

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm(in_planes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm(out_planes)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.droprate = dropRate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = None
        if not self.equalInOut:
            self.convShortcut = nn.Conv(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )

    def execute(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = nn.dropout(out, p=self.droprate, is_training=self.training)
        out = self.conv2(out)
        shortcut = x if self.equalInOut else self.convShortcut(x)
        return out + shortcut


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, dropRate
        )

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(
                    in_planes if i == 0 else out_planes,
                    out_planes,
                    stride if i == 0 else 1,
                    dropRate,
                )
            )
        return nn.Sequential(*layers)

    def execute(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6
        block = BasicBlock

        # First conv layer
        self.conv1 = nn.Conv(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)

        # Network blocks
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        # Final layers
        self.bn1 = nn.BatchNorm(nChannels[3])
        self.relu = nn.ReLU()
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        self.stage_channels = nChannels

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv):
                n = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
                std = math.sqrt(2.0 / n)
                nn.init.gauss_(m.weight, 0, std)
            elif isinstance(m, nn.BatchNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0.0)


    def get_feat_modules(self):
        feat_m = nn.ModuleList([
            self.conv1,
            self.block1,
            self.block2,
            self.block3
        ])
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block2.layer[0].bn1
        bn2 = self.block3.layer[0].bn1
        bn3 = self.bn1
        return [bn1, bn2, bn3]

    def get_stage_channels(self):
        return self.stage_channels

    def execute(self, x):
        out = self.conv1(x)
        f0 = out
        out = self.block1(out)
        f1 = out
        out = self.block2(out)
        f2 = out
        out = self.block3(out)
        f3 = out
        out = self.relu(self.bn1(out))
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.reshape([out.shape[0], -1])
        f4 = out
        out = self.fc(out)
        # 预激活特征
        f1_pre = self.block2.layer[0].bn1(f1)
        f2_pre = self.block3.layer[0].bn1(f2)
        f3_pre = self.bn1(f3)

        feats = {
            "feats": [f0, f1, f2, f3],
            "preact_feats": [f0, f1_pre, f2_pre, f3_pre],
            "pooled_feat": f4,
        }
        return out, feats

def wrn(**kwargs):
    """
    Constructs a Wide Residual Network.
    """
    model = WideResNet(**kwargs)
    return model

def wrn_40_2(**kwargs):
    return WideResNet(depth=40, widen_factor=2, **kwargs)

def wrn_40_1(**kwargs):
    return WideResNet(depth=40, widen_factor=1, **kwargs)

def wrn_16_2(**kwargs):
    return WideResNet(depth=16, widen_factor=2, **kwargs)

def wrn_16_1(**kwargs):
    return WideResNet(depth=16, widen_factor=1, **kwargs)

if __name__ == "__main__":
    jt.flags.use_cuda = 1

    x = jt.randn(2, 3, 32, 32)
    net = wrn_40_2(num_classes=100)
    logit, feats = net(x)

    for f in feats["feats"]:
        print(f.shape, f.min().item())
    print(logit.shape)

