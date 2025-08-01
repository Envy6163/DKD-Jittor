import jittor as jt
import jittor.nn as nn
import math


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def execute(self, x):
        N, C, H, W = x.shape
        g = self.groups
        return x.reshape(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.stride = stride

        mid_planes = int(out_planes / 4)
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes, bias=False)
        self.bn2 = nn.BatchNorm(mid_planes)
        self.conv3 = nn.Conv(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm(out_planes)

        self.shortcut = nn.Identity()
        if stride == 2:
            self.shortcut = nn.Pool(kernel_size=3, stride=2, padding=1, op='mean')

    def execute(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = nn.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        preact = jt.concat([out, res], dim=1) if self.stride == 2 else out + res
        out = nn.relu(preact)
        return (out, preact) if self.is_last else out


class ShuffleNet(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(ShuffleNet, self).__init__()
        out_planes = cfg["out_planes"]
        num_blocks = cfg["num_blocks"]
        groups = cfg["groups"]

        self.conv1 = nn.Conv(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        self.linear = nn.Linear(out_planes[2], num_classes)

    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(
                self.in_planes,
                out_planes - cat_planes,
                stride=stride,
                groups=groups,
                is_last=(i == num_blocks - 1)
            ))
            self.in_planes = out_planes
        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList()
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def get_bn_before_relu(self):
        raise NotImplementedError(
            'ShuffleNet currently is not supported for "Overhaul" teacher'
        )

    def execute(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        f0 = out
        out, f1_pre = self.layer1(out)
        f1 = out
        out, f2_pre = self.layer2(out)
        f2 = out
        out, f3_pre = self.layer3(out)
        f3 = out
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.reshape(out.shape[0], -1)
        f4 = out
        out = self.linear(out)

        feats = {
            "feats": [f0, f1, f2, f3],
            "preact_feats": [f0, f1_pre, f2_pre, f3_pre],
            "pooled_feat": f4,
        }
        return out, feats


def ShuffleV1(**kwargs):
    cfg = {"out_planes": [240, 480, 960], "num_blocks": [4, 8, 4], "groups": 3}
    return ShuffleNet(cfg, **kwargs)


if __name__ == "__main__":
    x = jt.randn(2, 3, 32, 32)
    net = ShuffleV1(num_classes=100)
    import time

    a = time.time()
    logit, feats = net(x)
    b = time.time()
    print(b - a)
    for f in feats["feats"]:
        print(f.shape, f.min().item())
    print(logit.shape)
