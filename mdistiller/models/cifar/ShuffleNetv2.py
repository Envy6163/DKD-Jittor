import jittor as jt
import jittor.nn as nn
import math

class ShuffleBlock(nn.Module):
    def __init__(self, groups=2):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def execute(self, x):
        N, C, H, W = x.shape
        g = self.groups
        return x.reshape(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)

class SplitBlock(nn.Module):
    def __init__(self, ratio):
        super(SplitBlock, self).__init__()
        self.ratio = ratio

    def execute(self, x):
        c = int(x.shape[1] * self.ratio)
        return x[:, :c, :, :], x[:, c:, :, :]

class BasicBlock(nn.Module):
    def __init__(self, in_channels, split_ratio=0.5, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.split = SplitBlock(split_ratio)
        in_channels = int(in_channels * split_ratio)
        self.conv1 = nn.Conv(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(in_channels)
        self.conv2 = nn.Conv(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.bn2 = nn.BatchNorm(in_channels)
        self.conv3 = nn.Conv(in_channels, in_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm(in_channels)
        self.shuffle = ShuffleBlock()

    def execute(self, x):
        x1, x2 = self.split(x)
        out = nn.relu(self.bn1(self.conv1(x2)))
        out = self.bn2(self.conv2(out))
        preact = self.bn3(self.conv3(out))
        out = nn.relu(preact)
        preact = jt.concat([x1, preact], dim=1)
        out = jt.concat([x1, out], dim=1)
        out = self.shuffle(out)
        return (out, preact) if self.is_last else out

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        mid_channels = out_channels // 2
        self.conv1 = nn.Conv(in_channels, in_channels, kernel_size=3, stride=2, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm(in_channels)
        self.conv2 = nn.Conv(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm(mid_channels)

        self.conv3 = nn.Conv(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm(mid_channels)
        self.conv4 = nn.Conv(mid_channels, mid_channels, kernel_size=3, stride=2, padding=1, groups=mid_channels, bias=False)
        self.bn4 = nn.BatchNorm(mid_channels)
        self.conv5 = nn.Conv(mid_channels, mid_channels, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm(mid_channels)

        self.shuffle = ShuffleBlock()

    def execute(self, x):
        out1 = self.bn1(self.conv1(x))
        out1 = nn.relu(self.bn2(self.conv2(out1)))

        out2 = nn.relu(self.bn3(self.conv3(x)))
        out2 = self.bn4(self.conv4(out2))
        out2 = nn.relu(self.bn5(self.conv5(out2)))

        out = jt.concat([out1, out2], dim=1)
        out = self.shuffle(out)
        return out

class ShuffleNetV2(nn.Module):
    def __init__(self, net_size, num_classes=10):
        super(ShuffleNetV2, self).__init__()
        out_channels = configs[net_size]["out_channels"]
        num_blocks = configs[net_size]["num_blocks"]

        self.conv1 = nn.Conv(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm(24)
        self.in_channels = 24
        self.layer1 = self._make_layer(out_channels[0], num_blocks[0])
        self.layer2 = self._make_layer(out_channels[1], num_blocks[1])
        self.layer3 = self._make_layer(out_channels[2], num_blocks[2])
        self.conv2 = nn.Conv(out_channels[2], out_channels[3], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm(out_channels[3])
        self.linear = nn.Linear(out_channels[3], num_classes)
        self.stage_channels = out_channels

    def _make_layer(self, out_channels, num_blocks):
        layers = [DownBlock(self.in_channels, out_channels)]
        for i in range(num_blocks):
            layers.append(BasicBlock(out_channels, is_last=(i == num_blocks - 1)))
            self.in_channels = out_channels
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
        raise NotImplementedError('ShuffleNetV2 currently is not supported for "Overhaul" teacher')

    def get_stage_channels(self):
        return [24] + list(self.stage_channels[:-1])

    def execute(self, x):
        out = nn.relu(self.bn1(self.conv1(x)))
        f0 = out
        out, f1_pre = self.layer1(out)
        f1 = out
        out, f2_pre = self.layer2(out)
        f2 = out
        out, f3_pre = self.layer3(out)
        f3 = out
        out = nn.relu(self.bn2(self.conv2(out)))
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = out.reshape(out.shape[0], -1)
        avg = out
        f4 = out
        out = self.linear(out)

        feats = {
            "feats": [f0, f1, f2, f3],
            "preact_feats": [f0, f1_pre, f2_pre, f3_pre],
            "pooled_feat": f4
        }
        return out, feats

# 配置映射
configs = {
    0.2: {"out_channels": (40, 80, 160, 512), "num_blocks": (3, 3, 3)},
    0.3: {"out_channels": (40, 80, 160, 512), "num_blocks": (3, 7, 3)},
    0.5: {"out_channels": (48, 96, 192, 1024), "num_blocks": (3, 7, 3)},
    1:   {"out_channels": (116, 232, 464, 1024), "num_blocks": (3, 7, 3)},
    1.5: {"out_channels": (176, 352, 704, 1024), "num_blocks": (3, 7, 3)},
    2:   {"out_channels": (224, 488, 976, 2048), "num_blocks": (3, 7, 3)},
}

def ShuffleV2(**kwargs):
    model = ShuffleNetV2(net_size=1, **kwargs)
    return model

if __name__ == "__main__":
    import time
    import jittor as jt

    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    net = ShuffleV2(num_classes=100)
    x = jt.randn(3, 3, 32, 32)

    a = time.time()
    logit, feats = net(x)
    b = time.time()
    print("Forward time:", b - a)

    for f in feats["feats"]:
        print(f.shape, f.min().item())
    print("Logits shape:", logit.shape)

