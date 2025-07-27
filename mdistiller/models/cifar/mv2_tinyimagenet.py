import jittor as jt
import jittor.nn as nn

class LinearBottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.residual = nn.Sequential(
            nn.Conv(in_channels, in_channels * t, 1),
            nn.BatchNorm(in_channels * t),
            nn.ReLU6(),

            nn.Conv(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm(in_channels * t),
            nn.ReLU6(),

            nn.Conv(in_channels * t, out_channels, 1),
            nn.BatchNorm(out_channels)
        )

    def execute(self, x):
        residual = self.residual(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            residual = residual + x
        return residual

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv(3, 32, 1, padding=1),
            nn.BatchNorm(32),
            nn.ReLU6()
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv(320, 1280, 1),
            nn.BatchNorm(1280),
            nn.ReLU6()
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv2 = nn.Conv(1280, num_classes, 1)

    def execute(self, x):
        x = self.pre(x)
        f0 = x
        x = self.stage1(x)
        x = self.stage2(x)
        f1 = x
        x = self.stage3(x)
        f2 = x
        x = self.stage4(x)
        f3 = x
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        f4 = x
        x = self.pool(x)
        avg = x
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)

        feats = {
            "feats": [f0, f1, f2, f3, f4],
            "pooled_feat": avg
        }
        return x, feats

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):
        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))
        for _ in range(repeat - 1):
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
        return nn.Sequential(*layers)

def mobilenetv2_tinyimagenet(**kwargs):
    return MobileNetV2(**kwargs)
