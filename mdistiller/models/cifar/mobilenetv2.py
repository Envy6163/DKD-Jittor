import jittor as jt
import jittor.nn as nn
import math

__all__ = ["mobilenetv2_T_w", "mobile_half"]

BN = None

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm(oup),
        nn.ReLU()
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm(oup),
        nn.ReLU()
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.blockname = None

        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv(inp, inp * expand_ratio, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm(inp * expand_ratio),
            nn.ReLU(),
            # dw
            nn.Conv(
                inp * expand_ratio,
                inp * expand_ratio,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=inp * expand_ratio,
                bias=False,
            ),
            nn.BatchNorm(inp * expand_ratio),
            nn.ReLU(),
            # pw-linear
            nn.Conv(inp * expand_ratio, oup, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm(oup),
        )
        self.names = ["0", "1", "2", "3", "4", "5", "6", "7"]

    def execute(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    """mobilenetV2"""

    def __init__(self, T, feature_dim, input_size=32, width_mult=1.0, remove_avg=False):
        super(MobileNetV2, self).__init__()
        self.remove_avg = remove_avg

        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [T, 24, 2, 1],
            [T, 32, 3, 2],
            [T, 64, 4, 2],
            [T, 96, 3, 1],
            [T, 160, 3, 2],
            [T, 320, 1, 1],
        ]

        assert input_size % 32 == 0
        input_channel = int(32 * width_mult)
        self.conv1 = conv_bn(3, input_channel, 2)

        self.blocks = nn.ModuleList()
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            layers = []
            strides = [s] + [1] * (n - 1)
            for stride in strides:
                layers.append(InvertedResidual(input_channel, output_channel, stride, t))
                input_channel = output_channel
            self.blocks.append(nn.Sequential(*layers))

        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.conv2 = conv_1x1_bn(input_channel, self.last_channel)

        self.classifier = nn.Sequential(
            nn.Linear(self.last_channel, feature_dim),
        )

        H = input_size // (32 // 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._initialize_weights()
        print(T, width_mult)
        self.stage_channels = [32, 24, 32, 96, 320]
        self.stage_channels = [int(c * width_mult) for c in self.stage_channels]

    def get_bn_before_relu(self):
        bn1 = self.blocks[1][-1].conv[-1]
        bn2 = self.blocks[2][-1].conv[-1]
        bn3 = self.blocks[4][-1].conv[-1]
        bn4 = self.blocks[6][-1].conv[-1]
        return [bn1, bn2, bn3, bn4]

    def get_feat_modules(self):
        feat_m = nn.ModuleList()
        feat_m.append(self.conv1)
        feat_m.append(self.blocks)
        return feat_m

    def get_stage_channels(self):
        return self.stage_channels

    def execute(self, x):
        out = self.conv1(x)
        f0 = out
        out = self.blocks[0](out)
        out = self.blocks[1](out)
        f1 = out
        out = self.blocks[2](out)
        f2 = out
        out = self.blocks[3](out)
        out = self.blocks[4](out)
        f3 = out
        out = self.blocks[5](out)
        out = self.blocks[6](out)
        f4 = out
        out = self.conv2(out)

        if not self.remove_avg:
            out = self.avgpool(out)
        out = out.reshape([out.shape[0], -1])
        avg = out
        out = self.classifier(out)

        feats = {}
        feats["feats"] = [f0, f1, f2, f3, f4]
        feats["pooled_feat"] = avg

        return out, feats

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv):
                n = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
                std = math.sqrt(2.0 / n)
                m.weight.gauss_(0, std)
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, nn.BatchNorm):
                m.weight.fill_(1.0)
                m.bias.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.gauss_(0, 0.01)
                m.bias.zero_()


def mobilenetv2_T_w(T, W, feature_dim=100):
    model = MobileNetV2(T=T, feature_dim=feature_dim, width_mult=W)
    return model

def mobile_half(num_classes):
    return mobilenetv2_T_w(6, 0.5, num_classes)

if __name__ == "__main__":
    x = jt.randn(2, 3, 32, 32)
    net = mobile_half(100)
    logit, feats = net(x)
    for f in feats["feats"]:
        print(f.shape, f.min().item())
    print(logit.shape)

