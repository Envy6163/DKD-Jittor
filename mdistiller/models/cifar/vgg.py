import jittor as jt
import jittor.nn as nn
import math

__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]

model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
}

class VGG(nn.Module):
    def __init__(self, cfg, batch_norm=False, num_classes=1000):
        super(VGG, self).__init__()
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.Pool(2, stride=2, op="maximum")
        self.pool1 = nn.Pool(2, stride=2, op="maximum")
        self.pool2 = nn.Pool(2, stride=2, op="maximum")
        self.pool3 = nn.Pool(2, stride=2, op="maximum")
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()
        self.stage_channels = [c[-1] for c in cfg]

    def get_feat_modules(self):
        feat_m = nn.ModuleList()
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def get_stage_channels(self):
        return self.stage_channels

    def execute(self, x):
        h = x.shape[2]
        x = nn.relu(self.block0(x))
        f0 = x
        x = self.pool0(x)
        x = self.block1(x)
        f1_pre = x
        x = nn.relu(x)
        f1 = x
        x = self.pool1(x)
        x = self.block2(x)
        f2_pre = x
        x = nn.relu(x)
        f2 = x
        x = self.pool2(x)
        x = self.block3(x)
        f3_pre = x
        x = nn.relu(x)
        f3 = x
        if h == 64:
            x = self.pool3(x)
        x = self.block4(x)
        f4_pre = x
        x = nn.relu(x)
        f4 = x
        x = self.pool4(x)
        x = x.view(x.shape[0], -1)
        f5 = x
        x = self.classifier(x)

        feats = {
            "feats": [f0, f1, f2, f3, f4],
            "preact_feats": [f0, f1_pre, f2_pre, f3_pre, f4_pre],
            "pooled_feat": f5,
        }

        return x, feats

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == "M":
                layers += [nn.Pool(2, stride=2, op="maximum")]
            else:
                conv2d = nn.Conv(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm(v), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]
                in_channels = v
        layers = layers[:-1]  # remove last relu
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv):
                n = m.weight.shape[2] * m.weight.shape[3] * m.weight.shape[1]
                std = math.sqrt(2.0 / n)
                nn.init.gauss_(m.weight, 0, std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.gauss_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


cfg = {
    "A": [[64], [128], [256, 256], [512, 512], [512, 512]],
    "B": [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    "D": [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    "E": [
        [64, 64],
        [128, 128],
        [256, 256, 256, 256],
        [512, 512, 512, 512],
        [512, 512, 512, 512],
    ],
    "S": [[64], [128], [256], [512], [512]],
}

def vgg8(**kwargs):
    """VGG 8-layer model (configuration "S")"""
    model = VGG(cfg["S"], **kwargs)
    return model

def vgg8_bn(**kwargs):
    """VGG 8-layer model (configuration "S") with batch normalization"""
    model = VGG(cfg["S"], batch_norm=True, **kwargs)
    return model

def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")"""
    model = VGG(cfg["A"], **kwargs)
    return model

def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(cfg["A"], batch_norm=True, **kwargs)
    return model

def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")"""
    model = VGG(cfg["B"], **kwargs)
    return model

def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(cfg["B"], batch_norm=True, **kwargs)
    return model

def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")"""
    model = VGG(cfg["D"], **kwargs)
    return model

def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(cfg["D"], batch_norm=True, **kwargs)
    return model

def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")"""
    model = VGG(cfg["E"], **kwargs)
    return model

def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration "E") with batch normalization"""
    model = VGG(cfg["E"], batch_norm=True, **kwargs)
    return model

if __name__ == "__main__":
    jt.flags.use_cuda = 1

    x = jt.randn((2, 3, 32, 32))
    net = vgg19_bn(num_classes=100)
    logit, feats = net(x)

    for f in feats["feats"]:
        print(f.shape, f.min().item())
    print(logit.shape)

