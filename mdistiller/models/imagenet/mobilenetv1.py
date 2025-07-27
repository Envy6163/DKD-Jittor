import jittor as jt
import jittor.nn as nn

class MobileNetV1(nn.Module):
    def __init__(self, **kwargs):
        super(MobileNetV1, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv(inp, oup, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm(oup),
                nn.ReLU()
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv(inp, inp, 3, stride=stride, padding=1, groups=inp, bias=False),
                nn.BatchNorm(inp),
                nn.ReLU(),
                nn.Conv(inp, oup, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm(oup),
                nn.ReLU()
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.Pool(kernel_size=7, op='mean')  # replaces AvgPool2d(7)
        )

        self.fc = nn.Linear(1024, 1000)

    def execute(self, x, is_feat=False):
        feat1 = self.model[3][:-1](self.model[0:3](x))
        feat2 = self.model[5][:-1](self.model[4:5](nn.relu(feat1)))
        feat3 = self.model[11][:-1](self.model[6:11](nn.relu(feat2)))
        feat4 = self.model[13][:-1](self.model[12:13](nn.relu(feat3)))
        feat5 = self.model[14](nn.relu(feat4))
        avg = feat5.reshape(feat5.shape[0], -1)
        out = self.fc(avg)

        feats = {
            "pooled_feat": avg,
            "feats": [nn.relu(feat1), nn.relu(feat2), nn.relu(feat3), nn.relu(feat4)],
            "preact_feats": [feat1, feat2, feat3, feat4]
        }
        return out, feats

    def get_bn_before_relu(self):
        bn1 = self.model[3][-2]
        bn2 = self.model[5][-2]
        bn3 = self.model[11][-2]
        bn4 = self.model[13][-2]
        return [bn1, bn2, bn3, bn4]

    def get_stage_channels(self):
        return [128, 256, 512, 1024]
