import torch
import torch.nn as nn
import torch.nn.functional as F

# 将学生模型 (student) 的特征图调整为与教师模型 (teacher) 特征图相同的尺寸，以便进行特征层面的知识蒸馏
class ConvReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_shape, t_shape, use_relu=True): #学生特征的形状，教师特征的形状，是否在输出端加 ReLU
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H: # 下采样（学生特征图比教师大一倍）
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H: # 上采样（学生特征图比教师小一倍）
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H: # 精确卷积裁剪到目标大小
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H)) #不支持教师特征比学生还小的情况
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        else:
            return self.bn(x)

# 自动获取学生模型和教师模型在各层输出的特征图尺寸
def get_feat_shapes(student, teacher, input_size):
    data = torch.randn(1, 3, *input_size)
    with torch.no_grad():
        _, feat_s = student(data)
        _, feat_t = teacher(data)
    feat_s_shapes = [f.shape for f in feat_s["feats"]]
    feat_t_shapes = [f.shape for f in feat_t["feats"]]
    return feat_s_shapes, feat_t_shapes
