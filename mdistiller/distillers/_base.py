import jittor as jt
from jittor import nn

class Distiller(nn.Module):  # 抽象的蒸馏器基类
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher

    def eval(self):
        return self.train(False)

    def train(self, mode=True):
        """student 按 mode 切换，teacher 永远 eval 并 stop_grad。"""
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        #super().train()
        # 让全部子模块跟随mode
        if mode:
            self.student.train()
        else:
            self.student.eval()

        # 把teacher强制eval+stop_grad
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.stop_grad() # 冻结梯度
        return self

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def get_extra_parameters(self):
        return 0

    def forward_train(self, **kwargs):
        raise NotImplementedError()

    def forward_test(self, image):
        return self.student(image)[0]

    def execute(self, **kwargs):  # execute() 取代 forward()
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])


class Vanilla(nn.Module):  # 无蒸馏，仅监督训练
    def __init__(self, student):
        super().__init__()
        self.student = student
        self.ce = nn.CrossEntropyLoss()

    def get_learnable_parameters(self):
        return [v for k, v in self.student.named_parameters()]

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        loss = self.ce(logits_student, target)
        return logits_student, {"ce": loss}

    def execute(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(kwargs["image"])

    def forward_test(self, image):
        return self.student(image)[0]
