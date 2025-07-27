import jittor as jt
from jittor import nn
from ._base import Distiller        # 抽象基类

# ---------- 工具函数 ---------- #
def kl_div(log_p, p, reduction='batchmean'):

    # KL(target‖input) ，其中 log_p = log(softmax(student/T))， p = softmax(teacher/T).

    loss = p * (jt.log(p + 1e-6) - log_p)       # 1e-6 防止 log(0)
    if reduction == 'batchmean':
        return loss.sum() / log_p.shape[0]
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def _get_gt_mask(logits, target):
    # 真实类别mask
    target = target.reshape((-1,))
    mask = jt.zeros_like(logits)
    ones  = jt.ones((target.shape[0], 1))
    mask  = mask.scatter(1, target.unsqueeze(1), ones)
    return mask.float32() # shape [B,C]

def _get_other_mask(logits, target):
    # 非真实类别 mask
    target = target.reshape((-1,))
    mask   = jt.ones_like(logits)
    zeros  = jt.zeros((target.shape[0], 1))
    mask   = mask.scatter(1, target.unsqueeze(1), zeros)
    return mask.float32()

def cat_mask(t, mask1, mask2):
    """按论文把概率压缩成两维 [p_gt, p_other]."""
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    return jt.concat([t1, t2], dim=1)

def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask    = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    ps = nn.softmax(logits_student / temperature, dim=1)
    pt = nn.softmax(logits_teacher / temperature, dim=1)

    ps = cat_mask(ps, gt_mask, other_mask)
    pt = cat_mask(pt, gt_mask, other_mask)
    log_ps = jt.log(ps + 1e-6)

    tckd = kl_div(log_ps, pt, reduction='batchmean') * (temperature**2)

    # 排除目标类
    pt_part2  = nn.softmax(logits_teacher / temperature - 1000.0*gt_mask, dim=1)
    log_ps_p2 = nn.log_softmax(logits_student / temperature - 1000.0*gt_mask, dim=1)
    nckd = kl_div(log_ps_p2, pt_part2, reduction='batchmean') * (temperature**2)

    return alpha * tckd + beta * nckd

class DKD(Distiller):
    def __init__(self, student, teacher, cfg):
        super().__init__(student, teacher)
        self.ce_loss_weight = cfg.DKD.CE_WEIGHT
        self.alpha          = cfg.DKD.ALPHA
        self.beta           = cfg.DKD.BETA
        self.temperature    = cfg.DKD.T
        self.warmup         = cfg.DKD.WARMUP
        self.ce_loss_fn     = nn.CrossEntropyLoss()

    def forward_train(self, image, target, **kwargs):
        # student前向
        logits_s, _ = self.student(image)

        # teacher前向，冻结梯度
        self.teacher.eval()
        with jt.no_grad():
            logits_t, _ = self.teacher(image)

        # loss
        loss_ce  = self.ce_loss_weight * self.ce_loss_fn(logits_s, target)
        warmup   = min(kwargs["epoch"] / self.warmup, 1.0)
        loss_dkd = warmup * dkd_loss(
            logits_s, logits_t, target,
            self.alpha, self.beta, self.temperature
        )

        return logits_s, {"loss_ce": loss_ce, "loss_kd": loss_dkd}
