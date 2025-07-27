import os
import jittor as jt
from jittor import nn
import numpy as np
import time
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def validate(val_loader, distiller):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    start_time = time.time()
    for idx, (image, target) in enumerate(val_loader):
        image = image.float32()
        target = target.int32()

        output = distiller(image=image)
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]

        losses.update(float(loss), batch_size)
        top1.update(float(acc1[0]), batch_size)
        top5.update(float(acc5[0]), batch_size)

        batch_time.update(time.time() - start_time)
        start_time = time.time()

        msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
            top1=top1, top5=top5
        )
        pbar.set_description(log_msg(msg, "EVAL"))
        pbar.update()
    pbar.close()
    return top1.avg, top5.avg, losses.avg


def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def adjust_learning_rate(epoch, cfg, optimizer):
    steps = np.sum(epoch > np.asarray(cfg.SOLVER.LR_DECAY_STAGES))
    if steps > 0:
        new_lr = cfg.SOLVER.LR * (cfg.SOLVER.LR_DECAY_RATE ** steps)
        optimizer.lr = new_lr
        return new_lr
    return cfg.SOLVER.LR


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.shape[0]

    _, pred = jt.topk(output, k=maxk, dim=1)
    pred = pred.transpose(0, 1)  # shape: [maxk, batch]
    correct = pred == target.unsqueeze(0).expand_as(pred)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).sum().float32()
        res.append(correct_k * (100.0 / batch_size))
    return res


def save_checkpoint(obj, path):
    jt.save(obj, path)


def load_checkpoint(path):
    return jt.load(path)
