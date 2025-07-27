import math
import jittor as jt
from jittor import nn
from typing import List, Optional

def check_in(t, l):
    for i in l:
        if t is i:
            return True
    return False

def dot(params: List[jt.Var],
        d_p_list: List[jt.Var],
        momentum_buffer_list: List[Optional[jt.Var]],
        kd_grad_buffer: List[Optional[jt.Var]],
        kd_momentum_buffer: List[Optional[jt.Var]],
        kd_params: List[jt.Var],
        *,
        weight_decay: float,
        momentum: float,
        momentum_kd: float,
        lr: float,
        dampening: float):

    for i, param in enumerate(params):
        d_p = d_p_list[i]
        if weight_decay != 0:
            d_p = d_p + param * weight_decay

        if momentum != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = d_p.clone().detach()
                momentum_buffer_list[i] = buf
            elif check_in(param, kd_params):
                buf = buf * momentum + d_p * (1 - dampening)
            else:
                buf = buf * ((momentum_kd + momentum) / 2.) + d_p * (1 - dampening)
            d_p = buf

        param -= lr * d_p

    for i, (d_p, buf, p) in enumerate(zip(kd_grad_buffer, kd_momentum_buffer, kd_params)):
        if buf is None:
            buf = d_p.clone().detach()
            kd_momentum_buffer[i] = buf
        elif check_in(p, params):
            buf = buf * momentum_kd + d_p * (1 - dampening)
        else:
            if weight_decay != 0:
                d_p = d_p + p * weight_decay
            buf = buf * ((momentum_kd + momentum) / 2.) + d_p * (1 - dampening)
        p -= lr * buf


class DistillationOrientedTrainer:
    """
    Distillation-Oriented Trainer for Jittor.

    Usage:
        ...
        optimizer = DistillationOrientedTrainer(model.parameters(), lr=0.1, ...)
        optimizer.zero_grad()
        kd_loss.backward()
        optimizer.step_kd()  # Register KD gradients and momentum
        optimizer.zero_grad()
        task_loss.backward()
        optimizer.step()  # Update parameters with both KD and task loss
    """

    def __init__(self, params, lr, momentum=0, momentum_kd=0, dampening=0, weight_decay=0):
        self.param_groups = list(params)
        self.lr = lr
        self.momentum = momentum
        self.momentum_kd = momentum_kd
        self.dampening = dampening
        self.weight_decay = weight_decay

        # Buffers for momentum
        self.state = {}
        self.kd_grad_buffer = []
        self.kd_grad_params = []
        self.kd_momentum_buffer = []

        for p in self.param_groups:
            self.state[p] = {}

    def zero_grad(self):
        for p in self.param_groups:
            if p.grad is not None:
                p.grad.stop_grad()
                p.grad.assign(jt.zeros_like(p))

    def step_kd(self):
        """Store current gradients as KD gradients, and init KD momentum buffers"""
        self.kd_grad_buffer = []
        self.kd_grad_params = []
        self.kd_momentum_buffer = []

        for p in self.param_groups:
            if p.grad is not None:
                self.kd_grad_params.append(p)
                self.kd_grad_buffer.append(p.grad.clone())
                buf = self.state[p].get("momentum_kd_buffer", None)
                self.kd_momentum_buffer.append(buf)

    def step(self):
        """Apply main update using task loss gradients + KD gradients if set"""
        params_with_grad = []
        d_p_list = []
        momentum_buffer_list = []

        for p in self.param_groups:
            if p.grad is not None:
                params_with_grad.append(p)
                d_p_list.append(p.grad)
                buf = self.state[p].get("momentum_buffer", None)
                momentum_buffer_list.append(buf)

        dot(
            params_with_grad,
            d_p_list,
            momentum_buffer_list,
            self.kd_grad_buffer,
            self.kd_momentum_buffer,
            self.kd_grad_params,
            weight_decay=self.weight_decay,
            momentum=self.momentum,
            momentum_kd=self.momentum_kd,
            lr=self.lr,
            dampening=self.dampening,
        )

        # Update momentum buffers
        for p, buf in zip(params_with_grad, momentum_buffer_list):
            self.state[p]["momentum_buffer"] = buf
        for p, buf in zip(self.kd_grad_params, self.kd_momentum_buffer):
            self.state[p]["momentum_kd_buffer"] = buf

        # Reset KD buffers
        self.kd_grad_buffer = []
        self.kd_grad_params = []
        self.kd_momentum_buffer = []
