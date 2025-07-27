import os
import argparse
import jittor as jt
from jittor import nn

jt.flags.use_cuda = 1

from mdistiller.models import cifar_model_dict, imagenet_model_dict, tiny_imagenet_model_dict
from mdistiller.distillers import distiller_dict  # 各种知识蒸馏方法（如KD, CRD等）的注册表
from mdistiller.dataset import get_dataset  # 获取数据集
from mdistiller.engine.utils import load_checkpoint, log_msg
from mdistiller.engine.cfg import CFG as cfg  # 配置文件
from mdistiller.engine.cfg import show_cfg
from mdistiller.engine import trainer_dict  # 训练器

def main(cfg, resume, opts):
    experiment_name = cfg.EXPERIMENT.NAME
    if experiment_name == "":
        experiment_name = cfg.EXPERIMENT.TAG
    tags = cfg.EXPERIMENT.TAG.split(",")
    if opts:
        addtional_tags = ["{}:{}".format(k, v) for k, v in zip(opts[::2], opts[1::2])]
        tags += addtional_tags
        experiment_name += ",".join(addtional_tags)
    experiment_name = os.path.join(cfg.EXPERIMENT.PROJECT, experiment_name)

    if cfg.LOG.WANDB:
        try:
            import wandb
            wandb.init(project=cfg.EXPERIMENT.PROJECT, name=experiment_name, tags=tags)
        except:
            print(log_msg("Failed to use WANDB", "INFO"))
            cfg.LOG.WANDB = False

    # 打印配置参数
    show_cfg(cfg)

    # 初始化数据集和模型
    train_loader, val_loader, num_data, num_classes = get_dataset(cfg)

    # 原始训练（无蒸馏）
    if cfg.DISTILLER.TYPE == "NONE":
        if cfg.DATASET.TYPE == "imagenet":
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        elif cfg.DATASET.TYPE == "tiny_imagenet":
            model_student = tiny_imagenet_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
        else:
            model_student = cifar_model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)
        distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student)

    else:
        print(log_msg("Loading teacher model", "INFO"))
        if cfg.DATASET.TYPE == "imagenet":
            model_teacher = imagenet_model_dict[cfg.DISTILLER.TEACHER](pretrained=True)
            model_student = imagenet_model_dict[cfg.DISTILLER.STUDENT](pretrained=False)
        else:
            model_dict = tiny_imagenet_model_dict if cfg.DATASET.TYPE == "tiny_imagenet" else cifar_model_dict
            net, pretrain_model_path = model_dict[cfg.DISTILLER.TEACHER]
            assert pretrain_model_path is not None, f"No pretrain model for teacher {cfg.DISTILLER.TEACHER}"
            model_teacher = net(num_classes=num_classes)
            model_teacher.load_parameters(load_checkpoint(pretrain_model_path)["model"])
            model_student = model_dict[cfg.DISTILLER.STUDENT][0](num_classes=num_classes)

        if cfg.DISTILLER.TYPE == "CRD":
            distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student, model_teacher, cfg, num_data)
        else:
            distiller = distiller_dict[cfg.DISTILLER.TYPE](model_student, model_teacher, cfg)

    # Jittor 不需要 DataParallel

    if cfg.DISTILLER.TYPE != "NONE":
        print(log_msg(
            "Extra parameters of {}: {}\033[0m".format(
                cfg.DISTILLER.TYPE,
                distiller.get_extra_parameters() if hasattr(distiller, "get_extra_parameters") else "N/A"
            ), "INFO"
        ))

    # 训练入口
    trainer = trainer_dict[cfg.SOLVER.TRAINER](
        experiment_name, distiller, train_loader, val_loader, cfg
    )
    trainer.train(resume=resume)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument("--cfg", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    main(cfg, args.resume, args.opts)
