import argparse
import jittor as jt
from jittor import nn
import time

jt.flags.use_cuda = 1

from mdistiller.distillers import Vanilla
from mdistiller.models import cifar_model_dict, imagenet_model_dict, tiny_imagenet_model_dict
from mdistiller.dataset import get_dataset
from mdistiller.dataset.imagenet import get_imagenet_val_loader
from mdistiller.engine.utils import load_checkpoint, validate
from mdistiller.engine.cfg import CFG as cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="")
    parser.add_argument("-c", "--ckpt", type=str, default="pretrain")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="cifar100",
        choices=["cifar100", "imagenet", "tiny_imagenet"],
    )
    parser.add_argument("-bs", "--batch-size", type=int, default=64)
    args = parser.parse_args()

    cfg.DATASET.TYPE = args.dataset
    cfg.DATASET.TEST.BATCH_SIZE = args.batch_size

    if args.dataset == "imagenet":
        val_loader = get_imagenet_val_loader(args.batch_size)
        if args.ckpt == "pretrain":
            model = imagenet_model_dict[args.model](pretrained=True)
        else:
            model = imagenet_model_dict[args.model](pretrained=False)
            model.load_parameters(load_checkpoint(args.ckpt)["model"])
    elif args.dataset in ("cifar100", "tiny_imagenet"):
        train_loader, val_loader, num_data, num_classes = get_dataset(cfg)
        model_dict = tiny_imagenet_model_dict if args.dataset == "tiny_imagenet" else cifar_model_dict
        model, pretrain_model_path = model_dict[args.model]
        model = model(num_classes=num_classes)
        ckpt = pretrain_model_path if args.ckpt == "pretrain" else args.ckpt
        model.load_parameters(load_checkpoint(ckpt)["model"])

    model = Vanilla(model)

    start_time = time.time()
    
    test_acc, test_acc_top5, test_loss = validate(val_loader, model)

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference Time: {inference_time:.2f} seconds")

