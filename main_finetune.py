# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import argparse
import collections.abc
import datetime
import json
import os
import sys
import time
import types
from pathlib import Path

# Fix timm import
torch_six = types.SimpleNamespace()
torch_six.container_abcs = collections.abc
sys.modules["torch._six"] = torch_six  # type: ignore

import numpy as np
import timm
import torch
import torch.backends.cudnn as cudnn
from timm.models.layers import trunc_normal_
from torch.utils.tensorboard.writer import SummaryWriter

import models_vit
import util.lr_decay as lrd
import util.misc as misc
from engine_finetune import evaluate, train_one_epoch
from util.datasets import build_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.pos_embed import interpolate_pos_embed


def get_args_parser():
    parser = argparse.ArgumentParser(
        "MAE fine-tuning for image regression", add_help=False
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )

    parser.add_argument(
        "--stack",
        action="store_true",
        default=False,
        help="Use 3D stacks of images as model inputs",
    )

    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=None,
        help="layer-wise lr decay from ELECTRA/BEiT (default: None)",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=10, metavar="N", help="epochs to warmup LR"
    )

    parser.add_argument(
        "--eps", type=float, default=1e-08, help="Optimizer EPS (default: 1e-08)"
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    )

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Finetuning params
    parser.add_argument(
        "--finetune", default="", type=str, help="finetune from checkpoint"
    )
    parser.add_argument("--task", default="", type=str, help="finetune from checkpoint")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )
    parser.add_argument(
        "--linear_probe",
        action="store_true",
        help="Finetune head only",
    )

    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="./datasets/bb_age_+0_+0",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--norm_mean",
        default=None,
        type=float,
        help="Target variable normalization mean. By default the training set mean is used.",
    )
    parser.add_argument(
        "--norm_std",
        default=None,
        type=float,
        help="Target variable normalization std. By default the training set mean is used.",
    )

    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./output_dir", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument(
        "--reset_epoch",
        default=False,
        action="store_true",
        help="use start_epoch upon finetune resume",
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)
    parser.add_argument(
        "--flip_str",
        default=None,
        help="Horizontally flip images that include this string",
    )

    return parser


def main(args):
    if args.model == "vit_large_patch16":
        assert timm.__version__ == "0.3.2"
    else:
        assert timm.__version__ == "0.6.5"

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train="train", args=args)
    dataset_val = build_dataset(is_train="val", args=args)
    dataset_test = build_dataset(is_train="test", args=args)

    if args.norm_mean is None or args.norm_std is None:
        norm_params = dataset_train.norm_params
        print("Using training set normalization parameters:", norm_params)
    else:
        norm_params = {"mean": args.norm_mean, "std": args.norm_std}
        print("Using custom normalization parameters:", norm_params)

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_train_eval = torch.utils.data.SequentialSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir + args.task)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_train_eval = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train_eval,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    nb_targets = 1  # Currently limited to one regression target
    model = models_vit.__dict__[args.model](
        img_size=args.input_size,
        num_classes=nb_targets,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )

    if args.finetune and not args.eval:
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint = torch.load(args.finetune, map_location="cpu")
        checkpoint_model = checkpoint["model"]

        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        if not args.linear_probe:
            # interpolate position embedding
            interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    if args.linear_probe:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
        for p in model.fc_norm.parameters():
            p.requires_grad = True

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * args.accum_iter

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.linear_probe:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)
    elif args.layer_decay is not None:
        # build optimizer with layer-wise lr decay (lrd)
        param_groups = lrd.param_groups_lrd(
            model,
            args.weight_decay,
            no_weight_decay_list=model.no_weight_decay(),  # type: ignore
            layer_decay=args.layer_decay,
        )
        optimizer = torch.optim.AdamW(param_groups, lr=args.lr, eps=args.eps)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            eps=args.eps,
            weight_decay=args.weight_decay,
        )

    loss_scaler = NativeScaler()
    criterion = torch.nn.MSELoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(
        args=args,
        model=model,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    if args.eval:
        val_stats, val_mse = evaluate(
            data_loader_val,
            model,
            criterion,
            norm_params,
            device,
            args.task,
            epoch=0,
            mode="val_final",
        )
        test_stats, test_mse = evaluate(
            data_loader_test,
            model,
            criterion,
            norm_params,
            device,
            args.task,
            epoch=0,
            mode="test",
        )
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    min_mse = None
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            norm_params,
            args,
            max_norm=args.clip_grad,
            log_writer=log_writer,
        )

        val_stats, val_mse = evaluate(
            data_loader_val,
            model,
            criterion,
            norm_params,
            device,
            args.task,
            epoch,
            mode="val",
        )
        if min_mse is None or val_mse < min_mse:
            min_mse = val_mse

            if args.output_dir:
                misc.save_model(
                    args=args,
                    model=model,
                    optimizer=optimizer,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )

        if epoch == (args.epochs - 1):
            test_stats, test_mse = evaluate(
                data_loader_test,
                model,
                criterion,
                norm_params,
                device,
                args.task,
                epoch,
                mode="test",
            )

        if log_writer is not None:
            log_writer.add_scalar("perf/val_rmse", val_stats["rmse"], epoch)
            log_writer.add_scalar("perf/val_r2", val_stats["r2"], epoch)
            log_writer.add_scalar("perf/val_loss", val_stats["loss"], epoch)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
