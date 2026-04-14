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
    parser = argparse.ArgumentParser("MAE image regression evaluation", add_help=False)
    parser.add_argument("--task", required=True, help="Output directory path")
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size",
    )

    # Model parameters
    parser.add_argument("--pth", default="", type=str, help="Model .pth path")
    parser.add_argument(
        "--model",
        default="vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
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
        required=True,
        type=float,
        help="Target variable normalization mean",
    )
    parser.add_argument(
        "--norm_std",
        required=True,
        type=float,
        help="Target variable normalization std",
    )

    # Other parameters
    parser.add_argument("--device", default="cuda", help="device to use for testing")
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):
    if args.model == "vit_large_patch16":
        assert timm.__version__ == "0.3.2"
    else:
        assert timm.__version__ == "0.6.5"

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    args.flip_str = None
    dataset_test = build_dataset(is_train="test", args=args)
    norm_params = {"mean": args.norm_mean, "std": args.norm_std}

    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

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
        drop_path_rate=0.2,
        global_pool=args.global_pool,
    )
    criterion = torch.nn.MSELoss()

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    checkpoint = torch.load(args.pth, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    evaluate(
        data_loader_test,
        model,
        criterion,
        norm_params,
        device,
        args.task,
        epoch=0,
        mode="test",
    )


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    main(args)
