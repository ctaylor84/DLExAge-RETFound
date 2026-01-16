# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from typing import Any, Callable, List, Optional, Tuple, Dict, Union
import torch
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import pandas as pd
from PIL.ImageOps import mirror


class RegressionDataset(datasets.VisionDataset):
    def __init__(
        self,
        root: str,
        subset: str,
        transform: Optional[Callable] = None,
        flip_str: Union[str, None] = None,
        return_fnames: bool = False,
    ) -> None:
        super().__init__(os.path.join(root, "images"), transform=transform)
        self.samples, self.norm_params = self.make_dataset(root, subset, flip_str)
        self.loader = datasets.folder.default_loader
        self.return_fnames = return_fnames

    @staticmethod
    def make_dataset(
        directory: str,
        subset: str,
        flip_str: Union[str, None],
    ) -> Tuple[List[Tuple[str, int, bool, str]], Dict[str, torch.Tensor]]:
        directory = os.path.expanduser(directory)
        image_dir = os.path.join(directory, "images")
        use_subdir = os.path.isdir(os.path.join(image_dir, "1"))
        subset = "train" if subset == "train_eval" else subset
        subset_csv = os.path.join(directory, subset + ".csv")

        targets_df = pd.read_csv(
            subset_csv, header=0, dtype={"file": str, "target": float}
        )
        targets = dict(zip(targets_df["file"], targets_df["target"]))

        instances = list()
        for target_file, target_value in targets.items():
            paths = list()
            flip_list = list()
            for sub_file in target_file.split(":"):
                if use_subdir:
                    image_path = os.path.join(image_dir, sub_file[0], sub_file)
                else:
                    image_path = os.path.join(image_dir, sub_file)

                paths.append(image_path)
                flip_list.append(flip_str is not None and sub_file.find(flip_str) != -1)

            item = paths, target_value, flip_list, target_file
            instances.append(item)

        target_values = torch.tensor(list(targets.values()))
        dataset_mean = torch.mean(target_values)
        dataset_std = torch.std(target_values)
        norm_params = {"mean": dataset_mean, "std": dataset_std}

        return instances, norm_params

    def __getitem__(self, index: int) -> Union[Tuple[Any, Any], Tuple[Any, Any, str]]:
        paths, target, flip_list, target_file = self.samples[index]
        sample_array = list()
        for path, flip in zip(paths, flip_list):
            sample = self.loader(path)
            if flip:
                sample = mirror(sample)
            sample_array.append(sample)

        if self.transform is not None:
            for i in range(len(sample_array)):
                sample_array[i] = self.transform(sample_array[i])

        if len(sample_array) > 1:
            sample_array = torch.stack(sample_array)
        else:
            sample_array = sample_array[0]

        if self.return_fnames:
            return sample_array, target, target_file
        else:
            return sample_array, target

    def __len__(self) -> int:
        return len(self.samples)


def build_dataset(is_train, args, return_fnames=False):
    dataset = RegressionDataset(
        args.data_path,
        is_train,
        transform=build_transform(is_train, args),
        flip_str=args.flip_str,
        return_fnames=return_fnames,
    )
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train == "train":
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation="bicubic",
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
