# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import timm.models.vision_transformer
import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """Vision Transformer with support for global average pooling"""

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            embed_dim = kwargs["embed_dim"]
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_base_patch16_pt(**kwargs):
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=True,
        num_classes=kwargs["num_classes"],
        drop_path_rate=kwargs["drop_path_rate"],
    )
    return model


def vit_large_patch16_pt(**kwargs):
    model = timm.create_model(
        "vit_large_patch16_224",
        pretrained=True,
        num_classes=kwargs["num_classes"],
        drop_path_rate=kwargs["drop_path_rate"],
    )
    return model


def vit_small_patch16_pt(**kwargs):
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=True,
        num_classes=kwargs["num_classes"],
        drop_path_rate=kwargs["drop_path_rate"],
    )
    return model


def efficientnetv2_m(**kwargs):
    model = timm.create_model(
        "efficientnetv2_rw_m",
        pretrained=True,
        num_classes=kwargs["num_classes"],
        drop_path_rate=kwargs["drop_path_rate"],
    )
    return model


def resnet50(**kwargs):
    model = timm.create_model(
        "resnet50",
        pretrained=True,
        num_classes=kwargs["num_classes"],
        drop_path_rate=kwargs["drop_path_rate"],
    )
    return model


class VisionTransformerSiamese(VisionTransformer):
    def __init__(self, **kwargs):
        super(VisionTransformerSiamese, self).__init__(**kwargs)
        self.head = nn.Linear(kwargs["embed_dim"] * 2, kwargs["num_classes"])

    def forward(self, x):
        assert len(x.shape) == 5 and x.shape[1] == 2
        y, z = x[:, 0], x[:, 1]
        y = self.forward_features(y)
        z = self.forward_features(z)
        x = torch.cat((y, z), dim=1)
        x = self.head(x)
        return x


def vit_large_patch16_siamese(**kwargs):
    model = VisionTransformerSiamese(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
