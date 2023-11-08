# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        print("x:", x.size())
        print("m:", m.size())
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        print("mask_keep:", mask_keep.size())
        all_x += [torch.gather(x, dim=1, index=mask_keep)]

    print(torch.cat(all_x, dim=0).size())
    return torch.cat(all_x, dim=0)
