# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import subprocess
import time

import numpy as np

from logging import getLogger

from torch.utils.data import Dataset

import torch
import torchvision

_GLOBAL_SEED = 0
logger = getLogger()


def make_ukbb(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    data_file=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None
):
    dataset = Ukbb(
        root=root_path,
        data_file=data_file,
        transform=transform,
        train=training,
        copy_data=copy_data,
        index_targets=False)
    
    logger.info('UKBB dataset created')
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('Ukbb unsupervised data loader created')

    return dataset, data_loader


class Ukbb(Dataset):

    def __init__(
        self,
        root,
        data_file='imagenet_full_size/061417/',
        transform=None,
        train=True,
        job_id=None,
        local_rank=None,
        copy_data=True,
        index_targets=False
    ):
        super(EvalECGDataset, self).__init__()
    self.data = torch.load(data_path)
    self.data = [d.unsqueeze(0) for d in self.data]
    self.data = [d[:, :args.input_electrodes, :] for d in self.data]
    self.labels = torch.load(labels_path)
    self.augmentation_rate = augmentation_rate
    self.train = train
    self.args = args

    self.transform_train = transforms.Compose([
      augmentations.CropResizing(fixed_crop_len=args.args.input_size[-1], resize=False),
      augmentations.FTSurrogate(phase_noise_magnitude=args.ft_surr_phase_noise),
      augmentations.Jitter(sigma=args.jitter_sigma),
      augmentations.Rescaling(sigma=args.rescaling_sigma),
      augmentations.TimeFlip(prob=0.5),
      augmentations.SignFlip(prob=0.5),
      augmentations.SpecAugment(masking_ratio=0.25, n_fft=120)
    ])

    self.transform_val = transforms.Compose([
      augmentations.CropResizing(fixed_crop_len=args.args.input_size[-1], start_idx=0, resize=False)
    ])

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    data, label = self.data[index], self.labels[index]

    if self.train and (random.random() <= self.eval_train_augment_rate):
      data = self.transform_train(data)
    else:
      data = self.transform_val(data)
    
    return data, label

