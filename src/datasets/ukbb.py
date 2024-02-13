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
from typing import Tuple

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
    subset_file=None,
    persistent_workers=False
):
    
    data_path = root_path + data_file
    labels_path = root_path + data_file.replace("ecgs", "labels").replace("_noBase_gn","")
    dataset = Ukbb(
        data_path=data_path,
        labels_path=labels_path,
        transform=transform)
    
    logger.info('UKBB dataset created')
    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank
        )
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=persistent_workers)
    logger.info('Ukbb unsupervised data loader created')

    return dataset, data_loader, dist_sampler


class Ukbb(Dataset):

    def __init__(
        self, data_path: str, labels_path: str, transform):
        super(Ukbb, self).__init__()
        logger.info('Initialized UKBB')
        self.data = torch.load(data_path, map_location=torch.device('cpu'))
        #self.data = [d.clone().unsqueeze(0) for d in self.data]
        #self.data = [d[:, :args.input_electrodes, :] for d in self.data]
        try:
            self.labels = torch.load(labels_path, map_location=torch.device('cpu'))
        except:
           self.labels = torch.zeros(len(self.data))

        self.transform_train = transform
        
        #transforms.Compose([
        #  augmentations.CropResizing(fixed_crop_len=args.args.input_size[-1], resize=False),
        #  augmentations.FTSurrogate(phase_noise_magnitude=args.ft_surr_phase_noise),
        #  augmentations.Jitter(sigma=args.jitter_sigma),
        #  augmentations.Rescaling(sigma=args.rescaling_sigma),
        #  augmentations.TimeFlip(prob=0.5),
        #  augmentations.SignFlip(prob=0.5),
        #  augmentations.SpecAugment(masking_ratio=0.25, n_fft=120)
        #])

    def __len__(self) -> int:
      return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
      #data = torch.load(self.data, map_location=torch.device('cpu'))[index].unsqueeze(0)
      data = self.data[index].unsqueeze(0)
      label = self.labels[index]
      if self.transform_train:
        data = self.transform_train(data)
      
      #data = data.clone()
      return data, label

