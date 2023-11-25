import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from logging import getLogger


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


_GLOBAL_SEED = 0
logger = getLogger()


def make_mimic(
    transform=None,
    batch_size=16,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None
):
    dataset = Mimic(
        mimic_tensor=image_folder,
        transform=transform)
    
    #if subset_file is not None:
    #    dataset = ImageNetSubset(dataset, subset_file)

    logger.info('Mimic dataset created')
    

    data_loader = DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=False)
    logger.info('ImageNet unsupervised data loader created')

    return dataset, data_loader




class Mimic(Dataset):
    """Mimic dataset."""

    def __init__(self, mimic_tensor=None, transform=None):
        """
        Arguments:
            mimic_tensor (string): Path to the mimic pytorch file
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mimic_tensor = torch.load(mimic_tensor+".pt").float()
        self.labels = torch.load(mimic_tensor+"_labels.pt").int()
        self.transform = transform

    def __len__(self):
        return len(self.mimic_tensor)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.mimic_tensor[idx]

        if self.transform:
            sample = self.transform(sample)

        sample = torch.unsqueeze(sample,0)

        sample = torch.nan_to_num(sample)

        target = torch.zeros(sample.shape[0])
        return (sample, target)

