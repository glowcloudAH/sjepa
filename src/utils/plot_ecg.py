import numpy as numpy
import torch
import pandas
import matplotlib.pyplot as plt


def plot_ecg(ecg, masks = None):
    ecg = ecg.detach().cpu()
    if masks is not None:
        ecg =  apply_masks(ecg, masks)
        
    
    _plot(ecg)

def _plot(ecg):
    fig, axs = plt.subplots(len(ecg))
    ecg = ecg.numpy()
    for i,channel in enumerate(ecg):
        axs[i].plot(channel)

def apply_masks(ecg, masks):
    h,w = 12,50
    
    zeros = torch.zeros(h*w)
    zeros[masks] = 1
        
    zeros = zeros.reshape((h,w))
    zeros = torch.repeat_interleave(zeros,100, dim=1)

    return zeros*ecg
