import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity as csim
import numpy as np

def closest_prediction_accuracy(predictions, targets):
    preds_pooled = predictions.mean(dim=1)
    targets_pooled = targets.mean(dim=1)
    sim_mat = csim(preds_pooled.detach().cpu().numpy(),targets_pooled.detach().cpu().numpy())
    max_sim = np.max(sim_mat, axis=1)
    diag = np.diag(sim_mat)
    n = len(diag)
    correct = (max_sim == diag).sum()

    return correct/n

def closest_prediction_probability(predictions, targets):
    preds_pooled = predictions.mean(dim=1)
    targets_pooled = targets.mean(dim=1)
    sim_mat = torch.Tensor(csim(preds_pooled.detach().cpu(),targets_pooled.detach().cpu()))

    sim_prob = F.softmax(sim_mat, dim=0)
    
    diag = torch.diag(sim_prob)

    results = {}
    results["mean"] = diag.mean() * len(diag)
    results["std"] = torch.std(diag)
    results["min"] = diag.min() * len(diag)
    results["max"] = diag.max() * len(diag)

    return results


def split_batch_and_targets(tensor, num_targets = 4, batchsize=128):

    assert tensor.dim()==2
    tensor = torch.permute(tensor, (1, 0))

    tensor = tensor.reshape(tensor.shape[0], num_targets, batchsize)

    tensor = torch.permute(tensor, (2, 1, 0))

    return tensor


def closest_prediction_probability_per_ecg(predictions, targets, num_targets = 4, batchsize=128):


    preds_pooled = predictions.mean(dim=1)
    preds = split_batch_and_targets(preds_pooled, num_targets=num_targets, batchsize=batchsize)

    targets_pooled = targets.mean(dim=1)
    target = split_batch_and_targets(targets_pooled, num_targets=num_targets, batchsize=batchsize)

    temp = []

    for p,t in zip(preds,target):
                
        sim_mat = torch.Tensor(csim(p.detach().cpu(),t.detach().cpu()))
        sim_prob = F.softmax(sim_mat, dim=0)
            
        diag = torch.diag(sim_prob)
        temp.append(diag.mean())

    temp = torch.Tensor(temp)
    results = {}
    results["mean"] = temp.mean() 
    results["std"] = torch.std(temp)
    results["min"] = temp.min() 
    results["max"] = temp.max() 

    return results



def global_pooled_cosine_sim_score(context, target, masks = None):
    if masks is not None:
        targets = []
        for m in masks:
            mask_keep = m.unsqueeze(-1).repeat(1, 1, target.size(-1))
            targets += [torch.gather(target, dim=1, index=mask_keep)]
        
    else:
        targets = target
        #assert(targets.dim == 4)

    sims = []
    for target in targets:
        ctx = context.mean(dim=1)
        target = target.mean(dim=1)
        
        sim_func = F.cosine_similarity(ctx, target, dim=1) # shape (n,)
        sims.append(sim_func)

    return torch.stack(sims).mean()
        

