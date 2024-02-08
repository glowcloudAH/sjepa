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

    return diag.mean()



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
        

