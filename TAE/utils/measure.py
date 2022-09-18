import time
import torch
import math
import numpy as np

def get_rank(scores, targets):
    _, indices = torch.sort(scores, dim=1, descending=True)
    indices = torch.nonzero(indices == targets.view(-1, 1))
    ranks = indices[:, 1].view(-1) + 1

    return ranks

def get_performanceIndex(scores, targets):
    with torch.no_grad():
        ranks = get_rank(scores, targets)

        mrr = torch.mean(1.0 / ranks.float())
        hits1 = torch.mean((ranks <= 1).float())
        hits3 = torch.mean((ranks <= 3).float())
        hits10 = torch.mean((ranks <= 10).float())

    return mrr.item(), hits1.item(), hits3.item(), hits10.item()
