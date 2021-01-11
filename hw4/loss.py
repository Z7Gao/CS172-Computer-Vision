import torch
import torch.nn as nn

class ScaleInvariantLoss(nn.Module):
    def __init__(self, lambda_ = 0.5):
        super(ScaleInvariantLoss, self).__init__()
        self.lambda_ = lambda_

    def forward(self, predict, gt):
        # mask out missing value in ground truth
        d = predict[~(gt == 0)] - torch.log(gt[~(gt == 0)])
        n = list(d.size())[0]
        loss = torch.mean(d**2) - self.lambda_ / (n**2) * (torch.sum(d)**2)  
        return loss