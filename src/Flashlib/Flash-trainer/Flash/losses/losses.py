import torch 
from torch import nn 
import torch.nn.functional as F 
from torch.autograd import Variable


'''
no losses are present at this moment.
'''


class LabelSmoothingForBCE(nn.Module):
    def __init__(self, smoothing = 0.09):
        super(LabelSmoothingForBCE, self).__init__()
        self.smoothing = smoothing 

    def forward(self,logits, labels):
        labels[labels == 1] = 1 - self.smoothing 
        labels[labels == 0] = self.smoothing 
        return F.binary_cross_entropy_with_logits(logits, labels)
        
        
