from torch import nn 
import torch.nn.functional as f

from torchvision import models
from torchvision import transforms as T

import segmentation_models_pytorch as smp 

class PneumothoraxModel(nn.Module):
    
    def __init__(self, pretrained = False):
        super(PneumothoraxModel, self).__init__()
        
        self.backbone = smp.Unet(
            encoder_name = 'timm-efficientnet-b0',
            in_channels = 3,
            classes = 1,
            activation = None
        )
    
    def forward(self, images):

        logits = self.backbone(images)

        return logits

class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
