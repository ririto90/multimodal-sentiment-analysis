import torch
import torch.nn as nn
import torch.nn.functional as F

class DMLANFUSION(nn.Module):
    def __init__(self, opt, text_feature_dim, image_feature_dim):
        super(DMLANFUSION, self).__init__()
        self.opt = opt
        
    def forward(self, text_features, image_features):
        
        return 
