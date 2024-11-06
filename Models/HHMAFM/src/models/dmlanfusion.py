import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attention import SelfAttention
from layers.cross_attention import CrossAttention
from layers.co_attention import CoAttention

class DMLANFUSION (nn.Module):
  def __init__(self, opt):
    super (DMLANFUSION, self).__init__()
    self.opt = opt
    
    
  
  def forward(self, ):
      pass
    