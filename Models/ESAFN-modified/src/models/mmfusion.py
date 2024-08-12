import torch
import torch.nn as nn
import torch.nn.functional as F

class MMFUSION(nn.Module):
  def __init__(self, embedding_matrix, opt):
    super().__init__()
    self.opt = opt
    
    # Embedding layer for text
    self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
    
    # LSTM layer for processing test
    self.lstm = nn.LSTM(input_size=opt.embed_dim, hidden_size=opt.hidden_dim, num_layers=1, batch_first=True)
    
    
    
  def forward():
    pass