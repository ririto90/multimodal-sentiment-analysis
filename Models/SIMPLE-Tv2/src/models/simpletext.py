import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleText(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.num_classes = opt.num_classes
        self.hidden_dim = opt.hidden_dim
        self.text_dim = 1536

        # Projection layers
        self.text_proj = nn.Linear(self.text_dim, self.hidden_dim)
        self.hidden_layer = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Normalization and dropout
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim)
        self.dropout = nn.Dropout(p=opt.dropout_rate)

        # Classifier
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, text_feature):
        t_proj = F.relu(self.text_proj(text_feature))
        t_hidden = F.relu(self.hidden_layer(t_proj))
        logits = self.classifier(t_hidden)
        return logits