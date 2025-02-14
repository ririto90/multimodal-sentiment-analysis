import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attention import Attention

class SimpleTextAtt(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.num_classes = opt.num_classes
        self.hidden_dim = opt.hidden_dim

        self.lstm_dim = 2 * 768  
        self.attention = Attention(
            embed_dim=self.lstm_dim,
            hidden_dim=self.lstm_dim // opt.n_head,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            dropout=opt.dropout_rate
        )
        self.post_att_proj = nn.Linear(self.lstm_dim, self.hidden_dim)
        self.batch_norm = nn.BatchNorm1d(self.hidden_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.hidden_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lstm_outputs):
        """
        lstm_outputs: [batch_size, seq_len, lstm_dim]
        """

        attended_seq = self.attention(k=lstm_outputs, q=lstm_outputs)
        
        context_vector = attended_seq.mean(dim=1)

        t_proj = self.post_att_proj(context_vector)

        t_proj = self.batch_norm(t_proj)
        t_proj = self.dropout(t_proj)

        t_hidden = F.relu(self.hidden_layer(t_proj))
        logits = self.classifier(t_hidden)
        

        return logits
