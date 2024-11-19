import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CoAttention(nn.Module):
    def __init__(self, embed_dim1, embed_dim2, hidden_dim=None, n_head=1, score_function='bi_linear', dropout=0.1):
        super(CoAttention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim1 // n_head
        self.embed_dim1 = embed_dim1
        self.embed_dim2 = embed_dim2
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function

        # Projection matrices for the first input
        self.w_q1 = nn.Parameter(torch.FloatTensor(n_head, embed_dim1, hidden_dim))
        self.w_k1 = nn.Parameter(torch.FloatTensor(n_head, embed_dim1, hidden_dim))
        self.w_v1 = nn.Parameter(torch.FloatTensor(n_head, embed_dim1, hidden_dim))

        # Projection matrices for the second input
        self.w_q2 = nn.Parameter(torch.FloatTensor(n_head, embed_dim2, hidden_dim))
        self.w_k2 = nn.Parameter(torch.FloatTensor(n_head, embed_dim2, hidden_dim))
        self.w_v2 = nn.Parameter(torch.FloatTensor(n_head, embed_dim2, hidden_dim))

        # Output projections
        self.proj1 = nn.Linear(n_head * hidden_dim, embed_dim1)
        self.proj2 = nn.Linear(n_head * hidden_dim, embed_dim2)
        self.dropout = nn.Dropout(dropout)

        if score_function == 'mlp':
            self.weight_mlp = nn.Linear(hidden_dim * 2, 1)
            nn.init.xavier_uniform_(self.weight_mlp.weight)
            nn.init.zeros_(self.weight_mlp.bias)
        elif score_function == 'bi_linear':
            self.weight_bilinear = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
            nn.init.xavier_uniform_(self.weight_bilinear)

        # Initialize parameters
        nn.init.xavier_uniform_(self.w_q1)
        nn.init.xavier_uniform_(self.w_k1)
        nn.init.xavier_uniform_(self.w_v1)
        nn.init.xavier_uniform_(self.w_q2)
        nn.init.xavier_uniform_(self.w_k2)
        nn.init.xavier_uniform_(self.w_v2)
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.xavier_uniform_(self.proj2.weight)

    def forward(self, x1, x2, mask1=None, mask2=None):
        batch_size = x1.size(0)
        len1 = x1.size(1)
        len2 = x2.size(1)

        # Projections for x1
        q1 = x1.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim1)
        k1 = x1.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim1)
        v1 = x1.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim1)

        q1 = torch.bmm(q1, self.w_q1).view(-1, len1, self.hidden_dim)
        k1 = torch.bmm(k1, self.w_k1).view(-1, len1, self.hidden_dim)
        v1 = torch.bmm(v1, self.w_v1).view(-1, len1, self.hidden_dim)

        # Projections for x2
        q2 = x2.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim2)
        k2 = x2.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim2)
        v2 = x2.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim2)

        q2 = torch.bmm(q2, self.w_q2).view(-1, len2, self.hidden_dim)
        k2 = torch.bmm(k2, self.w_k2).view(-1, len2, self.hidden_dim)
        v2 = torch.bmm(v2, self.w_v2).view(-1, len2, self.hidden_dim)

        # Compute attention from x1 to x2
        if self.score_function == 'scaled_dot_product':
            score12 = torch.bmm(q1, k2.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        elif self.score_function == 'bi_linear':
            W_k2_T = torch.matmul(k2, self.weight_bilinear)
            score12 = torch.bmm(q1, W_k2_T.transpose(1, 2))
        elif self.score_function == 'mlp':
            q1_expanded = q1.unsqueeze(2).expand(-1, -1, len2, -1)
            k2_expanded = k2.unsqueeze(1).expand(-1, len1, -1, -1)
            concat_qk = torch.cat((q1_expanded, k2_expanded), dim=-1)
            concat_qk = concat_qk.view(-1, self.hidden_dim * 2)
            energy = self.weight_mlp(concat_qk).view(-1, len1, len2)
            score12 = energy
        else:
            raise ValueError('Invalid score function')

        if mask2 is not None:
            mask2 = mask2.unsqueeze(1).repeat(self.n_head * batch_size, len1, 1)
            score12 = score12.masked_fill(mask2 == 0, -1e9)

        attention_weights12 = F.softmax(score12, dim=-1)
        out1 = torch.bmm(attention_weights12, v2)

        # Compute attention from x2 to x1
        if self.score_function == 'scaled_dot_product':
            score21 = torch.bmm(q2, k1.transpose(1, 2)) / math.sqrt(self.hidden_dim)
        elif self.score_function == 'bi_linear':
            W_k1_T = torch.matmul(k1, self.weight_bilinear)
            score21 = torch.bmm(q2, W_k1_T.transpose(1, 2))
        elif self.score_function == 'mlp':
            q2_expanded = q2.unsqueeze(2).expand(-1, -1, len1, -1)
            k1_expanded = k1.unsqueeze(1).expand(-1, len2, -1, -1)
            concat_qk = torch.cat((q2_expanded, k1_expanded), dim=-1)
            concat_qk = concat_qk.view(-1, self.hidden_dim * 2)
            energy = self.weight_mlp(concat_qk).view(-1, len2, len1)
            score21 = energy
        else:
            raise ValueError('Invalid score function')

        if mask1 is not None:
            mask1 = mask1.unsqueeze(1).repeat(self.n_head * batch_size, len2, 1)
            score21 = score21.masked_fill(mask1 == 0, -1e9)

        attention_weights21 = F.softmax(score21, dim=-1)
        out2 = torch.bmm(attention_weights21, v1)

        # Concatenate heads
        out1 = torch.cat(torch.split(out1, batch_size, dim=0), dim=-1)
        out2 = torch.cat(torch.split(out2, batch_size, dim=0), dim=-1)

        out1 = self.proj1(out1)
        out2 = self.proj2(out2)
        out1 = self.dropout(out1)
        out2 = self.dropout(out2)

        return out1, out2, attention_weights12, attention_weights21