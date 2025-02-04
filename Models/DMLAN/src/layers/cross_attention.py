import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, embed_dim_q, embed_dim_kv, hidden_dim=None, n_head=1, score_function='scaled_dot_product', dropout=0.1):
        ''' Cross Attention Mechanism
        :param embed_dim_q: embedding dimension of the query
        :param embed_dim_kv: embedding dimension of the key and value
        :param hidden_dim: hidden dimension for projections
        :param n_head: number of heads (Multi-Head Attention)
        :param score_function: 'scaled_dot_product', 'mlp', or 'bi_linear'
        '''
        super(CrossAttention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim_q // n_head
        self.embed_dim_q = embed_dim_q
        self.embed_dim_kv = embed_dim_kv
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function

        self.w_q = nn.Parameter(torch.FloatTensor(n_head, embed_dim_q, hidden_dim))
        self.w_k = nn.Parameter(torch.FloatTensor(n_head, embed_dim_kv, hidden_dim))
        self.w_v = nn.Parameter(torch.FloatTensor(n_head, embed_dim_kv, hidden_dim))

        self.proj = nn.Linear(n_head * hidden_dim, embed_dim_q)
        self.dropout = nn.Dropout(dropout)

        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim*2, 1))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)

        # Initialize parameters
        nn.init.xavier_uniform_(self.w_q)
        nn.init.xavier_uniform_(self.w_k)
        nn.init.xavier_uniform_(self.w_v)
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, query, key, value, mask=None):
        # query: (batch_size, q_len, embed_dim_q)
        # key, value: (batch_size, kv_len, embed_dim_kv)

        batch_size = query.size(0)
        q_len = query.size(1)
        kv_len = key.size(1)

        # Prepare the projections
        q = query.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim_q)  # (n_head, batch_size*q_len, embed_dim_q)
        k = key.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim_kv)    # (n_head, batch_size*kv_len, embed_dim_kv)
        v = value.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim_kv)  # (n_head, batch_size*kv_len, embed_dim_kv)

        q = torch.bmm(q, self.w_q).view(-1, q_len, self.hidden_dim)  # (n_head*batch_size, q_len, hidden_dim)
        k = torch.bmm(k, self.w_k).view(-1, kv_len, self.hidden_dim) # (n_head*batch_size, kv_len, hidden_dim)
        v = torch.bmm(v, self.w_v).view(-1, kv_len, self.hidden_dim) # (n_head*batch_size, kv_len, hidden_dim)

        if self.score_function == 'scaled_dot_product':
            k_t = k.transpose(1, 2)  # (n_head*batch_size, hidden_dim, kv_len)
            score = torch.bmm(q, k_t) / math.sqrt(self.hidden_dim)  # (n_head*batch_size, q_len, kv_len)
        elif self.score_function == 'mlp':
            # Not implemented in this simplified version
            raise NotImplementedError('MLP score function not implemented in CrossAttention')
        elif self.score_function == 'bi_linear':
            # Not implemented in this simplified version
            raise NotImplementedError('Bi-linear score function not implemented in CrossAttention')
        else:
            raise ValueError('Invalid score function')

        if mask is not None:
            # Apply mask (batch_size, 1, kv_len)
            mask = mask.repeat(self.n_head, 1, 1)  # (n_head*batch_size, 1, kv_len)
            score = score.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(score, dim=-1)  # (n_head*batch_size, q_len, kv_len)

        # Apply attention weights to values
        out = torch.bmm(attention_weights, v)  # (n_head*batch_size, q_len, hidden_dim)

        # Concatenate heads
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)  # (batch_size, q_len, n_head*hidden_dim)

        # Final linear projection
        out = self.proj(out)  # (batch_size, q_len, embed_dim_q)
        out = self.dropout(out)

        return out, attention_weights