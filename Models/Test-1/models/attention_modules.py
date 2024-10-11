import torch
import torch.nn as nn
import math

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module for text or image features.
    """
    def __init__(self, d_model=768, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 2048),
            nn.ReLU(),
            nn.Linear(2048, d_model)
        )

    def scaled_dot_product_attention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)

    def combine_heads(self, x):
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * d_k)

    def forward(self, x):
        residual = x

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_output = self.scaled_dot_product_attention(Q, K, V)
        attention_output = self.combine_heads(attention_output)
        attention_output = self.W_O(attention_output)

        x = self.layer_norm1(residual + attention_output)  # Add & Norm

        residual = x
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(residual + ff_output)  # Add & Norm

        return x  # Output shape: (batch_size, seq_len, d_model)

class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention module between two modalities.
    """
    def __init__(self, d_model=768, num_heads=8):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.layer_norm = nn.LayerNorm(d_model)

    def scaled_dot_product_attention(self, Q, K, V):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)  # (batch_size, num_heads, seq_len, d_k)

    def combine_heads(self, x):
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * d_k)

    def forward(self, Q_input, K_input, V_input):
        Q = self.W_Q(Q_input)
        K = self.W_K(K_input)
        V = self.W_V(V_input)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_output = self.scaled_dot_product_attention(Q, K, V)
        attention_output = self.combine_heads(attention_output)
        attention_output = self.W_O(attention_output)

        output = self.layer_norm(Q_input + attention_output)  # Add & Norm
        return output  # Output shape: (batch_size, seq_len, d_model)
