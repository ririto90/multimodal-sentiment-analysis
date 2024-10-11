import torch
import torch.nn as nn
import torch.nn.functional as F

class CoAttentionModule(nn.Module):
    """
    Computes co-attention between text and image features with gating mechanisms.
    """
    def __init__(self, d_model=768):
        super(CoAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Gated adjustment vectors
        self.W_GS = nn.Linear(d_model, d_model)
        self.b_GS = nn.Parameter(torch.zeros(d_model))
        self.W_GI = nn.Linear(d_model, d_model)
        self.b_GI = nn.Parameter(torch.zeros(d_model))

        # Multi-head attention
        self.cross_attention = MultiHeadCrossAttention(d_model=d_model, num_heads=8)

    def forward(self, S_output, I_output):
        # Compute average pool
        g_S = self.avg_pool(S_output.transpose(1, 2)).squeeze(-1)  # (batch_size, d_model)
        g_I = self.avg_pool(I_output.transpose(1, 2)).squeeze(-1)  # (batch_size, d_model)

        # Compute gated adjustment vectors
        G_SI = torch.sigmoid(self.W_GS(g_S) + self.b_GS)
        G_IS = torch.sigmoid(self.W_GI(g_I) + self.b_GI)

        # Compute new queries and keys
        Q_tilde_S = (1 + G_IS).unsqueeze(1) * S_output  # Broadcasting over sequence length
        K_tilde_S = (1 + G_IS).unsqueeze(1) * S_output

        Q_tilde_I = (1 + G_SI).unsqueeze(1) * I_output
        K_tilde_I = (1 + G_SI).unsqueeze(1) * I_output

        # Compute co-attention
        H_SI = self.cross_attention(Q_tilde_S, K_tilde_I, I_output)  # H_(S←I)
        H_IS = self.cross_attention(Q_tilde_I, K_tilde_S, S_output)  # H_(I←S)

        return H_SI, H_IS
