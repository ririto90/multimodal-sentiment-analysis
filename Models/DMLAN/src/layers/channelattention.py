import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim=256):
        super(ChannelAttention, self).__init__()
        # Two-layer MLP (W0 -> ReLU -> W1 in the paper)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels)
        )

    def forward(self, x):
        """
        x: [B, C, H, W] (feature map from the backbone, e.g. Inception V3)
        Returns: [B, C, H, W] (channel-refined feature map)
        """
        b, c, h, w = x.shape

        # 1) Global Max Pool -> [B, C]
        max_pooled = F.adaptive_max_pool2d(x, 1).view(b, c)
        # 2) Global Avg Pool -> [B, C]
        avg_pooled = F.adaptive_avg_pool2d(x, 1).view(b, c)

        # 3) MLP transformations
        mp_out = self.mlp(max_pooled)
        ap_out = self.mlp(avg_pooled)

        # 4) Element-wise sum -> ReLU -> channel attention map [B, C]
        A_c = F.relu(mp_out + ap_out)

        # 5) Reshape to broadcast over spatial dims [B, C, 1, 1]
        A_c_reshaped = A_c.view(b, c, 1, 1)

        # Multiply with input x to get channel-refined features [B, C, H, W]
        channel_refined = x * A_c_reshaped

        return channel_refined