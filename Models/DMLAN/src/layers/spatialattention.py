import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        # Convolution for 2-channel input (avg + max) -> single channel
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, 
                                kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        """
        x: [B, C, H, W] (channel-refined feature map F in the paper)
        Returns: [B, C, H, W] (spatial-refined feature map)
        """
        b, c, h, w = x.shape

        # 1) Channel-wise average: [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # 2) Channel-wise max: [B, 1, H, W]
        max_out = torch.max(x, dim=1, keepdim=True)[0]

        # 3) Concat along channel dimension -> [B, 2, H, W]
        concat_out = torch.cat([avg_out, max_out], dim=1)

        # 4) Convolution + ReLU -> spatial attention map [B, 1, H, W]
        A_s = F.relu(self.conv2d(concat_out), inplace=True)

        # 5) Multiply with x to refine features -> [B, C, H, W]
        spatial_refined = x * A_s

        return spatial_refined