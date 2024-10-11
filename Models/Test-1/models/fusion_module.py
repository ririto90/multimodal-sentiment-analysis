import torch
import torch.nn as nn

class FusionModule(nn.Module):
    """
    Fuses features hierarchically from different modalities.
    """
    def __init__(self, d_model=768):
        super(FusionModule, self).__init__()
        self.linear = nn.Linear(d_model * 2, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, H_SI, H_IS):
        # Concatenate the co-attended features
        fused_features = torch.cat((H_SI, H_IS), dim=-1)
        fused_features = self.linear(fused_features)
        fused_features = self.layer_norm(fused_features)
        return fused_features
