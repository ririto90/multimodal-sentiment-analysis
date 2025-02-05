import torch
import torch.nn as nn
import torch.nn.functional as F

class DMLANFUSION(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.hidden_dim = opt.hidden_dim
        self.text_feature_dim = 1536
        self.image_feature_dim = 2048

        self.channel_downsample = nn.Conv2d(
            in_channels=self.image_feature_dim,
            out_channels=self.hidden_dim,
            kernel_size=1, stride=1, bias=False
        )

        # Channel Attention
        reduction_ratio = 8
        hidden_dim_ca = self.hidden_dim // reduction_ratio
        self.mlp_avg = nn.Sequential(
            nn.Linear(self.hidden_dim, hidden_dim_ca, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim_ca, self.hidden_dim, bias=False),
        )
        self.mlp_max = nn.Sequential(
            nn.Linear(self.hidden_dim, hidden_dim_ca, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim_ca, self.hidden_dim, bias=False),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        # Spatial Attention
        self.conv_spatial = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=7,
            padding=3,
            bias=False
        )

        # 4) Text Projection: 1536 -> 256
        self.text_proj = nn.Linear(self.text_feature_dim, self.hidden_dim, bias=False)

        # 5) Fusion
        self.fusion_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

        self.final_dim = self.hidden_dim * 2
        self.classifier = nn.Linear(self.final_dim, opt.num_classes)

    def forward(self, text_features, image_features):
        # text_features => [B, 1536]
        # image_features => [B, 2048, H, W]

        # Downsample image
        M_common = self.channel_downsample(image_features)

        # Channel Attention
        avg_out = self.global_avg_pool(M_common).view(M_common.size(0), -1)
        max_out = self.global_max_pool(M_common).view(M_common.size(0), -1)
        channel_attn = F.relu(self.mlp_avg(avg_out) + self.mlp_max(max_out), inplace=True)
        channel_attn = channel_attn.unsqueeze(2).unsqueeze(3)
        F_channel = M_common * channel_attn

        # Spatial Attention
        avg_spatial = torch.mean(F_channel, dim=1, keepdim=True)
        max_spatial, _ = torch.max(F_channel, dim=1, keepdim=True)
        spatial_cat = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_attn = F.relu(self.conv_spatial(spatial_cat), inplace=True)
        F_bi = F_channel * spatial_attn

        # Flatten
        B, C, H, W = F_bi.shape
        v_f = F_bi.view(B, C, H*W).permute(0, 2, 1)

        # Project text (1536 -> 256)
        t_f_proj = self.text_proj(text_features)

        t_f_expanded = t_f_proj.unsqueeze(1).expand(-1, v_f.size(1), -1)

        # Fusion
        vf_times_tf = v_f * t_f_expanded
        mf = torch.tanh(self.fusion_linear(vf_times_tf))

        # G) Visual attention [B, N, 1]
        attn_scores = torch.sum(mf, dim=-1)
        alpha_f = F.softmax(attn_scores, dim=1).unsqueeze(-1)

        # Weighted sum [B, 256]
        s_f = torch.sum(alpha_f * t_f_expanded, dim=1)
        
        # Average-pool [B, 256]
        v_f_pooled = torch.mean(v_f, dim=1)

        # H) Concatenate [B, 512]
        Jf = torch.cat([s_f, v_f_pooled], dim=-1)

        # I) Classify [B, num_classes]
        logits = self.classifier(Jf)
        return logits
