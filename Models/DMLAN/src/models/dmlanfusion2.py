import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import SelfAttention
from layers.cross_attention import CrossAttention
from layers.co_attention import CoAttention

class DMLANFUSION2(nn.Module):
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
        
        # Attention Mechanisms
        self.self_attention_text = SelfAttention(
            embed_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            q_len=1,
            dropout=opt.dropout_rate
        )
        self.cross_attention_text_image = CrossAttention(
            embed_dim_q=self.hidden_dim,
            embed_dim_kv=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            dropout=opt.dropout_rate
        )
        self.co_attention_text_image = CoAttention(
            embed_dim1=self.hidden_dim,
            embed_dim2=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            dropout=opt.dropout_rate
        )

        # 5) Fusion
        self.fusion_linear = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)

        self.final_dim = self.hidden_dim * 2
        self.classifier = nn.Linear(self.final_dim, opt.num_classes)

    def forward(self, text_features, image_features):
        """
        :param text_features: [B, 1536] raw text embeddings
        :param image_features: [B, 2048, H, W] image feature map
        :return: logits [B, opt.num_classes]
        """

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
        F_bi = F_channel * spatial_attn  # [B, hidden_dim, H, W]

        # Flatten
        B, C, H, W = F_bi.shape
        v_f = F_bi.view(B, C, H*W).permute(0, 2, 1)  # [B, N, hidden_dim]

        # Project text (1536 -> 256) & Self-Attention
        t_proj = self.text_proj(text_features)  # [B, hidden_dim]
        t_proj = t_proj.unsqueeze(1)  # [B, 1, hidden_dim]
        t_self = self.self_attention_text(t_proj).squeeze(1)  # [B, hidden_dim]
        t_self = t_self + t_proj.squeeze(1)  # [B, hidden_dim]

        # t_f_expanded = t_proj.unsqueeze(1).expand(-1, v_f.size(1), -1)
        
        # C) Cross-Attention
        t_self_expanded = t_self.unsqueeze(1)               # [B, 1, hidden_dim]
        cross_text, _ = self.cross_attention_text_image(
            query=t_self_expanded,
            key=v_f,
            value=v_f
        )
        cross_text = cross_text.squeeze(1) + t_self
        
        # Co-Attention
        co_text_out1, co_image_out2, _, _ = self.co_attention_text_image(
            x1=cross_text.unsqueeze(1),   # [B, 1, hidden_dim]
            x2=v_f                        # [B, 1, hidden_dim]
        )
            
        co_text_out1 = co_text_out1.squeeze(1) + cross_text
        v_f_enriched = co_image_out2 + v_f

        # Fusion
        t_expanded = co_text_out1.unsqueeze(1).expand(-1, v_f_enriched.size(1), -1)
        fused = torch.tanh(self.fusion_linear(v_f_enriched * t_expanded))  # [B, N, hid]

        # Visual attention
        attn_scores = torch.sum(fused, dim=-1)  # [B, N]
        alpha = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # [B, N, 1]

        # Weighted text, average-pooled image
        s_f = torch.sum(alpha * t_expanded, dim=1)   # [B, hid]
        v_f_pooled = torch.mean(v_f_enriched, dim=1) # [B, hid]

        # Final concatenation
        Jf = torch.cat([s_f, v_f_pooled], dim=-1)    # [B, 2*hid]
        logits = self.classifier(Jf)

        return logits
