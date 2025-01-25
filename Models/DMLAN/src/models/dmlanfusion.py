import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attention import SelfAttention

class DMLANFUSION(nn.Module):
    def __init__(self, opt, text_feature_dim, image_feature_dim):
        super(DMLANFUSION, self).__init__()
        self.opt = opt

        self.channel_pool_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_pool_max = nn.AdaptiveMaxPool2d(1)
        reduction_ratio = 16
        self.channel_mlp = nn.Sequential(
            nn.Linear(image_feature_dim, image_feature_dim // reduction_ratio),
            nn.ReLU(),
            nn.Linear(image_feature_dim // reduction_ratio, image_feature_dim)
        )

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        d_model = 256
        self.text_projection = nn.Linear(text_feature_dim, d_model)
        self.image_projection = nn.Linear(image_feature_dim, d_model)

        self.W_mf = nn.Linear(d_model, 1)
        self.self_attention = SelfAttention(embed_dim=d_model * 2, n_head=4, score_function='scaled_dot_product')

        self.classifier = nn.Linear(d_model * 2, opt.num_classes)

    def forward(self, text_features, image_features):
        # text_features: [batch_size, seq_len, text_feature_dim]
        # image_features: [batch_size, C, H, W]
        if self.opt.counter == 0:
            print(self.opt.counter)
            print("text_features:", text_features.shape, "image_features:", image_features.shape)

        batch_size, C, H, W = image_features.size()
        if self.opt.counter == 0:
            print("batch_size:", batch_size, "C:", C, "H:", H, "W:", W)

        ### Channel Attention ###
        avg_pool = self.channel_pool_avg(image_features).view(batch_size, C)
        max_pool = self.channel_pool_max(image_features).view(batch_size, C)
        if self.opt.counter == 0:
            print("avg_pool:", avg_pool.shape, "max_pool:", max_pool.shape)

        avg_pool_proj = self.channel_mlp(avg_pool)
        max_pool_proj = self.channel_mlp(max_pool)
        if self.opt.counter == 0:
            print("avg_pool_proj:", avg_pool_proj.shape, "max_pool_proj:", max_pool_proj.shape)

        channel_attention = F.relu(avg_pool_proj + max_pool_proj).unsqueeze(2).unsqueeze(3)
        if self.opt.counter == 0:
            print("channel_attention:", channel_attention.shape)
            
        channel_refined_feature = image_features * channel_attention
        if self.opt.counter == 0:
            print("channel_refined_feature:", channel_refined_feature.shape)

        ### Spatial Attention ###
        avg_pool_spatial = torch.mean(channel_refined_feature, dim=1, keepdim=True)  # [batch_size, 1, H, W]
        max_pool_spatial, _ = torch.max(channel_refined_feature, dim=1, keepdim=True)  # [batch_size, 1, H, W]
        
        spatial_pool = torch.cat([avg_pool_spatial, max_pool_spatial], dim=1)  # [batch_size, 2, H, W]

        spatial_attention = F.relu(self.spatial_conv(spatial_pool))  # [batch_size, 1, H, W]
        
        # Apply spatial attention
        bi_attentive_features = channel_refined_feature * spatial_attention  # [batch_size, C, H, W]

        # Flatten spatial dimensions
        num_regions = H * W
        flattened_visual_features = bi_attentive_features.view(batch_size, C, num_regions).permute(0, 2, 1)  # [batch_size, m, C]

        # Project image features
        projected_visual_features = self.image_projection(flattened_visual_features)  # [batch_size, m, d_model]

        projected_text_features = self.text_projection(text_features)  # [batch_size, seq_len, d_model]

        ### Joint Attended Multimodal Learning ###

        # Process visual features
        avg_visual_features = torch.mean(projected_visual_features, dim=1)
        expanded_avg_visual_features = avg_visual_features.unsqueeze(1)

        # Combine text and visual features
        joint_features = projected_text_features * expanded_avg_visual_features

        # Compute attention scores
        joint_attention_logits = torch.tanh(self.W_mf(joint_features))
        attention_scores = F.softmax(joint_attention_logits.squeeze(2), dim=1).unsqueeze(2)

        # Compute attended text features
        attended_text_features = torch.sum(attention_scores * projected_text_features, dim=1)

        # Combine attended text and visual features
        mean_visual_features = torch.mean(projected_visual_features, dim=1)
        joint_features = torch.cat([attended_text_features, mean_visual_features], dim=1)

        # Apply self-attention
        attended_multimodal_features = self.self_attention(joint_features)
        attended_multimodal_features = attended_multimodal_features.squeeze(1)

        # Classification
        logits = self.classifier(attended_multimodal_features)
            
        self.opt.counter += 1
        if self.opt.counter < 3:
            print(self.opt.counter)
        return logits
