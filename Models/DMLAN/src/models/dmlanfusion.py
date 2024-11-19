import torch
import torch.nn as nn
import torch.nn.functional as F

class DMLANFUSION(nn.Module):
    def __init__(self, opt, text_feature_dim, image_feature_dim):
        super(DMLANFUSION, self).__init__()
        self.opt = opt
 
        # Visual Attention Module
        # Channel Attention
        self.channel_pool_avg = nn.AdaptiveAvgPool2d(1)
        self.channel_pool_max = nn.AdaptiveMaxPool2d(1)
        reduction_ratio = 16  # You can adjust this value
        self.channel_mlp = nn.Sequential(
            nn.Linear(image_feature_dim, image_feature_dim // reduction_ratio),
            nn.ReLU(),
            nn.Linear(image_feature_dim // reduction_ratio, image_feature_dim)
        )

        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

        # Project text and image features to the same dimension
        d_model = 256  # Common feature dimension
        self.text_projection = nn.Linear(text_feature_dim, d_model)
        self.image_projection = nn.Linear(image_feature_dim, d_model)

        # For computing m_f in Joint Attended Multimodal Learning
        self.W_mf = nn.Linear(d_model, 1)

        # Self-Attention over Joint Features
        self.W_self_attn = nn.Linear(d_model * 2, 1)

        # Final classifier
        self.classifier = nn.Linear(d_model * 2, opt.num_classes)

    def forward(self, text_features, image_features):
        # text_features: [batch_size, seq_len, text_feature_dim]
        # image_features: [batch_size, C, H, W]

        batch_size = image_features.size(0)
        C = image_features.size(1)
        H = image_features.size(2)
        W = image_features.size(3)

        # Channel Attention
        avg_pool = self.channel_pool_avg(image_features).view(batch_size, C)
        max_pool = self.channel_pool_max(image_features).view(batch_size, C)

        avg_pool_proj = self.channel_mlp(avg_pool)
        max_pool_proj = self.channel_mlp(max_pool)

        channel_attention = F.relu(avg_pool_proj + max_pool_proj).unsqueeze(2).unsqueeze(3)
        # Apply channel attention
        channel_refined_feature = image_features * channel_attention  # [batch_size, C, H, W]

        # Spatial Attention
        avg_pool_spatial = torch.mean(channel_refined_feature, dim=1, keepdim=True)  # [batch_size, 1, H, W]
        max_pool_spatial, _ = torch.max(channel_refined_feature, dim=1, keepdim=True)  # [batch_size, 1, H, W]
        spatial_pool = torch.cat([avg_pool_spatial, max_pool_spatial], dim=1)  # [batch_size, 2, H, W]

        spatial_attention = F.relu(self.spatial_conv(spatial_pool))  # [batch_size, 1, H, W]
        # Apply spatial attention
        bi_attentive_features = channel_refined_feature * spatial_attention  # [batch_size, C, H, W]

        # Flatten spatial dimensions
        m = H * W
        v_f = bi_attentive_features.view(batch_size, C, m).permute(0, 2, 1)  # [batch_size, m, C]

        # Project image features
        v_f_proj = self.image_projection(v_f)  # [batch_size, m, d_model]

        # Text features: outputs from LSTM at each time step
        t_f = text_features  # [batch_size, seq_len, text_feature_dim]
        t_f_proj = self.text_projection(t_f)  # [batch_size, seq_len, d_model]

        ### Joint Attended Multimodal Learning ###
        # Simplify by averaging image features
        v_f_proj_avg = torch.mean(v_f_proj, dim=1)  # [batch_size, d_model]
        v_f_proj_avg_expanded = v_f_proj_avg.unsqueeze(1)  # [batch_size, 1, d_model]

        # Element-wise multiplication
        joint_features = t_f_proj * v_f_proj_avg_expanded  # [batch_size, seq_len, d_model]

        # Compute m_f
        m_f = torch.tanh(self.W_mf(joint_features))  # [batch_size, seq_len, 1]

        # Compute attention scores over text sequence
        alpha_f = F.softmax(m_f.squeeze(2), dim=1).unsqueeze(2)  # [batch_size, seq_len, 1]

        # Compute attended text features s_f
        s_f = torch.sum(alpha_f * t_f_proj, dim=1)  # [batch_size, d_model]

        # Mean of image features
        v_f_mean = torch.mean(v_f_proj, dim=1)  # [batch_size, d_model]

        # Concatenate attended text features and image features
        J_f = torch.cat([s_f, v_f_mean], dim=1)  # [batch_size, d_model * 2]

        # Self-Attention over Joint Features
        attn_weights = F.softmax(self.W_self_attn(J_f), dim=1)  # [batch_size, 1]
        M = attn_weights * J_f  # [batch_size, d_model * 2]

        # Classification
        logits = self.classifier(M)  # [batch_size, num_classes]

        return logits
