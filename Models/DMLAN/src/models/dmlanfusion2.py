import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.channelattention import ChannelAttention
from layers.spatialattention import SpatialAttention

class DMLANFUSION2(nn.Module):
    def __init__(self, opt, text_feature_dim, image_feature_dim):
        super(DMLANFUSION2, self).__init__()
        self.opt = opt

        # ------------------- Channel & Spatial Attention ------------------- #
        self.channel_attention = ChannelAttention(in_channels=image_feature_dim)
        self.spatial_attention = SpatialAttention(kernel_size=7)

        # ------------------- Dimensions & Projections ------------------- #
        # We'll assume final 'C' matches text_feature_dim, or we do a projection
        # from 'C' to text_feature_dim. Adjust as needed.
        self.visual_to_text_dim = nn.Linear(image_feature_dim, text_feature_dim)
        self.text_to_visual_dim = nn.Linear(text_feature_dim, image_feature_dim)

        # The joint transform for eq(4): mf = tanh(W(vf ⊙ tf)).
        # We'll do it in text-dim space for convenience
        self.joint_transform = nn.Linear(text_feature_dim, text_feature_dim)

        # Scorer for eq(5) => alpha_f (attention weights)
        self.att_scorer = nn.Linear(text_feature_dim, 1)

        # Combine attended text sf + aggregated visual => Jf
        # Then possibly a final classification layer
        # Output dimension = #classes
        fusion_dim = text_feature_dim + image_feature_dim
        self.classifier = nn.Linear(fusion_dim, opt.num_classes)

    def forward(self, text_features, image_features):
        """
        text_features: [B, seq_len, text_feature_dim]  (LSTM output)
        image_features: [B, image_feature_dim, H, W]   (Inception feature map)
        Returns: logits => [B, num_classes]
        """
        B, C, H, W = image_features.shape

        # ------------------- 1) Channel + Spatial Attention ------------------- #
        # channel_refined -> [B, C, H, W]
        channel_refined = self.channel_attention(image_features)
        # bi_attentive -> [B, C, H, W]
        bi_attentive = self.spatial_attention(channel_refined)

        # ------------------- 2) Flatten to get vf (Equation (3)) ------------------- #
        # vf = [B, m, C] where m=H*W
        vf = bi_attentive.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]

        # ------------------- 3) Extract or pool text feature tf ------------------- #
        # For simplicity, let's take the LAST hidden state => [B, text_feature_dim]
        tf = text_features[:, -1, :]

        # If needed, unify dims. E.g., project visual => text_dim
        # or text => C. We'll do visual->text:
        vf_projected = self.visual_to_text_dim(vf)  # [B, m, text_feature_dim]
        # So now vf_projected and tf have the same dimension "text_feature_dim"

        # ------------------- 4) Equation (4): mf = tanh(W(vf ⊙ tf)) ------------------- #
        # Expand tf => [B, 1, text_feature_dim] for broadcasting
        tf_expanded = tf.unsqueeze(1)  # [B, 1, text_feature_dim]
        # Element-wise multiply -> [B, m, text_feature_dim]
        joint_mult = vf_projected * tf_expanded
        # Apply linear + tanh
        mf = torch.tanh(self.joint_transform(joint_mult))  # [B, m, text_feature_dim]

        # ------------------- 5) Equation (5): alpha_f = softmax(...) ------------------- #
        # We'll reduce dimension to a scalar before softmax => [B, m, 1]
        logits = self.att_scorer(mf)  # => [B, m, 1]
        alpha_f = F.softmax(logits, dim=1)  # [B, m, 1]

        # ------------------- 6) Equation (6): sf = Σ alpha_f * tf ------------------- #
        # Note that alpha_f multiplies tf_expanded => [B, m, text_feature_dim]
        # Summation along 'm' => [B, text_feature_dim]
        weighted_text = alpha_f * tf_expanded  # [B, m, text_feature_dim]
        sf = weighted_text.sum(dim=1)          # [B, text_feature_dim]

        # ------------------- 7) Concatenate sf + aggregated vf => Jf ------------------- #
        # We'll do a simple average of vf across m to get a single vector => [B, C]
        # But remember original C might differ from text_feature_dim
        # so we use the original vf = [B, m, C].
        vf_mean = vf.mean(dim=1)  # [B, C]

        # Jf => [B, text_feature_dim + C]
        Jf = torch.cat([sf, vf_mean], dim=-1)  # eq(7) says we combine them

        # (Optional) If you want a self-attention across multiple Jf vectors
        # you'd typically treat Jf as a sequence. Here we only have 1 combined vector.

        # ------------------- 8) Classification => eq(9): P(s) = softmax(Ws; M) ------------------- #
        logits = self.classifier(Jf)  # [B, num_classes]

        return logits
