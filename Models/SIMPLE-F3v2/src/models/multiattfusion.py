import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attention import SelfAttention
from layers.cross_attention import CrossAttention

class MultiAttFusion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        # Self-attention for text and image features
        self.text_self_attention = SelfAttention(embed_dim=768, n_head=8, score_function='scaled_dot_product')
        self.image_self_attention = SelfAttention(embed_dim=2048, n_head=8, score_function='scaled_dot_product')

        # Cross-attention between text and image features
        self.text_to_image_attention = CrossAttention(embed_dim_q=768, embed_dim_kv=2048, n_head=8, score_function='scaled_dot_product')
        self.image_to_text_attention = CrossAttention(embed_dim_q=2048, embed_dim_kv=768, n_head=8, score_function='scaled_dot_product')

        # Additional combination attention layer after fusion
        self.fusion_attention = SelfAttention(embed_dim=2816, n_head=8, score_function='scaled_dot_product')

        # Fully connected layer for classification
        self.fc = nn.Linear(2816, opt.num_classes)

        # Dropout
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, text_features, image_features):
        """
        text_features:  [B, seq_len, 768]
        image_features: [B, 49, 2048]
        """

        # 1) Self-attention over text tokens
        #    text_self_attention expects [B, seq_len, 768]
        text_attended = self.text_self_attention(text_features)    # [B, seq_len, 768]

        # 2) Self-attention over image regions
        #    image_self_attention expects [B, 49, 2048]
        image_attended = self.image_self_attention(image_features) # [B, 49, 2048]

        # (Optional) If you want to multiply original features by attended results
        # it should be done elementwise:
        text_weighted = text_attended * text_features    # [B, seq_len, 768]
        image_weighted = image_attended * image_features # [B, 49, 2048]

        # 3) Cross-attention
        # text_to_image_attention expects (query=[B, seq_len, 768], key=[B, 49, 2048], value=[B, 49, 2048])
        text_crossed, _ = self.text_to_image_attention(
            text_weighted, image_weighted, image_weighted
        )  # [B, seq_len, 768]

        # image_to_text_attention expects (query=[B, 49, 2048], key=[B, seq_len, 768], value=[B, seq_len, 768])
        image_crossed, _ = self.image_to_text_attention(
            image_weighted, text_weighted, text_weighted
        )  # [B, 49, 2048]

        # 4) Flatten or pool the final cross-attended features
        #    The simplest approach: average-pool each modality’s tokens/regions
        text_pooled  = torch.mean(text_crossed, dim=1)   # [B, 768]
        image_pooled = torch.mean(image_crossed, dim=1)  # [B, 2048]

        # 5) Concatenate for final fusion
        fusion_output = torch.cat([text_pooled, image_pooled], dim=-1)  # [B, 2816]

        # 6) Optionally apply the additional fusion_attention across a sequence dimension,
        #    but if you only have 1 fusion vector per sample, you can skip or adapt it
        # fusion_output = fusion_output.unsqueeze(1)       # [B, 1, 2816]
        # fusion_output = self.fusion_attention(fusion_output)
        # fusion_output = fusion_output.squeeze(1)         # [B, 2816]

        fusion_output = torch.tanh(fusion_output)
        logits = self.fc(self.dropout(fusion_output))  # [B, num_classes]
        return logits