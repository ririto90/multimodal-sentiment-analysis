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

        # Fully connected layer for classification
        self.fc = nn.Linear(768 + 2048, opt.num_classes)

        # Dropout
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, text_feature, image_feature):
        """
        text_feature:  [B, 768]   -> Text embedding from BERT
        image_feature: [B, 2048]  -> Image embedding from ResNet

        return: logits [B, num_classes]
        """

        # Add sequence dimension
        text_feature = text_feature.unsqueeze(1)  # [B, 1, 768]
        image_feature = image_feature.unsqueeze(1)  # [B, 1, 2048]

        # Self-attention for text and image
        text_attended = self.text_self_attention(text_feature)  # [B, 1, 768]
        image_attended = self.image_self_attention(image_feature)  # [B, 1, 2048]

        # Cross-attention (text attends to image and vice versa)
        text_crossed, _ = self.text_to_image_attention(text_attended, image_attended, image_attended)  # [B, 1, 768]
        image_crossed, _ = self.image_to_text_attention(image_attended, text_attended, text_attended)  # [B, 1, 2048]

        # Remove sequence dimension
        text_crossed = text_crossed.squeeze(1)  # [B, 768]
        image_crossed = image_crossed.squeeze(1)  # [B, 2048]

        # Concatenate fused features
        fusion_output = torch.cat([text_crossed, image_crossed], dim=-1)  # [B, 2816]

        # Pass through fully connected layer
        logits = self.fc(self.dropout(fusion_output))  # [B, num_classes]

        return logits