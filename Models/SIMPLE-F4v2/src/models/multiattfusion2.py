# multiattfusion2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attention import Attention

class MultiAttFusion2(nn.Module):
    _message_printed = False
    
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        if not MultiAttFusion2._message_printed:
            print("This model changes the final linear layer from 384 to 768")
            MultiAttFusion2._message_printed = True
        
        # (A) Linear: project the 2048 image vector -> 768
        self.linear_img = nn.Linear(2048, 768)
        self.relu1 = nn.ReLU()

        # (B) Multi-head cross-attention
        self.attn = Attention(
            embed_dim=768, 
            n_head=4, 
            score_function='scaled_dot_product', 
            dropout=0.1
        )

        # (C) MLP: after combining [visual_reps + attended_text], go directly to 768
        # 1) From 1536 -> 768
        self.linear2 = nn.Linear(1536, 768)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        # 2) Directly 768 -> num_classes (no 384 step)
        self.linear3 = nn.Linear(768, opt.num_classes)

    def forward(self, text_features, image_features):
        B, seq_len, _ = text_features.shape
        cls_rep = text_features[:, 0, :]             # [B, 768]
        img_pooled = image_features.mean(dim=1)      # [B, 2048]
        visual_reps = self.relu1(self.linear_img(img_pooled))  # [B, 768]

        # Cross-modal attention
        k = visual_reps.unsqueeze(1)       # [B, 1, 768]
        q = text_features                  # [B, seq_len, 768]
        attended_text = self.attn(k=k, q=q).mean(dim=1)  # [B, 768]

        # Combine CLS with attended text
        attn_mult = cls_rep * attended_text
        multimodal_reps = torch.cat([visual_reps, attn_mult], dim=-1)  # [B, 1536]

        # Feed-forward
        x = self.relu2(self.linear2(multimodal_reps))
        x = self.dropout1(x)
        logits = self.linear3(x)
        return logits