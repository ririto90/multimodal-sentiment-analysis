# multiattfusion3.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import Attention

class MultiAttFusion3(nn.Module):
    """
    Similar to MultiAttFusion, but includes an additional full textual representation
    (averaged over all tokens) in the final classification layer.
    """
    _message_printed = False

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        if not MultiAttFusion3._message_printed:
            print("This model is unfinished")
            MultiAttFusion3._message_printed = True

        # (A) Project the 2048 image vector -> 768
        self.linear_img = nn.Linear(2048, 768)
        self.relu1 = nn.ReLU()

        # (B) Multi-head cross-attention (scaled dot-product)
        #     K=the image vector, Q=the text tokens
        self.attn = Attention(
            embed_dim=768, 
            hidden_dim=None,
            n_head=4, 
            score_function='scaled_dot_product', 
            dropout=0.1
        )

        # (C) MLP: We now concat three things: 
        #     [visual_reps (768) + (cls_rep * attended_text_mean) (768) + cls_rep (768)] = 2304.
        self.linear2 = nn.Linear(2304, 768)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.linear3 = nn.Linear(768, 384)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.25)

        self.linear4 = nn.Linear(384, opt.num_classes)
        # (Optional) self.softmax = nn.Softmax(dim=-1)

    def forward(self, text_features, image_features):
        """
        text_features:  [B, seq_len, 768]  from BERT
        image_features: [B, 49, 2048]      from ResNet
        """
        B, seq_len, _ = text_features.shape

        # 1) CLS from text
        cls_rep = text_features[:, 0, :]  # [B, 768]

        # 2) Average-pool the image features -> [B, 2048]
        img_pooled = image_features.mean(dim=1)

        # 3) Project the image to 768
        visual_reps = self.relu1(self.linear_img(img_pooled))  # [B, 768]

        # 4) Cross-attention: text tokens (Q) attend to single visual vector (K=V)
        k = visual_reps.unsqueeze(1)  # [B, 1, 768]
        q = text_features            # [B, seq_len, 768]
        attended_text = self.attn(k=k, q=q)  # [B, seq_len, 768]

        # 5) Summarize attended text with mean
        attended_text_mean = attended_text.mean(dim=1)  # [B, 768]

        # 6) Multiply with CLS (as in original multiattfusion)
        attn_mult = cls_rep * attended_text_mean  # [B, 768]

        # 7) Also gather the "original" text representation (CLS)
        #    (instead of the full_text_mean)
        text_original = cls_rep  # [B, 768]
        
        # Fuse: [visual_reps + attn_mult] => [B, 1536]
        fused = torch.cat([visual_reps, attn_mult], dim=-1)

        # 9) Feed-forward MLP
        x = self.relu2(self.linear2(fused))  # [B, 768]
        x = self.dropout1(x)
        x = self.relu3(self.linear3(x))      # [B, 384]
        x = self.dropout2(x)
        logits = self.linear4(x)             # [B, num_classes]

        return logits
