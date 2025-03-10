# multiattfusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.attention import Attention

class MultiAttFusion(nn.Module):
    _message_printed = False
    """
    Recreates the old SentimentClassifier logic, but uses:
      - ResNet features instead of VGG+AlexNet.
      - Multi-head cross-attention from attention.py (instead of old single-head).
    """
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        if not MultiAttFusion._message_printed:
            print("This model replicates MOA with scaled dot product attention, and BERT-RESNET")
            MultiAttFusion._message_printed = True
        
        # (A) Linear: project the 2048 image vector -> 768
        self.linear_img = nn.Linear(2048, 768)
        self.relu1 = nn.ReLU()

        # (B) Multi-head cross-attention
        # We pass K=the image vector, Q=the text tokens
        # embed_dim=768, n_head could be 1..4, etc.
        self.attn = Attention(
            embed_dim=768, 
            hidden_dim=None,
            n_head=4, 
            score_function='scaled_dot_product', 
            dropout=0.1
        )

        # (C) MLP: after we combine [visual_reps + attended_text], do a few layers
        # 1) Expand from 768+768 = 1536 down to 768
        self.linear2 = nn.Linear(1536, 768)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        # 2) Then 768 -> 384
        self.linear3 = nn.Linear(768, 384)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.25)

        # 3) Finally 384 -> num_classes
        self.linear4 = nn.Linear(384, opt.num_classes)

        # We keep a softmax here, but for CrossEntropyLoss usage, 
        # the forward returns raw logits from linear4
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, text_features, image_features):
        """
        text_features:  [B, seq_len, 768]  from BERT
        image_features: [B, 49, 2048]      from ResNet
        """
        B, seq_len, _ = text_features.shape
        # 1) Get the [CLS] embedding from text
        cls_rep = text_features[:, 0, :]   # [B, 768]

        # 2) Average-pool image features -> [B, 2048]
        img_pooled = image_features.mean(dim=1)
        
        # 3) Project image to 768
        visual_reps = self.relu1(self.linear_img(img_pooled))  # [B, 768]

        # 4) Cross-modal attention: Let the text tokens (Q) attend over the single visual vector (K=V)
        k = visual_reps.unsqueeze(1)       # [B, 1, 768]
        q = text_features                  # [B, seq_len, 768]
        attended_text = self.attn(k=k, q=q)  # [B, seq_len, 768]

        # 5) Summarize the attended text.
        attended_text_mean = attended_text.mean(dim=1)  # [B, 768]

        # Multiply with the original CLS embedding
        attn_mult = cls_rep * attended_text_mean  # [B, 768]

        # 6) Concat that with the projected image
        multimodal_reps = torch.cat([visual_reps, attn_mult], dim=-1)  # [B, 1536]

        # 7) Pass it through a small feed-forward MLP
        x = self.relu2(self.linear2(multimodal_reps))
        x = self.dropout1(x)
        x = self.relu3(self.linear3(x))
        x = self.dropout2(x)
        logits = self.linear4(x)  # raw logits for CrossEntropy

        return logits