# multiattfusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiAttFusion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        # A single linear layer to fuse text[768] + image[2048] => num_classes
        self.fc = nn.Linear(768 + 2048, opt.num_classes)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, text_features, image_features):
        """
        text_features:  [B, seq_len, 768]  (BERT output, we typically take the [CLS] token)
        image_features: [B, 49, 2048]      (CNN feature map, we pool across regions)
        """

        # 1) Take the [CLS] vector from text (the first token):
        #    If your BERT output is [B, seq_len, 768], "text_features[:, 0, :]" is your CLS embedding.
        text_cls = text_features[:, 0, :]  # [B, 768]

        # 2) Average pool the image features across the 49 regions:
        image_pooled = torch.mean(image_features, dim=1)  # [B, 2048]

        # 3) Concatenate for final fusion:
        fusion_output = torch.cat([text_cls, image_pooled], dim=-1)  # [B, 768+2048=2816]

        # 4) A small nonlinearity + dropout + final FC classification
        fusion_output = F.relu(fusion_output)
        logits = self.fc(self.dropout(fusion_output))  # [B, num_classes]

        return logits
