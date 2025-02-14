import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFusion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.num_classes = opt.num_classes
        self.hidden_dim = opt.hidden_dim
        
        # BERT gives 768-d pooler_output; ResNet yields a 2048-d vector
        self.text_dim = 768
        self.image_dim = 2048
        
        # Projections
        self.text_proj = nn.Linear(self.text_dim, self.hidden_dim)
        self.image_proj = nn.Linear(self.image_dim, self.hidden_dim)
        
        # Classifier over concatenation
        self.classifier = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def forward(self, text_feature, image_feature):
        """
        text_feature:  [B, 768]
        image_feature: [B, 2048]
        return: logits [B, num_classes]
        """
        # Project text & image
        t_proj = F.relu(self.text_proj(text_feature))
        i_proj = F.relu(self.image_proj(image_feature))

        # Simple concatenation fusion
        fused = torch.cat([t_proj, i_proj], dim=-1)  # [B, 2 * hidden_dim]

        # Classify
        logits = self.classifier(fused)  # [B, num_classes]
        return logits