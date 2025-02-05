import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFusion(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.num_classes = opt.num_classes
        self.hidden_dim = opt.hidden_dim
        self.text_dim = 1536
        self.image_dim = 2048
        
        # Projection
        self.text_proj = nn.Linear(self.text_dim, self.hidden_dim)
        self.image_proj = nn.Linear(self.image_dim, self.hidden_dim)
        
        # Classifier
        self.classifier = nn.Linear(self.hidden_dim * 2, self.num_classes)

    def forward(self, text_feature, image_feature):
        """
        :param text_features: [B, 1536] raw text embeddings
        :param image_features: [B, 2048, H, W] image feature map
        :return: logits [B, opt.num_classes]
        """
        # Global Average Pool
        pooled_image = F.adaptive_avg_pool2d(image_feature, 1).squeeze(-1).squeeze(-1)  # [B, 2048]

        # Projection
        t_proj = F.relu(self.text_proj(text_feature))
        i_proj = F.relu(self.image_proj(pooled_image))

        # Fusion (Concatenation)
        fused = torch.cat([t_proj, i_proj], dim=-1)  # [B, 2 * hidden_dim]

        # Classify
        logits = self.classifier(fused)  # [B, num_classes]
        return logits