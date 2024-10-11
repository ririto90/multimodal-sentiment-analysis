import torch
import torch.nn as nn
import torch.nn.functional as F

class MMFUSION(nn.Module):
    def __init__(self, opt):
        super(MMFUSION, self).__init__()
        self.opt = opt

        # Define dimensions for each feature set
        roberta_text_feature_dim = opt.roberta_text_feature_dim  # e.g., 768
        roberta_topic_feature_dim = opt.roberta_topic_feature_dim  # e.g., 50
        resnet_feature_dim = opt.resnet_feature_dim  # e.g., 2048
        densenet_feature_dim = opt.densenet_feature_dim  # e.g., 1024

        common_dim = opt.common_dim  # e.g., 512
        num_classes = opt.num_classes  # Number of output classes

        # Define projection layers to a common dimension
        self.roberta_text_proj = nn.Linear(roberta_text_feature_dim, common_dim)
        self.roberta_topic_proj = nn.Linear(roberta_topic_feature_dim, common_dim)
        self.resnet_proj = nn.Linear(resnet_feature_dim, common_dim)
        self.densenet_proj = nn.Linear(densenet_feature_dim, common_dim)

        # Define the classifier
        self.classifier = nn.Linear(common_dim * 4, num_classes)
        
    def forward(self, roberta_text_features, roberta_topic_features, resnet_features, densenet_features):
        # Project each feature set to the common dimension
        roberta_text_proj = F.relu(self.roberta_text_proj(roberta_text_features))
        roberta_topic_proj = F.relu(self.roberta_topic_proj(roberta_topic_features))
        resnet_proj = F.relu(self.resnet_proj(resnet_features))
        densenet_proj = F.relu(self.densenet_proj(densenet_features))

        # Concatenate the projected features
        fusion = torch.cat([roberta_text_proj, roberta_topic_proj, resnet_proj, densenet_proj], dim=1)

        # Pass the fused features through the classifier
        out = self.classifier(fusion)
        return out
