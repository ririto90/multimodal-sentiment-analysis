import torch
import torch.nn as nn
import torch.nn.functional as F

class MMFUSION(nn.Module):
    def __init__(self, opt):
        super(MMFUSION, self).__init__()
        self.opt = opt

        roberta_text_feature_dim = 768
        roberta_topic_feature_dim = 768
        resnet_feature_dim = 1000
        densenet_feature_dim = 1000

        common_dim = opt.common_dim  # e.g., 512
        num_classes = opt.num_classes  # Number of output classes

        # Define projection layers to a common dimension
        self.roberta_text_proj = nn.Linear(roberta_text_feature_dim, common_dim)
        self.roberta_topic_proj = nn.Linear(roberta_topic_feature_dim, common_dim)
        self.resnet_proj = nn.Linear(resnet_feature_dim, common_dim)
        self.densenet_proj = nn.Linear(densenet_feature_dim, common_dim)

        self.classifier = nn.Linear(common_dim * 4, num_classes)
        
    def forward(self, roberta_text_features, roberta_topic_features, resnet_features, densenet_features):
        
        # Print dimensions of the inputs before projection
        # print(f"roberta_text_features shape before projection: {roberta_text_features.shape}")
        # print(f"roberta_topic_features shape before projection: {roberta_topic_features.shape}")
        # print(f"resnet_features shape before projection: {resnet_features.shape}")
        # print(f"densenet_features shape before projection: {densenet_features.shape}")
        
        # Project each feature set to the common dimension
        roberta_text_proj = F.relu(self.roberta_text_proj(roberta_text_features))
        roberta_topic_proj = F.relu(self.roberta_topic_proj(roberta_topic_features))
        resnet_proj = F.relu(self.resnet_proj(resnet_features))
        densenet_proj = F.relu(self.densenet_proj(densenet_features))
        
        # Print dimensions of the outputs for debugging
        # print(f"roberta_text_proj shape: {roberta_text_proj.shape}")
        # print(f"roberta_topic_proj shape: {roberta_topic_proj.shape}")
        # print(f"resnet_proj shape: {resnet_proj.shape}")
        # print(f"densenet_proj shape: {densenet_proj.shape}")

        # Concatenate the projected features
        fusion = torch.cat([roberta_text_proj, roberta_topic_proj, resnet_proj, densenet_proj], dim=1)

        # Pass the fused features through the classifier
        out = self.classifier(fusion)
        return out
