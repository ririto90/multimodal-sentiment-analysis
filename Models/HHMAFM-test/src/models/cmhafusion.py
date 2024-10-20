import torch
import torch.nn as nn
import torch.nn.functional as F

class CMHAFUSION(nn.Module):
    def __init__(self, opt):
        super(CMHAFUSION, self).__init__()
        self.opt = opt

        # Define dimensions for each feature set
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

        # Cross-modal fusion layers
        self.global_attention = nn.MultiheadAttention(common_dim, num_heads=4, batch_first=True)
        self.semantic_attention = nn.MultiheadAttention(common_dim, num_heads=4, batch_first=True)

        # Define classifier layers
        self.global_fc = nn.Linear(common_dim, common_dim)
        self.semantic_fc = nn.Linear(common_dim, common_dim)
        self.add_fc = nn.Linear(common_dim, common_dim)
        self.fc1 = nn.Linear(common_dim * 2, common_dim)
        self.fc2 = nn.Linear(common_dim, num_classes)
        
    def forward(self, roberta_text_features, roberta_topic_features, resnet_features, densenet_features):
        
        # Project to a common dimension
        roberta_text_proj = F.relu(self.roberta_text_proj(roberta_text_features))
        roberta_topic_proj = F.relu(self.roberta_topic_proj(roberta_topic_features))
        resnet_proj = F.relu(self.resnet_proj(resnet_features))
        densenet_proj = F.relu(self.densenet_proj(densenet_features))

        # Cross-modal Global Feature Fusion (text + low-level visual)
        low_level_fusion = torch.cat([roberta_text_proj.unsqueeze(1), resnet_proj.unsqueeze(1)], dim=1)
        global_attention_out, _ = self.global_attention(low_level_fusion, low_level_fusion, low_level_fusion)
        global_fusion = F.relu(self.global_fc(global_attention_out[:, 0, :]))

        # Cross-modal High-Level Semantic Fusion (topic + high-level visual)
        query = densenet_proj.unsqueeze(1)  # Image high-level semantic feature as Query
        key_value = roberta_topic_proj.unsqueeze(1)  # Topic feature as Key and Value
        semantic_attention_out, _ = self.semantic_attention(query, key_value, key_value)
        semantic_fusion = F.relu(self.semantic_fc(semantic_attention_out[:, 0, :]))

        # Combine global and semantic fusion features (add)
        combined_fusion = global_fusion + semantic_fusion
        combined_fusion = F.relu(self.add_fc(combined_fusion))

        # Concatenate combined fusion with text features
        final_fusion = torch.cat([combined_fusion, roberta_text_proj], dim=1)
        
        # Pass the fused features through two fully connected layers
        x = F.relu(self.fc1(final_fusion))
        x = self.fc2(x)

        # Apply softmax to obtain the probability distribution over classes
        out = F.softmax(x, dim=1)
        
        return out