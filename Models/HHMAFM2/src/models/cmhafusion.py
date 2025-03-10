import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attention import SelfAttention, Attention
from layers.cross_attention import CrossAttention
from layers.cnn import LowLevelCNN

class CMHAFUSION(nn.Module):
    def __init__(self, opt):
        super(CMHAFUSION, self).__init__()
        self.opt = opt

        # Define dimensions for each feature set
        text_dim = 768
        visual_dim = 1000

        hidden_dim = opt.hidden_dim  # Example: 512 or 1024
        num_classes = opt.num_classes  # Example: 3

        # Projection layers to a common hidden dimension
        self.roberta_text_proj = nn.Linear(text_dim, hidden_dim)
        self.roberta_topic_proj = nn.Linear(text_dim, hidden_dim)
        
        self.densenet_mlp = nn.Sequential(
            nn.Linear(1024, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.self_att_global = SelfAttention(
            embed_dim=2 * hidden_dim,
            hidden_dim=2 * hidden_dim,
            n_head=self.opt.n_head,
            score_function='scaled_dot_product',
            q_len=1,
            dropout=self.opt.dropout_rate
        )
        self.semantic_attention = Attention(
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            dropout=opt.dropout_rate
        )

        # Feed Forward Networks (FFN) after attention
        self.global_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.semantic_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.custom_cnn = LowLevelCNN(hidden_dim)

        # LayerNorm layers 
        self.semantic_attention_norm = nn.LayerNorm(hidden_dim)
        self.semantic_ffn_norm = nn.LayerNorm(hidden_dim)
        self.semantic_final_norm = nn.LayerNorm(hidden_dim)
        
        self.semantic_output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # classification 
        self.add_fc = nn.Linear(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, roberta_text_features, roberta_topic_features, _, densenet_features, images):
        
        # Feautres
        roberta_text_proj = F.relu(self.roberta_text_proj(roberta_text_features))
        roberta_topic_proj = F.relu(self.roberta_topic_proj(roberta_topic_features))
        
        low_level_features = self.custom_cnn(images)
        low_level_proj = F.relu(low_level_features)
        
        densenet_mlp_output = self.densenet_mlp(densenet_features)
        densenet_proj = F.relu(densenet_mlp_output)

        ### Cross-modal Global Feature Fusion (Text + Low-level Visual) ###
        # Concatenate text and low-level visual features
        low_level_fusion = torch.cat([roberta_text_proj, low_level_proj], dim=1)
        low_level_features_seq = low_level_fusion.unsqueeze(1)
        
        # Self Attention
        global_attention_out = self.self_att_global(low_level_features_seq)  # Returns only output
        global_attention_out = global_attention_out.squeeze(1)

        # MLP
        global_fusion = self.global_mlp(global_attention_out)

        ### Cross-modal High-Level Semantic Fusion (Topic + High-level Visual) ###
        # Prepare query, key, and value tensors
        query = densenet_proj.unsqueeze(1)  # High-level visual features
        key_value = roberta_topic_proj.unsqueeze(1)  # Topic features

        # Apply multi-head attention
        semantic_attention_out = self.semantic_attention(k=key_value, q=query)

        # Add residual connection and apply LayerNorm
        semantic_fusion = self.semantic_attention_norm(semantic_attention_out + query)
        semantic_fusion = semantic_fusion.squeeze(1)
        
        semantic_fusion_clone = semantic_fusion.clone()
        
        semantic_fusion_mlp = self.semantic_mlp(semantic_fusion)
        
        # Add residual connection and apply LayerNorm
        semantic_fusion = self.semantic_ffn_norm(semantic_fusion_mlp + semantic_fusion)  # [batch_size, hidden_dim]

        # Third Residual Connection and LayerNorm
        semantic_fusion = self.semantic_final_norm(semantic_fusion + semantic_fusion_clone)  # [batch_size, hidden_dim]

        # Add sequence dimension for Global Max Pooling
        semantic_fusion_seq = semantic_fusion.unsqueeze(1)
        
        O_i = torch.max(semantic_fusion_seq, dim=1).values
        
        # Final MLP to obtain the high-level semantic feature
        semantic_fusion = self.semantic_output_mlp(O_i)

        ### Combine Global and Semantic Fusion Features ###
        combined_fusion = global_fusion + semantic_fusion

        # Concatenate combined fusion with text features
        final_fusion = torch.cat([combined_fusion, roberta_text_proj], dim=1)

        # Final classification layers
        x = F.relu(self.fc1(final_fusion))
        x = self.fc2(x)
        
        return x
