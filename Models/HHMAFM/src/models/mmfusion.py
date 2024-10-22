from layers.attention import SelfAttention
from layers.cross_attention import CrossAttention
import torch
import torch.nn as nn
import torch.nn.functional as F

class MMFUSION(nn.Module):
    def __init__(self, opt):
        super(MMFUSION, self).__init__()
        self.opt = opt

        text_dim = 768
        visual_dim = 1000

        hidden_dim = opt.common_dim  # 512
        num_classes = opt.num_classes  # 3

        # Self Attention for each individual feature representation
        self.self_attention_text = SelfAttention(
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            q_len=1,
            dropout=opt.dropout_rate
        )
        self.self_attention_topic = SelfAttention(
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            q_len=1,
            dropout=opt.dropout_rate
        )
        self.self_attention_resnet = SelfAttention(
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            q_len=1,
            dropout=opt.dropout_rate
        )
        self.self_attention_densenet = SelfAttention(
            embed_dim=hidden_dim,
            hidden_dim=hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            q_len=1,
            dropout=opt.dropout_rate
        )
        
        # Cross-Attention layers
        self.cross_attention_text_resnet = CrossAttention(
            embed_dim_q=hidden_dim,
            embed_dim_kv=hidden_dim,
            hidden_dim=hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            dropout=opt.dropout_rate
        )
        self.cross_attention_topic_densenet = CrossAttention(
            embed_dim_q=hidden_dim,
            embed_dim_kv=hidden_dim,
            hidden_dim=hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            dropout=opt.dropout_rate
        )

        # Projection layers to a common dimension
        self.roberta_text_proj = nn.Linear(text_dim, hidden_dim)
        self.roberta_topic_proj = nn.Linear(text_dim, hidden_dim)
        self.resnet_proj = nn.Linear(visual_dim, hidden_dim)
        self.densenet_proj = nn.Linear(visual_dim, hidden_dim)

        self.classifier = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, roberta_text_features, roberta_topic_features, resnet_features, densenet_features):
        # Project each feature set to the common dimension
        roberta_text_proj = F.relu(self.roberta_text_proj(roberta_text_features))
        roberta_topic_proj = F.relu(self.roberta_topic_proj(roberta_topic_features))
        resnet_proj = F.relu(self.resnet_proj(resnet_features))
        densenet_proj = F.relu(self.densenet_proj(densenet_features))

        # Add sequence dimension
        roberta_text_proj = roberta_text_proj.unsqueeze(1)
        roberta_topic_proj = roberta_topic_proj.unsqueeze(1)
        resnet_proj = resnet_proj.unsqueeze(1)
        densenet_proj = densenet_proj.unsqueeze(1)

        # Apply self-attention
        text_attended = self.self_attention_text(roberta_text_proj).squeeze(1)    
        topic_attended = self.self_attention_topic(roberta_topic_proj).squeeze(1) 
        resnet_attended = self.self_attention_resnet(resnet_proj).squeeze(1)      
        densenet_attended = self.self_attention_densenet(densenet_proj).squeeze(1)

        # Apply cross-attention
        text_resnet_attended, _ = self.cross_attention_text_resnet(
            query=text_attended.unsqueeze(1),
            key=resnet_attended.unsqueeze(1),
            value=resnet_attended.unsqueeze(1)
        )
        topic_densenet_attended, _ = self.cross_attention_topic_densenet(
            query=topic_attended.unsqueeze(1),
            key=densenet_attended.unsqueeze(1),
            value=densenet_attended.unsqueeze(1)
        )

        # Squeeze the sequence dimension
        text_resnet_attended = text_resnet_attended.squeeze(1)
        topic_densenet_attended = topic_densenet_attended.squeeze(1)

        # Concatenate the self-attended features
        fusion = torch.cat([text_resnet_attended, topic_densenet_attended], dim=1)

        # Pass the fused features through the classifier
        out = self.classifier(fusion)
        return out