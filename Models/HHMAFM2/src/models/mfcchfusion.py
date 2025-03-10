import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attention import SelfAttention
from layers.cross_attention import CrossAttention
from layers.co_attention import CoAttention

class MFCCHFUSION(nn.Module):
    def __init__(self, opt, text_dim, resnet_dim, densenet_dim):
        super(MFCCHFUSION, self).__init__()
        self.opt = opt
        self.first_batch = True

        hidden_dim = opt.hidden_dim  # 512, 1024
        num_classes = opt.num_classes  # 3

        # Self-Attention layers
        self.self_attention_text = SelfAttention(
            embed_dim=text_dim,
            hidden_dim=text_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            q_len=1,
            dropout=opt.dropout_rate
        )
        self.self_attention_topic = SelfAttention(
            embed_dim=text_dim,
            hidden_dim=text_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            q_len=1,
            dropout=opt.dropout_rate
        )
        self.self_attention_resnet = SelfAttention(
            embed_dim=resnet_dim,
            hidden_dim=resnet_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            q_len=1,
            dropout=opt.dropout_rate
        )
        self.self_attention_densenet = SelfAttention(
            embed_dim=densenet_dim,
            hidden_dim=densenet_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            q_len=1,
            dropout=opt.dropout_rate
        )
        
        # Projection for Cross-Attention to common dimension
        self.roberta_text_proj = nn.Linear(text_dim, hidden_dim)
        self.roberta_topic_proj = nn.Linear(text_dim, hidden_dim)
        self.resnet_proj = nn.Linear(resnet_dim, hidden_dim)
        self.densenet_proj = nn.Linear(densenet_dim, hidden_dim)

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

        # Co-Attention layers
        self.co_attention_text_resnet = CoAttention(
            embed_dim1=hidden_dim,
            embed_dim2=hidden_dim,
            hidden_dim=hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            dropout=opt.dropout_rate
        )
        self.co_attention_topic_densenet = CoAttention(
            embed_dim1=hidden_dim,
            embed_dim2=hidden_dim,
            hidden_dim=hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            dropout=opt.dropout_rate
        )

        # Classifier input dimension
        fusion_dim = text_dim + 4 * hidden_dim
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, roberta_text_features, roberta_topic_features, resnet_features, densenet_features, _):
        
        text_proj = self.roberta_text_proj(roberta_text_features)
        
        ### Self-attention on original features ###
        text_self_attended = self.self_attention_text(roberta_text_features.unsqueeze(1)).squeeze(1)
        topic_self_attended = self.self_attention_topic(roberta_topic_features.unsqueeze(1)).squeeze(1)
        resnet_self_attended = self.self_attention_resnet(resnet_features.unsqueeze(1)).squeeze(1)
        densenet_self_attended = self.self_attention_densenet(densenet_features.unsqueeze(1)).squeeze(1)

        # Project to common hidden dimension (with activation)
        # text_proj = F.relu(self.roberta_text_proj(text_self_attended))
        # topic_proj = F.relu(self.roberta_topic_proj(topic_self_attended))
        # resnet_proj = F.relu(self.resnet_proj(resnet_self_attended))
        # densenet_proj = F.relu(self.densenet_proj(densenet_self_attended))
        
        # Project to common hidden dimension (without activation)
        text_proj = self.roberta_text_proj(text_self_attended)
        topic_proj = self.roberta_topic_proj(topic_self_attended)
        resnet_proj = self.resnet_proj(resnet_self_attended)
        densenet_proj = self.densenet_proj(densenet_self_attended)

        ### Apply Cross-Attention ###
        # Cross-attention (text and resnet)
        text_resnet_attended, _ = self.cross_attention_text_resnet(
            query=text_proj.unsqueeze(1),
            key=resnet_proj.unsqueeze(1),
            value=resnet_proj.unsqueeze(1)
        )
        text_resnet_attended = text_resnet_attended.squeeze(1)

        # Cross-attention (topic and densenet)
        topic_densenet_attended, _ = self.cross_attention_topic_densenet(
            query=topic_proj.unsqueeze(1),
            key=densenet_proj.unsqueeze(1),
            value=densenet_proj.unsqueeze(1)
        )
        topic_densenet_attended = topic_densenet_attended.squeeze(1)

        ### Apply Co-Attention ###
        # Apply co-attention between text and ResNet features
        co_attended_text_resnet_out1, co_attended_text_resnet_out2, _, _ = self.co_attention_text_resnet(
            x1=text_proj.unsqueeze(1),
            x2=resnet_proj.unsqueeze(1)
        )
        co_attended_text_resnet = (co_attended_text_resnet_out1 + co_attended_text_resnet_out2).squeeze(1) / 2

        # Apply co-attention between topic and DenseNet features
        co_attended_topic_densenet_out1, co_attended_topic_densenet_out2, _, _ = self.co_attention_topic_densenet(
            x1=topic_proj.unsqueeze(1),
            x2=densenet_proj.unsqueeze(1)
        )
        co_attended_topic_densenet = (co_attended_topic_densenet_out1 + co_attended_topic_densenet_out2).squeeze(1) / 2

        # Concatenate the outputs
        fusion = torch.cat([
            roberta_text_features,
            text_resnet_attended,
            topic_densenet_attended,
            co_attended_text_resnet,
            co_attended_topic_densenet
        ], dim=1)

        # Classifier
        out = self.classifier(fusion)
        return out