import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.attention import SelfAttention
from layers.cross_attention import CrossAttention
from layers.co_attention import CoAttention

class MFCCHFUSION2(nn.Module):
    def __init__(self, opt):
        super(MFCCHFUSION2, self).__init__()
        self.opt = opt

        fc_dim = 256
        text_dim = 768
        visual_dim = 1000
        # self_att_score = opt.self_att_score
        # cross_att_score = opt.cross_att_score
        # co_att_score = opt.co_att_score
        

        hidden_dim = opt.common_dim  # 512, 1024
        num_classes = opt.num_classes  # 3

        # Self-Attention layers (adjusted embed_dim and hidden_dim)
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
            embed_dim=visual_dim,
            hidden_dim=visual_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            q_len=1,
            dropout=opt.dropout_rate
        )
        self.self_attention_densenet = SelfAttention(
            embed_dim=visual_dim,
            hidden_dim=visual_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            q_len=1,
            dropout=opt.dropout_rate
        )

        # Projection layers after self-attention
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.topic_proj = nn.Linear(text_dim, hidden_dim)
        self.resnet_proj = nn.Linear(visual_dim, hidden_dim)
        self.densenet_proj = nn.Linear(visual_dim, hidden_dim)

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
        self.co_attention_text_topic = CoAttention(
            embed_dim1=hidden_dim,
            embed_dim2=hidden_dim,
            hidden_dim=hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            dropout=opt.dropout_rate
        )
        self.co_attention_resnet_densenet = CoAttention(
            embed_dim1=hidden_dim,
            embed_dim2=hidden_dim,
            hidden_dim=hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            dropout=opt.dropout_rate
        )

        # Classifier input dimension
        # Fully connected layers at the end
        self.fc1 = nn.Linear(hidden_dim * 2, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.classifier = nn.Linear(fc_dim, num_classes)
        
        # self.classifier = nn.Linear(hidden_dim * 4, num_classes)

    def forward(self, roberta_text_features, roberta_topic_features, resnet_features, densenet_features):

        # Add sequence dimension
        roberta_text_features = roberta_text_features.unsqueeze(1)
        roberta_topic_features = roberta_topic_features.unsqueeze(1)
        resnet_features = resnet_features.unsqueeze(1)
        densenet_features = densenet_features.unsqueeze(1)

        # Apply self-attention
        text_attended = self.self_attention_text(roberta_text_features).squeeze(1)
        topic_attended = self.self_attention_topic(roberta_topic_features).squeeze(1)
        resnet_attended = self.self_attention_resnet(resnet_features).squeeze(1)
        densenet_attended = self.self_attention_densenet(densenet_features).squeeze(1)

        # Project outputs to a common hidden dimension
        text_attended_proj = self.text_proj(text_attended)
        topic_attended_proj = self.topic_proj(topic_attended)
        resnet_attended_proj = self.resnet_proj(resnet_attended)
        densenet_attended_proj = self.densenet_proj(densenet_attended)

        # Apply cross-attention
        text_resnet_attended, _ = self.cross_attention_text_resnet(
            query=text_attended_proj.unsqueeze(1),
            key=resnet_attended_proj.unsqueeze(1),
            value=resnet_attended_proj.unsqueeze(1)
        )
        topic_densenet_attended, _ = self.cross_attention_topic_densenet(
            query=topic_attended_proj.unsqueeze(1),
            key=densenet_attended_proj.unsqueeze(1),
            value=densenet_attended_proj.unsqueeze(1)
        )

        # Squeeze the sequence dimension
        text_resnet_attended = text_resnet_attended.squeeze(1)
        topic_densenet_attended = topic_densenet_attended.squeeze(1)

        # Apply co-attention between text and topic
        co_attended_text_topic_out1, co_attended_text_topic_out2, _, _ = self.co_attention_text_topic(
            x1=text_attended_proj.unsqueeze(1),
            x2=topic_attended_proj.unsqueeze(1)
        )
        # Combine the two outputs
        co_attended_text_topic = (co_attended_text_topic_out1 + co_attended_text_topic_out2).squeeze(1) / 2

        # Apply co-attention between resnet and densenet
        co_attended_resnet_densenet_out1, co_attended_resnet_densenet_out2, _, _ = self.co_attention_resnet_densenet(
            x1=resnet_attended_proj.unsqueeze(1),
            x2=densenet_attended_proj.unsqueeze(1)
        )
        co_attended_resnet_densenet = (co_attended_resnet_densenet_out1 + co_attended_resnet_densenet_out2).squeeze(1) / 2

        # Concatenate the outputs from cross-attention and co-attention
        
        mult_1 = (
            text_resnet_attended *
            co_attended_text_topic
        )

        mult_2 = (
            topic_densenet_attended *
            co_attended_resnet_densenet
        )
        
        fusion = torch.cat([
            mult_1,
            mult_2
        ], dim=1)

        x = F.relu(self.fc1(fusion))
        x = F.relu(self.fc2(x))
        out = self.classifier(x)

        return out
