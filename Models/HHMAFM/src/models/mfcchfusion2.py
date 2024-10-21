from layers.attention import SelfAttention  # Import the SelfAttention class
import torch
import torch.nn as nn
import torch.nn.functional as F

# Remove your custom SelfAttention class definition

class CrossAttention(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim):
        super(CrossAttention, self).__init__()
        self.query = nn.Linear(input_dim1, hidden_dim)
        self.key = nn.Linear(input_dim2, hidden_dim)
        self.value = nn.Linear(input_dim2, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        attention_weights = self.softmax(q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5))
        attended_output = attention_weights @ v
        return attended_output

class GatedAdjustment(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GatedAdjustment, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        gate = self.sigmoid(self.linear(x))
        adjusted_output = gate * x
        return adjusted_output

class CoAttention(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim):
        super(CoAttention, self).__init__()
        self.query1 = nn.Linear(input_dim1, hidden_dim)
        self.key1 = nn.Linear(input_dim2, hidden_dim)
        self.value1 = nn.Linear(input_dim2, hidden_dim)
        self.query2 = nn.Linear(input_dim2, hidden_dim)
        self.key2 = nn.Linear(input_dim1, hidden_dim)
        self.value2 = nn.Linear(input_dim1, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x1, x2):
        # Co-Attention for x1 attending to x2
        q1 = self.query1(x1)
        k1 = self.key1(x2)
        v1 = self.value1(x2)
        attention_weights1 = self.softmax(q1 @ k1.transpose(-2, -1) / (k1.size(-1) ** 0.5))
        attended_x1 = attention_weights1 @ v1
        
        # Co-Attention for x2 attending to x1
        q2 = self.query2(x2)
        k2 = self.key2(x1)
        v2 = self.value2(x1)
        attention_weights2 = self.softmax(q2 @ k2.transpose(-2, -1) / (k2.size(-1) ** 0.5))
        attended_x2 = attention_weights2 @ v2
        
        return attended_x1, attended_x2

class MFCCHFUSION2(nn.Module):
    def __init__(self, opt):
        super(MFCCHFUSION2, self).__init__()
        self.opt = opt
        
        text_dim = 768
        visual_dim = 1000
        
        hidden_dim = opt.common_dim  # e.g., 512
        num_classes = opt.num_classes  # Number of output classes
        
        # Self Attention for each individual feature representation using the imported SelfAttention
        self.self_attention_text = SelfAttention(
            embed_dim=text_dim,
            hidden_dim=hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            q_len=1,
            dropout=opt.dropout
        )
        self.self_attention_topic = SelfAttention(
            embed_dim=text_dim,
            hidden_dim=hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            q_len=1,
            dropout=opt.dropout
        )
        self.self_attention_resnet = SelfAttention(
            embed_dim=visual_dim,
            hidden_dim=hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            q_len=1,
            dropout=opt.dropout
        )
        self.self_attention_densenet = SelfAttention(
            embed_dim=visual_dim,
            hidden_dim=hidden_dim,
            n_head=opt.n_head,
            score_function='scaled_dot_product',
            q_len=1,
            dropout=opt.dropout
        )
        
        # Gated Adjustment for each representation
        self.gated_adjustment_text = GatedAdjustment(hidden_dim, hidden_dim)
        self.gated_adjustment_topic = GatedAdjustment(hidden_dim, hidden_dim)
        self.gated_adjustment_resnet = GatedAdjustment(hidden_dim, hidden_dim)
        self.gated_adjustment_densenet = GatedAdjustment(hidden_dim, hidden_dim)
        
        # Cross Attention between modalities
        self.cross_attention_topic_densenet = CrossAttention(hidden_dim, hidden_dim, hidden_dim)
        self.cross_attention_sentence_resnet = CrossAttention(hidden_dim, hidden_dim, hidden_dim)
        
        # Co-Attention between modalities
        self.co_attention_topic_densenet = CoAttention(hidden_dim, hidden_dim, hidden_dim)
        self.co_attention_sentence_resnet = CoAttention(hidden_dim, hidden_dim, hidden_dim)
        
        # Final classifier
        self.classifier = nn.Linear(6 * hidden_dim, num_classes)
    
    def forward(self, roberta_text_features, roberta_topic_features, resnet_features, densenet_features):
        # Step 1: Self Attention using the imported SelfAttention class
        text_attended = self.self_attention_text(roberta_text_features)
        topic_attended = self.self_attention_topic(roberta_topic_features)
        resnet_attended = self.self_attention_resnet(resnet_features)
        densenet_attended = self.self_attention_densenet(densenet_features)
        
        # Step 2: Gated Adjustment
        text_adjusted = self.gated_adjustment_text(text_attended)
        topic_adjusted = self.gated_adjustment_topic(topic_attended)
        resnet_adjusted = self.gated_adjustment_resnet(resnet_attended)
        densenet_adjusted = self.gated_adjustment_densenet(densenet_attended)
        
        # Step 3: Cross Attention
        cross_topic_densenet = self.cross_attention_topic_densenet(topic_adjusted, densenet_adjusted)
        cross_sentence_resnet = self.cross_attention_sentence_resnet(text_adjusted, resnet_adjusted)
        
        # Step 4: Co-Attention
        co_topic_densenet, co_densenet_topic = self.co_attention_topic_densenet(topic_adjusted, densenet_adjusted)
        co_sentence_resnet, co_resnet_sentence = self.co_attention_sentence_resnet(text_adjusted, resnet_adjusted)
        
        # Step 5: Concatenate Cross Attention and Co-Attention outputs (including second outputs)
        fusion_output = torch.cat((
            cross_topic_densenet,
            cross_sentence_resnet,
            co_topic_densenet,
            co_sentence_resnet,
            co_densenet_topic,
            co_resnet_sentence
        ), dim=1)
        
        # Step 6: Classification
        output = self.classifier(fusion_output)
        
        return output
