import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention_weights = self.softmax(q @ k.transpose(-2, -1) / (k.size(-1) ** 0.5))
        attended_output = attention_weights @ v
        return attended_output

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

class CoAttention(nn.Module):
    def __init__(self, input_dim1, input_dim2, hidden_dim):
        super(CoAttention, self).__init__()
        self.linear1 = nn.Linear(input_dim1, hidden_dim)
        self.linear2 = nn.Linear(input_dim2, hidden_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x1, x2):
        h1 = self.linear1(x1)
        h2 = self.linear2(x2)
        attention_weights1 = self.softmax(h1 @ h2.T)
        attention_weights2 = self.softmax(h2 @ h1.T)
        attended_x1 = attention_weights1 @ x2
        attended_x2 = attention_weights2 @ x1
        return attended_x1, attended_x2

class MFCCHFUSION(nn.Module):
    def __init__(self, opt):
        super(MFCCHFUSION, self).__init__()
        self.opt = opt
        
        text_dim = 768
        visual_dim = 1000
        
        hidden_dim = opt.common_dim  # e.g., 512
        num_classes = opt.num_classes  # Number of output classes

        # Projection layers to bring all features to the same dimension
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.topic_proj = nn.Linear(text_dim, hidden_dim)
        self.resnet_proj = nn.Linear(visual_dim, hidden_dim)
        self.densenet_proj = nn.Linear(visual_dim, hidden_dim)
        
        # Self Attention for each individual feature representation
        self.self_attention_text = SelfAttention(hidden_dim, hidden_dim)
        self.self_attention_topic = SelfAttention(hidden_dim, hidden_dim)
        self.self_attention_resnet = SelfAttention(hidden_dim, hidden_dim)
        self.self_attention_densenet = SelfAttention(hidden_dim, hidden_dim)
        
        # Cross Attention between modalities
        self.cross_attention_topic_densenet = CrossAttention(hidden_dim, hidden_dim, hidden_dim)
        self.cross_attention_sentence_resnet = CrossAttention(hidden_dim, hidden_dim, hidden_dim)
        
        # Co-Attention between modalities
        self.co_attention_topic_densenet = CoAttention(hidden_dim, hidden_dim, hidden_dim)
        self.co_attention_sentence_resnet = CoAttention(hidden_dim, hidden_dim, hidden_dim)
        
        # Final classifier
        self.classifier = nn.Linear(6 * hidden_dim, num_classes)
    
    def forward(self, roberta_text_features, roberta_topic_features, resnet_features, densenet_features):
        
        # Step 1: Project features to common dimension
        text_proj = self.text_proj(roberta_text_features)
        topic_proj = self.topic_proj(roberta_topic_features)
        resnet_proj = self.resnet_proj(resnet_features)
        densenet_proj = self.densenet_proj(densenet_features)
      
        # Step 2: Self Attention
        text_attended = self.self_attention_text(text_proj)
        topic_attended = self.self_attention_topic(topic_proj)
        resnet_attended = self.self_attention_resnet(resnet_proj)
        densenet_attended = self.self_attention_densenet(densenet_proj)
        
        # Step 2: Cross Attention
        cross_topic_densenet = self.cross_attention_topic_densenet(topic_attended, densenet_attended)
        cross_sentence_resnet = self.cross_attention_sentence_resnet(text_attended, resnet_attended)
        
        # Step 3: Co-Attention
        co_topic_densenet, co_densenet_topic = self.co_attention_topic_densenet(topic_attended, densenet_attended)
        co_sentence_resnet, co_resnet_sentence = self.co_attention_sentence_resnet(text_attended, resnet_attended)
        
        # Step 4: Concatenate Cross Attention and Co-Attention outputs
        fusion_output = torch.cat(
            (
                cross_topic_densenet,
                cross_sentence_resnet,
                co_topic_densenet,
                co_sentence_resnet,
                co_densenet_topic,
                co_resnet_sentence
            ),
            dim=1
)
        
        # Step 5: Classification
        output = self.classifier(fusion_output)
        
        return output
