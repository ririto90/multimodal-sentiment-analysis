import torch
import torch.nn as nn
from transformers import RobertaModel
from torchvision import models

class TextFeatureExtractor(nn.Module):
    """
    Extracts sentence embeddings using RoBERTa.
    S_i = RoBERTa(W_i), S_i ∈ R^(L_i × d)
    """
    def __init__(self, pretrained_model='roberta-base'):
        super(TextFeatureExtractor, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = outputs.last_hidden_state  # S_i ∈ R^(L_i × d)
        return sentence_embeddings

class TopicFeatureExtractor(nn.Module):
    """
    Extracts topic embeddings using RoBERTa.
    T_i = RoBERTa(P_i), T_i ∈ R^(L_p × d)
    """
    def __init__(self, pretrained_model='roberta-base'):
        super(TopicFeatureExtractor, self).__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        topic_embeddings = outputs.last_hidden_state  # T_i ∈ R^(L_p × d)
        return topic_embeddings

class ImageLowLevelFeatureExtractor(nn.Module):
    """
    Extracts low-level image features using a CNN.
    I_i^l = CNN(I_i) ∈ R^(H × W × c) → I_i^l ∈ R^(L_i × d)
    """
    def __init__(self, output_dim):
        super(ImageLowLevelFeatureExtractor, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])  # Remove fully connected layers
        self.output_dim = output_dim
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, images):
        features = self.cnn(images)  # Output shape: (batch_size, c, H, W)
        features = self.adaptive_pool(features)
        batch_size, c, h, w = features.size()
        features = features.view(batch_size, c, h * w).permute(0, 2, 1)  # Reshape to (batch_size, L_i, d)
        return features  # I_i^l ∈ R^(L_i × d)

class ImageHighLevelFeatureExtractor(nn.Module):
    """
    Extracts high-level image features using DenseNet.
    I_i^h = DenseNet(I_i) ∈ R^d
    """
    def __init__(self, output_dim):
        super(ImageHighLevelFeatureExtractor, self).__init__()
        self.densenet = models.densenet121(pretrained=True)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, output_dim)

    def forward(self, images):
        features = self.densenet(images)  # I_i^h ∈ R^d
        return features
