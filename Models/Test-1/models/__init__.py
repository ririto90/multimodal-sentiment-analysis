from .feature_extractors import (
    TextFeatureExtractor,
    TopicFeatureExtractor,
    ImageLowLevelFeatureExtractor,
    ImageHighLevelFeatureExtractor,
)
from .attention_modules import MultiHeadSelfAttention, MultiHeadCrossAttention
from .co_attention import CoAttentionModule
from .fusion_module import FusionModule

class MultimodalSentimentModel(nn.Module):
    """
    Main model that integrates all components.
    """
    def __init__(self, d_model=768):
        super(MultimodalSentimentModel, self).__init__()
        # Feature extractors
        self.text_extractor = TextFeatureExtractor()
        self.topic_extractor = TopicFeatureExtractor()
        self.image_low_extractor = ImageLowLevelFeatureExtractor(output_dim=d_model)
        self.image_high_extractor = ImageHighLevelFeatureExtractor(output_dim=d_model)

        # Self-attention modules
        self.text_self_attention = MultiHeadSelfAttention(d_model=d_model)
        self.image_self_attention = MultiHeadSelfAttention(d_model=d_model)

        # Co-attention module
        self.co_attention = CoAttentionModule(d_model=d_model)

        # Fusion module
        self.fusion_module = FusionModule(d_model=d_model)

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)  # Define num_classes

    def forward(self, text_inputs, topic_inputs, image_inputs):
        # Extract features
        S_i = self.text_extractor(**text_inputs)
        T_i = self.topic_extractor(**topic_inputs)
        I_i_low = self.image_low_extractor(image_inputs)
        I_i_high = self.image_high_extractor(image_inputs)

        # Apply self-attention
        S_i_attended = self.text_self_attention(S_i)
        I_i_attended = self.image_self_attention(I_i_low)

        # Compute co-attention
        H_SI, H_IS = self.co_attention(S_i_attended, I_i_attended)

        # Fuse features hierarchically
        fused_features = self.fusion_module(H_SI, H_IS)

        # Classification
        logits = self.classifier(fused_features.mean(dim=1))  # Pooling over sequence length
        return logits
