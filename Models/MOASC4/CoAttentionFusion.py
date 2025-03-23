import torch
import torch.nn as nn
import torch.nn.functional as F

class CoAttentionFusion(nn.Module):
    def __init__(self, d_model=768, num_heads=4):
        super().__init__()
        # Projection layers for image+text to same dimension
        self.img_proj = nn.Linear(d_model, d_model)
        self.txt_proj = nn.Linear(d_model, d_model)
        
        # Cross-attention: each modality can attend to the other
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, batch_first=True)
        # If needed, feed-forward or other layers can go here.

    def forward(self, text_emb, img_emb):
        """
        text_emb: [batch_size, seq_len, d_model]
        img_emb:  [batch_size, d_model] or [batch_size, num_img_regions, d_model]
        returns: fused features with cross-attention
        """
        # Project to same dimension
        txt = self.txt_proj(text_emb)  # [B, seq_len, d_model]
        if len(img_emb.shape) == 2:
            # If it's just a single [batch_size, d_model], make it [batch_size, 1, d_model]
            img_emb = img_emb.unsqueeze(1)
        img = self.img_proj(img_emb)   # [B, num_regions, d_model]

        # Let text attend to image features
        # MultiheadAttention with batch_first=True expects shape: [B, seq_len, d_model]
        # query=text, key=image, value=image => text attends to the image
        text_attn, _ = self.cross_attn(query=txt, key=img, value=img)
        
        # Optionally, let image attend to text similarly, or fuse text_attn + original txt.
        # For simplicity, we just return the text-attended output:
        return text_attn
