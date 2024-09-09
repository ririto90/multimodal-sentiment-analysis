# -*- coding: utf-8 -*-
# file: mmtan.py
# author: jyu5 <yujianfei1990@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

from layers.dynamic_rnn import DynamicLSTM
from layers.attention import Attention
from layers.mm_attention import MMAttention
from layers.attention2 import Attention2
import torch
import torch.nn as nn
import torch.nn.functional as F
            
class MMFUSION(nn.Module):
    def __init__(self, embedding_matrix, opt):
        super(MMFUSION, self).__init__()
        self.opt = opt
        
        # Embedding layer for text
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        
        # Single LSTM layer for processing the entire text sequence
        self.lstm = DynamicLSTM(opt.embed_dim, opt.hidden_dim, num_layers=1, batch_first=True)
        
        # Single attention layer for the text context
        # Query (q): This is what you want to pay attention with.
        # Key (k): This is what you want to pay attention to.
        # Value (v): This is what you actually extract from the attention process.
        self.attention = Attention(opt.hidden_dim, score_function='bi_linear', dropout=opt.dropout_rate)
        
        # Linear layers to convert visual embeddings to text-compatible features
        self.vis2text = nn.Linear(2048, opt.hidden_dim)
        
        # Final dense layer to produce the output for sentiment classification
        self.dense_3 = nn.Linear(opt.hidden_dim * 2, opt.polarities_dim)  # Adjusted for text and visual concatenation
    
    def forward(self, inputs, visual_embeds_global):

        x = inputs[0]  # Assuming the entire text sequence is provided as a single input
        ori_x_len = torch.sum(x != 0, dim=-1).cpu()

        # Embed the text input
        x = self.embed(x)
        context, (_, _) = self.lstm(x, ori_x_len)  # Process the entire text with one LSTM layer
        
        # print(f"x shape: {x.shape}")
        # print(f"context shape: {context.shape}")

        # Apply attention
        context_final = self.attention(context, context, ori_x_len).squeeze(dim=1)

        # Visual embeddings
        converted_vis_embed = self.vis2text(torch.tanh(visual_embeds_global))
        
        # print(f"context_final shape: {context_final.shape}")
        # print(f"converted_vis_embed shape: {converted_vis_embed.shape}")
        
        # Concatenate text and visual features
        text_representation = torch.cat((context_final, converted_vis_embed), dim=-1)
        
        out = self.dense_3(text_representation)
        return out
        