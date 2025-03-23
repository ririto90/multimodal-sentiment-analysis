# SentimentClassifier.py

from helpers import *

class TextSentimentClassifier(nn.Module):
    def __init__(self, freeze_bert = True, num_classes = 3):
        super(TextSentimentClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.save_path = '/home/rgg2706/Multimodal-Sentiment-Analysis/Models/MultimodalOpinionAnalysis2/SaveFiles/TextSentimentClassifier'

        # Freeze layers
        if freeze_bert:
            for params in self.bert_model.parameters():
                params.requires_grad = False

        # Classification layer
        self.linear = nn.Linear(768, num_classes)

    def forward(self, sequences, attn_masks):
        # Feeding the input to BERT model in order to obtain the contextualized representations
        cont_reps = self.bert_model(sequences, attn_masks)
        cont_reps = cont_reps[0]

        cls_rep = cont_reps[:, 0]

        out = self.linear(cls_rep)
        return out


class AttentionLayer(nn.Module):
    def __init__(self, d):
        super().__init__()

        W_w = nn.Parameter(torch.Tensor(d, d))
        nn.init.xavier_uniform_(W_w, gain=nn.init.calculate_gain('tanh'))
        W_os = nn.Parameter(torch.Tensor(d, d))
        nn.init.xavier_uniform_(W_os, gain=nn.init.calculate_gain('tanh'))
        bias = nn.Parameter(torch.Tensor(d))
        nn.init.zeros_(bias)
        u = nn.Parameter(torch.Tensor(d))
        nn.init.zeros_(u)

        self.W_w = W_w
        self.W_os = W_os
        self.bias = bias
        self.u = u

        self.tanh = torch.nn.Tanhshrink()
        # self.tanh = torch.nn.Tanh()

    def forward(self, visual_reps, contextual_text_reps):
        num_tokens = list(contextual_text_reps.size())[1]
        scene_object_reps = torch.repeat_interleave(visual_reps, repeats=num_tokens, dim=1)
        scene_object_reps = scene_object_reps.reshape(list(contextual_text_reps.size()))

        weighted_cont_reps = torch.matmul(contextual_text_reps, self.W_w)
        weighted_visual_reps = torch.matmul(scene_object_reps, self.W_os)

        u_t = self.tanh(weighted_cont_reps + weighted_visual_reps + self.bias)
        
        exp = torch.exp(torch.matmul(u_t, self.u))
        alpha = exp / (torch.sum(exp) + 1e-7) # Shape: [batch_size, num_tokens]


        weighted_text_reps = torch.matmul(alpha, contextual_text_reps) 
        v_alpha = torch.sum(weighted_text_reps, dim=1) # Shape: [batch_size, d]

        return v_alpha
    
class CoAttentionFusion(nn.Module):
    def __init__(self, img_dim=768, txt_dim=768, num_heads=4):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, txt_dim)
        self.txt_proj = nn.Linear(txt_dim, txt_dim)
        # Multi-head attention layer
        self.cross_attn = nn.MultiheadAttention(embed_dim=txt_dim, 
                                                num_heads=num_heads,
                                                batch_first=True)

    def forward(self, text_emb, img_emb):
        """
        text_emb: shape [B, seq_len, txt_dim]
        img_emb:  shape [B, img_dim] or [B, num_img_regions, img_dim]
        returns:  co-attended text features, shape [B, seq_len, txt_dim]
        """
        # Project both modalities to same dimension
        txt = self.txt_proj(text_emb)  # [B, seq_len, txt_dim]

        # If image is just a single vector per sample, make it [B, 1, img_dim]
        if img_emb.dim() == 2:
            img_emb = img_emb.unsqueeze(1)
        img = self.img_proj(img_emb)   # => [B, num_regions, txt_dim]

        # text queries the image => text attends to image
        text_attn, _ = self.cross_attn(query=txt, key=img, value=img)
        return text_attn


class SentimentClassifier(nn.Module):
    def __init__(self, freeze_bert=False, num_classes=3):
        super(SentimentClassifier, self).__init__()
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.resnet_model = models.resnet50(pretrained=True)
        # self.vgg_model.eval()

        self.__scene_alexnet_file = "/home/rgg2706/Multimodal-Sentiment-Analysis/Datasets/alexnet_places365.pth.tar"
        model = models.__dict__['alexnet'](num_classes=365)
        checkpoint = torch.load(self.__scene_alexnet_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
        # model.eval()
        self.alexnet_model = model

        # Freeze layers
        if freeze_bert:
            for params in self.bert_model.parameters():
                params.requires_grad = False

        # linear layers for combining object+scene
        self.linear1 = nn.Linear(in_features=1365, out_features=768)
        self.relu1 = nn.ReLU()
        
        self.coattention = CoAttentionFusion(img_dim=768, txt_dim=768, num_heads=4)
        # self.attention_layer = AttentionLayer(768)
        
        self.linear2 = nn.Linear(in_features=1536, out_features=768)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(in_features=768, out_features=384)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(in_features=384, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.dropout = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.25)

        self.count_epochs = 1


    def forward(self, sequences, attn_masks, images, unfreeze_bert=True):
        # if self.count_epochs >= 10:
        #     self.vgg_model.eval()
        #     self.alexnet_model.eval()

        if unfreeze_bert:
            for params in self.bert_model.parameters():
                params.requires_grad = True

        # Feeding the input to BERT model in order to obtain the contextualized representations
        cont_reps = self.bert_model(sequences, attn_masks) # [B, seq_len, 768]
        text_emb = cont_reps[0]
        cls_rep = text_emb[:, 0]  # [B, 768]

        # Get the image features 
        object_reps = self.resnet_model(images)

        # Scene feature representation
        with torch.no_grad():
            scene_reps = self.alexnet_model.forward(images) # [batch_size, 365]

        # Visual feature vector: Tensor of shape [batch_size, 1365].
        scene_object_reps = torch.cat((object_reps, scene_reps), dim=1)
        os = list(scene_object_reps.size())[1] # size of each visual feature vector

        visual_reps = self.relu1(self.linear1(scene_object_reps))

        # attn_mask = self.attention_layer(visual_reps, cont_reps)
        # attn = cls_rep * attn_mask
        
        co_text_attn = self.coattention(text_emb, visual_reps.unsqueeze(1)) # [B, seq_len, 768]
        co_text_vec = co_text_attn.mean(dim=1) # [B, 768]
        attn = cls_rep * co_text_vec
        

        multimodal_reps = torch.cat((visual_reps, attn), dim=1)
        V_mul = self.relu2(self.dropout(self.linear2(multimodal_reps)))
        V_int = self.relu3(self.dropout2(self.linear3(V_mul)))
        pred = self.softmax(self.linear4(V_int))

        self.count_epochs += 1

        return pred