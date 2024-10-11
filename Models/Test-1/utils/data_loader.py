import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
from PIL import Image

class MultimodalDataset(Dataset):
    """
    Custom Dataset for loading multimodal data.
    """
    def __init__(self, data_frame, transform=None):
        self.data_frame = data_frame
        self.transform = transform
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        text = self.data_frame.iloc[idx]['text']
        topic = self.data_frame.iloc[idx]['topic']
        image_path = self.data_frame.iloc[idx]['image_path']
        label = self.data_frame.iloc[idx]['label']

        # Tokenize text and topic
        text_inputs = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True)
        topic_inputs = self.tokenizer(topic, return_tensors='pt', padding='max_length', truncation=True)

        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        sample = {
            'text_inputs': {key: val.squeeze(0) for key, val in text_inputs.items()},
            'topic_inputs': {key: val.squeeze(0) for key, val in topic_inputs.items()},
            'image': image,
            'label': torch.tensor(label, dtype=torch.long)
        }
        return sample
