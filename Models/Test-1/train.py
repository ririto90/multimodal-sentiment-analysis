# Hierarchical Hybrid Multi-Attention Fusion Model (HHMAFM) 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models import MultimodalSentimentModel
from utils.data_loader import MultimodalDataset

# Define hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 1e-4
num_classes = 2  # Update as needed

# Prepare data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Assuming you have a DataFrame 'df' with your data
dataset = MultimodalDataset(data_frame=df, transform=transform)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, optimizer
model = MultimodalSentimentModel(d_model=768)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        text_inputs = batch['text_inputs']
        topic_inputs = batch['topic_inputs']
        images = batch['image']
        labels = batch['label']

        # Forward pass
        logits = model(text_inputs, topic_inputs, images)
        loss = criterion(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
