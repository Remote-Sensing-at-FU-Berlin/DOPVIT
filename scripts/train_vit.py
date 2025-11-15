import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.builder import get_model
from data.datasets import TIFDataset
from training.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TIFDataset(...)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset, batch_size=16)

model = get_model("vit", num_classes=2, pretrained=True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-5)

train(model, train_loader, val_loader, optimizer, criterion, device, epochs=30)
