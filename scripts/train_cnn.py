import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.builder import get_model
from data.datasets import TIFDataset
from training.train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = TIFDataset(...)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32)

model = get_model("cnn", num_classes=2, input_channels=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train(model, train_loader, val_loader, optimizer, criterion, device, epochs=50)
