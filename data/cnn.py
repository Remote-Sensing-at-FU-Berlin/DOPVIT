import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeciesClassifier(nn.Module):
  def __init__(self, input_channels=4):
    super().__init__()
    self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.adaptive_pool = nn.AdaptiveAvgPool2d((6,6))
    self.fc1 = nn.Linear(64 * 6 * 6, 128)
    self.fc2 = nn.Linear(128, 1)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.adaptive_pool(x)
    x = torch.flatten(x, 1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x
