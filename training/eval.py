import torch

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()

    return total_loss
  
