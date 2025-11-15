import torch
from torch.utils.data import DataLoader
from typing import Callable
from .early_stopping import EarlyStopping


def train(model,
          train_loader: DataLoader,
          val_loader: DataLoader,
          optimizer,
          criterion,
          device,
          epochs: int,
          save_path: str = None):

    model.to(device)
    early_stopping = EarlyStopping(patience=10)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        val_loss = evaluate(model, val_loader, criterion, device)

        if save_path is not None:
            torch.save(model.state_dict(), save_path)

        if early_stopping.step(val_loss):
            print("Early stopping triggered.")
            break

        print(f"Epoch {epoch:03d} | Train Loss {total_loss:.4f} | Val Loss {val_loss:.4f}")
