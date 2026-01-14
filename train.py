"""
Training Script for Mesh Retopology ML

Trains the ImportanceNet model on generated dataset.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import argparse

from importance_model import ImportanceNet, ImportanceDataset, save_model


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for features, importance in loader:
        features = features.to(device)
        importance = importance.to(device)

        optimizer.zero_grad()
        predictions = model(features).squeeze(-1)
        loss = criterion(predictions, importance)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for features, importance in loader:
            features = features.to(device)
            importance = importance.to(device)

            predictions = model(features).squeeze(-1)
            loss = criterion(predictions, importance)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def train(data_path, output_path, epochs=50, batch_size=256, lr=0.001,
          val_split=0.2, patience=10):
    """
    Train the importance prediction model.

    Args:
        data_path: Path to training data (.npz file)
        output_path: Path to save model checkpoint
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        val_split: Validation split ratio
        patience: Early stopping patience
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print(f"\nLoading data from {data_path}...")
    data = np.load(data_path)
    features = data['features']
    importance = data['importance']

    print(f"  Total samples: {len(features):,}")
    print(f"  Feature dimensions: {features.shape[1]}")
    print(f"  Importance range: [{importance.min():.3f}, {importance.max():.3f}]")

    # Create dataset
    dataset = ImportanceDataset(features, importance)

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"\n  Training samples: {train_size:,}")
    print(f"  Validation samples: {val_size:,}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # Create model
    input_dim = features.shape[1]
    model = ImportanceNet(input_dim=input_dim).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nStarting training for {epochs} epochs...")
    print("-" * 60)

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        # Logging
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {current_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            save_model(model, output_path, epoch=epoch, loss=val_loss, optimizer=optimizer)
            print(f"  -> Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break

    print("-" * 60)
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train importance prediction model")
    parser.add_argument("--data", type=str, default="data/training_data.npz",
                        help="Path to training data")
    parser.add_argument("--output", type=str, default="models/importance_v1.pt",
                        help="Output model path")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Validation split ratio")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience")

    args = parser.parse_args()

    print("=" * 60)
    print("Mesh Retopology ML - Training")
    print("=" * 60)
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 60)

    train(
        data_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
