"""
Importance Prediction Model for Mesh Retopology ML

Simple MLP that predicts per-face importance scores based on geometric features.
"""

import torch
import torch.nn as nn
import numpy as np


class ImportanceNet(nn.Module):
    """
    MLP for predicting face importance scores.

    Architecture: input_dim -> 64 -> 32 -> 16 -> 1
    Uses ReLU activations and sigmoid output.
    """

    def __init__(self, input_dim=11, hidden_dims=[64, 32, 16], dropout=0.1):
        """
        Initialize the network.

        Args:
            input_dim: Number of input features per face
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input features, shape (batch_size, input_dim)

        Returns:
            Importance scores, shape (batch_size, 1)
        """
        return self.model(x)

    def predict(self, features):
        """
        Convenience method for inference.

        Args:
            features: numpy array of shape (n_faces, n_features)

        Returns:
            numpy array of importance scores, shape (n_faces,)
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(features)
            if torch.cuda.is_available():
                x = x.cuda()
            scores = self(x).squeeze(-1).cpu().numpy()
        return scores


class ImportanceDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for face importance training."""

    def __init__(self, features, importance):
        """
        Initialize dataset.

        Args:
            features: np.ndarray of shape (n_samples, n_features)
            importance: np.ndarray of shape (n_samples,)
        """
        self.features = torch.FloatTensor(features)
        self.importance = torch.FloatTensor(importance)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.importance[idx]


def load_model(path, input_dim=11):
    """
    Load a trained model from disk.

    Args:
        path: Path to saved model checkpoint
        input_dim: Input feature dimension

    Returns:
        Loaded ImportanceNet model
    """
    model = ImportanceNet(input_dim=input_dim)

    checkpoint = torch.load(path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    return model


def save_model(model, path, epoch=None, loss=None, optimizer=None):
    """
    Save model checkpoint.

    Args:
        model: ImportanceNet model
        path: Output path
        epoch: Current epoch (optional)
        loss: Current loss (optional)
        optimizer: Optimizer state (optional)
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }

    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, path)


if __name__ == "__main__":
    # Quick test of the model
    print("Testing ImportanceNet...")

    # Create random test data
    batch_size = 32
    n_features = 11

    model = ImportanceNet(input_dim=n_features)
    print(f"Model architecture:\n{model}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {n_params:,}")

    # Test forward pass
    x = torch.randn(batch_size, n_features)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")

    # Test predict method
    features_np = np.random.randn(100, n_features).astype(np.float32)
    scores = model.predict(features_np)
    print(f"\nPredict input shape: {features_np.shape}")
    print(f"Predict output shape: {scores.shape}")
    print(f"Predict output range: [{scores.min():.4f}, {scores.max():.4f}]")

    print("\nModel test passed!")
