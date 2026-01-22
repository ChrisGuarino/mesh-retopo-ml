"""
Training Script for Multi-View CNN Importance Model

Trains the model to predict per-face importance using:
1. CNN features from multi-view renderings
2. Geometric features (curvature, etc.)

Labels come from saliency detection or manual annotation.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import argparse
import json
from tqdm import tqdm

from multiview_renderer import MultiViewRenderer
from multiview_cnn import MultiViewImportanceModel, save_model, load_model
from feature_extractor import FeatureExtractor


class MultiViewDataset(Dataset):
    """Dataset for multi-view importance training."""

    def __init__(self, data_path):
        """
        Load preprocessed dataset.

        Args:
            data_path: Path to .npz file with preprocessed features
        """
        data = np.load(data_path, allow_pickle=True)

        self.cnn_features = data['cnn_features']
        self.geo_features = data['geo_features']
        self.importance = data['importance']

        print(f"Loaded dataset with {len(self.importance)} samples")
        print(f"  CNN features shape: {self.cnn_features.shape}")
        print(f"  Geo features shape: {self.geo_features.shape}")
        print(f"  Importance range: [{self.importance.min():.3f}, {self.importance.max():.3f}]")

    def __len__(self):
        return len(self.importance)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.cnn_features[idx]).float(),
            torch.from_numpy(self.geo_features[idx]).float(),
            torch.tensor(self.importance[idx]).float()
        )


def preprocess_mesh(mesh, renderer, cnn_model, device='cpu'):
    """
    Preprocess a single mesh to extract CNN and geometric features.

    Args:
        mesh: trimesh.Trimesh object
        renderer: MultiViewRenderer instance
        cnn_model: CNNFeatureExtractor or MultiViewImportanceModel
        device: torch device

    Returns:
        dict with cnn_features and geo_features
    """
    import trimesh

    # Ensure it's a proper mesh
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # Render views
    views = renderer.render_views(mesh)

    # Extract CNN features
    if hasattr(cnn_model, 'extract_cnn_features'):
        cnn_features = cnn_model.extract_cnn_features(views)
    else:
        cnn_features = cnn_model.cnn.extract_features(views['color'])
        from multiview_cnn import FeatureProjector
        projector = FeatureProjector()
        cnn_features = projector.project_features_fast(
            cnn_features,
            np.stack(views['face_ids']),
            len(mesh.faces)
        )

    # Extract geometric features
    extractor = FeatureExtractor(mesh)
    geo_features = extractor.extract_all_features()
    geo_features = extractor.normalize_features(geo_features)

    return {
        'cnn_features': cnn_features,
        'geo_features': geo_features,
    }


def compute_saliency_labels(mesh, views):
    """
    Compute importance labels using image saliency as a proxy.

    This uses edge detection and gradient magnitude as a simple saliency measure.
    Faces visible in high-saliency regions get higher importance.

    Args:
        mesh: trimesh.Trimesh object
        views: dict from MultiViewRenderer.render_views()

    Returns:
        Importance scores (n_faces,)
    """
    from scipy import ndimage
    import cv2

    n_faces = len(mesh.faces)
    face_saliency = np.zeros(n_faces, dtype=np.float32)
    face_counts = np.zeros(n_faces, dtype=np.float32)

    for view_idx, (color, face_ids) in enumerate(zip(views['color'], views['face_ids'])):
        # Convert to grayscale
        if color.shape[-1] == 3:
            gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        else:
            gray = color[:, :, 0]

        # Compute saliency using multiple cues

        # 1. Edge detection (Canny)
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0

        # 2. Gradient magnitude
        grad_x = ndimage.sobel(gray.astype(np.float32), axis=1)
        grad_y = ndimage.sobel(gray.astype(np.float32), axis=0)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_mag = grad_mag / (grad_mag.max() + 1e-8)

        # 3. Laplacian (second derivative - detects fine detail)
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
        laplacian = laplacian / (laplacian.max() + 1e-8)

        # Combine saliency measures
        saliency = edges * 0.3 + grad_mag * 0.4 + laplacian * 0.3

        # Smooth slightly to reduce noise
        saliency = ndimage.gaussian_filter(saliency, sigma=1.0)

        # Accumulate saliency per face
        valid_mask = (face_ids >= 0) & (face_ids < n_faces)
        for face_idx in np.unique(face_ids[valid_mask]):
            mask = face_ids == face_idx
            face_saliency[face_idx] += saliency[mask].mean()
            face_counts[face_idx] += 1

    # Average across views
    nonzero = face_counts > 0
    face_saliency[nonzero] /= face_counts[nonzero]

    # Normalize to 0-1
    if face_saliency.max() > face_saliency.min():
        face_saliency = (face_saliency - face_saliency.min()) / (face_saliency.max() - face_saliency.min())

    # Apply sigmoid to create more contrast
    face_saliency = 1 / (1 + np.exp(-8 * (face_saliency - 0.5)))

    return face_saliency


def preprocess_dataset(mesh_paths, output_path, n_views=4, image_size=128, device='cpu',
                       batch_size=10, max_faces_per_mesh=50000):
    """
    Preprocess a list of meshes and save features + labels.

    Memory-efficient version that processes in batches and saves incrementally.

    Args:
        mesh_paths: List of paths to mesh files
        output_path: Path to save the preprocessed dataset
        n_views: Number of views to render (default reduced to 4)
        image_size: Size of rendered images (default reduced to 128)
        device: torch device
        batch_size: Number of meshes to process before saving
        max_faces_per_mesh: Skip meshes with more faces than this
    """
    import trimesh
    import gc

    # Initialize renderer and model with smaller settings
    renderer = MultiViewRenderer(image_size=image_size, n_views=n_views)

    # Use a lighter CNN backbone
    from multiview_cnn import CNNFeatureExtractor, FeatureProjector
    cnn = CNNFeatureExtractor(backbone='resnet18', feature_dim=128)  # Reduced from 256
    cnn.to(device)
    cnn.eval()

    projector = FeatureProjector()

    all_cnn_features = []
    all_geo_features = []
    all_importance = []
    all_mesh_ids = []

    processed_count = 0

    print(f"Preprocessing {len(mesh_paths)} meshes...")
    print(f"  Image size: {image_size}, Views: {n_views}")
    print(f"  Batch size: {batch_size}, Max faces: {max_faces_per_mesh}")

    for mesh_idx, mesh_path in enumerate(tqdm(mesh_paths)):
        try:
            # Load mesh
            mesh = trimesh.load(mesh_path, force='mesh')
            if isinstance(mesh, trimesh.Scene):
                mesh = mesh.dump(concatenate=True)

            # Skip very small or very large meshes
            if len(mesh.faces) < 100 or len(mesh.faces) > max_faces_per_mesh:
                continue

            # Render views
            views = renderer.render_views(mesh)

            # Extract CNN features
            with torch.no_grad():
                feature_maps = cnn.extract_features(views['color'])

            # Project to faces
            cnn_features = projector.project_features_fast(
                feature_maps,
                np.stack(views['face_ids']),
                len(mesh.faces)
            )

            # Extract geometric features
            extractor = FeatureExtractor(mesh)
            geo_features = extractor.extract_all_features()
            geo_features = extractor.normalize_features(geo_features)

            # Compute saliency-based labels
            try:
                importance = compute_saliency_labels(mesh, views)
            except Exception as e:
                # Fallback to geometric-based labels
                from dataset_generator import DatasetGenerator
                gen = DatasetGenerator()
                importance = gen.compute_face_importance(mesh)

            # Accumulate
            all_cnn_features.append(cnn_features.astype(np.float16))  # Use float16 to save memory
            all_geo_features.append(geo_features.astype(np.float16))
            all_importance.append(importance.astype(np.float16))
            all_mesh_ids.append(np.full(len(mesh.faces), mesh_idx, dtype=np.int32))

            processed_count += 1

            # Clear memory periodically
            del views, feature_maps, cnn_features, geo_features, importance, mesh
            if mesh_idx % 5 == 0:
                gc.collect()

        except Exception as e:
            print(f"  Failed to process {mesh_path}: {e}")
            continue

    if len(all_cnn_features) == 0:
        print("No meshes were successfully processed!")
        return

    # Stack all data
    print("\nStacking data...")
    cnn_features = np.vstack(all_cnn_features).astype(np.float32)
    geo_features = np.vstack(all_geo_features).astype(np.float32)
    importance = np.concatenate(all_importance).astype(np.float32)
    mesh_ids = np.concatenate(all_mesh_ids)

    # Clear lists to free memory
    del all_cnn_features, all_geo_features, all_importance, all_mesh_ids
    gc.collect()

    # Save
    print("Saving dataset...")
    np.savez_compressed(
        output_path,
        cnn_features=cnn_features,
        geo_features=geo_features,
        importance=importance,
        mesh_ids=mesh_ids,
        n_meshes=processed_count
    )

    print(f"\nDataset saved to {output_path}")
    print(f"  Total faces: {len(importance):,}")
    print(f"  CNN feature dim: {cnn_features.shape[1]}")
    print(f"  Geo feature dim: {geo_features.shape[1]}")


def train_epoch(model, loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for cnn_feat, geo_feat, importance in loader:
        cnn_feat = cnn_feat.to(device)
        geo_feat = geo_feat.to(device)
        importance = importance.to(device)

        optimizer.zero_grad()
        predictions = model(cnn_feat, geo_feat)
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
        for cnn_feat, geo_feat, importance in loader:
            cnn_feat = cnn_feat.to(device)
            geo_feat = geo_feat.to(device)
            importance = importance.to(device)

            predictions = model(cnn_feat, geo_feat)
            loss = criterion(predictions, importance)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def train(data_path, output_path, epochs=50, batch_size=256, lr=0.001,
          val_split=0.2, patience=10):
    """
    Train the multi-view importance model.

    Args:
        data_path: Path to preprocessed data (.npz file)
        output_path: Path to save model checkpoint
        epochs: Number of training epochs
        batch_size: Training batch size
        lr: Learning rate
        val_split: Validation split ratio
        patience: Early stopping patience
    """
    # Device setup
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading data from {data_path}...")
    dataset = MultiViewDataset(data_path)

    # Get dimensions from data
    cnn_dim = dataset.cnn_features.shape[1]
    geo_dim = dataset.geo_features.shape[1]

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"\n  Training samples: {train_size:,}")
    print(f"  Validation samples: {val_size:,}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Create model
    model = MultiViewImportanceModel(cnn_feature_dim=cnn_dim, geo_feature_dim=geo_dim)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")

    # Loss and optimizer - only train the importance head (CNN is frozen initially)
    criterion = nn.BCELoss()

    # Freeze CNN backbone initially
    for param in model.cnn.parameters():
        param.requires_grad = False

    # Only optimize importance head
    optimizer = torch.optim.Adam(model.importance_head.parameters(), lr=lr)
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
    parser = argparse.ArgumentParser(description="Train multi-view importance model")

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess meshes')
    preprocess_parser.add_argument("--input", type=str, required=True,
                                   help="Directory containing mesh files")
    preprocess_parser.add_argument("--output", type=str, default="data/multiview_data.npz",
                                   help="Output file path")
    preprocess_parser.add_argument("--n_views", type=int, default=4,
                                   help="Number of views to render (default: 4 for memory efficiency)")
    preprocess_parser.add_argument("--image_size", type=int, default=128,
                                   help="Rendered image size (default: 128 for memory efficiency)")
    preprocess_parser.add_argument("--max_meshes", type=int, default=100,
                                   help="Maximum number of meshes to process")
    preprocess_parser.add_argument("--max_faces", type=int, default=50000,
                                   help="Skip meshes with more faces than this")

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument("--data", type=str, default="data/multiview_data.npz",
                              help="Path to preprocessed data")
    train_parser.add_argument("--output", type=str, default="models/multiview_importance.pt",
                              help="Output model path")
    train_parser.add_argument("--epochs", type=int, default=50,
                              help="Number of training epochs")
    train_parser.add_argument("--batch_size", type=int, default=256,
                              help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.001,
                              help="Learning rate")
    train_parser.add_argument("--patience", type=int, default=10,
                              help="Early stopping patience")

    args = parser.parse_args()

    if args.command == 'preprocess':
        import trimesh
        from pathlib import Path

        # Find all mesh files
        input_dir = Path(args.input)
        extensions = ['.stl', '.obj', '.ply', '.off', '.STL', '.OBJ', '.PLY']
        mesh_paths = []
        for ext in extensions:
            mesh_paths.extend(input_dir.rglob(f'*{ext}'))

        print(f"Found {len(mesh_paths)} mesh files in {input_dir}")

        if len(mesh_paths) == 0:
            print("No mesh files found!")
            return

        # Limit number of meshes
        if len(mesh_paths) > args.max_meshes:
            import random
            random.shuffle(mesh_paths)
            mesh_paths = mesh_paths[:args.max_meshes]
            print(f"Using {len(mesh_paths)} meshes for training")

        preprocess_dataset(
            mesh_paths,
            args.output,
            n_views=args.n_views,
            image_size=args.image_size,
            max_faces_per_mesh=args.max_faces
        )

    elif args.command == 'train':
        print("=" * 60)
        print("Multi-View CNN - Training")
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
            patience=args.patience,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
