"""
Multi-View CNN for Mesh Importance Prediction

Uses a pretrained CNN backbone to extract features from rendered views,
then projects those features back to mesh faces.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from pathlib import Path


class CNNFeatureExtractor(nn.Module):
    """
    Extract features from images using a pretrained CNN backbone.

    Uses ResNet18 by default - lightweight but effective.
    Returns feature maps that can be projected back to mesh faces.
    """

    def __init__(self, backbone='resnet18', pretrained=True, feature_dim=256):
        """
        Initialize the CNN feature extractor.

        Args:
            backbone: Which pretrained model to use ('resnet18', 'resnet34', 'efficientnet_b0')
            pretrained: Whether to use pretrained weights
            feature_dim: Output feature dimension per pixel
        """
        super().__init__()

        self.feature_dim = feature_dim

        # Load pretrained backbone
        if backbone == 'resnet18':
            base_model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
            # Remove final FC and avgpool - we want spatial features
            self.backbone = nn.Sequential(*list(base_model.children())[:-2])
            backbone_dim = 512
        elif backbone == 'resnet34':
            base_model = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
            self.backbone = nn.Sequential(*list(base_model.children())[:-2])
            backbone_dim = 512
        elif backbone == 'efficientnet_b0':
            base_model = models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
            self.backbone = base_model.features
            backbone_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Project to desired feature dimension
        self.feature_proj = nn.Sequential(
            nn.Conv2d(backbone_dim, feature_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1),
        )

        # ImageNet normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, images):
        """
        Extract feature maps from images.

        Args:
            images: Tensor of shape (B, 3, H, W) with values in [0, 1]

        Returns:
            Feature maps of shape (B, feature_dim, H', W') where H', W' are downsampled
        """
        # Normalize
        x = self.normalize(images)

        # Extract features
        features = self.backbone(x)

        # Project to desired dimension
        features = self.feature_proj(features)

        return features

    def extract_features(self, images_np):
        """
        Extract features from numpy images.

        Args:
            images_np: List of numpy arrays (H, W, 3) with values in [0, 255]

        Returns:
            List of feature maps as numpy arrays
        """
        device = next(self.parameters()).device

        # Convert to tensor
        images = []
        for img in images_np:
            img_tensor = torch.from_numpy(img).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
            images.append(img_tensor)

        images = torch.stack(images).to(device)

        # Extract features
        with torch.no_grad():
            features = self.forward(images)

        return features.cpu().numpy()


class FeatureProjector:
    """
    Project 2D image features back to 3D mesh faces.

    Uses the face ID maps from rendering to map pixel features to faces.
    """

    def __init__(self, aggregation='mean'):
        """
        Initialize the feature projector.

        Args:
            aggregation: How to aggregate features from multiple views ('mean', 'max', 'weighted')
        """
        self.aggregation = aggregation

    def project_features(self, feature_maps, face_id_maps, n_faces):
        """
        Project 2D feature maps to mesh faces.

        Args:
            feature_maps: List of feature maps, each (C, H', W')
            face_id_maps: List of face ID maps, each (H, W) with face indices
            n_faces: Total number of faces in the mesh

        Returns:
            Face features array of shape (n_faces, C)
        """
        n_views = len(feature_maps)
        feature_dim = feature_maps[0].shape[0]

        # Accumulate features per face
        face_features = np.zeros((n_faces, feature_dim), dtype=np.float32)
        face_counts = np.zeros(n_faces, dtype=np.float32)

        for view_idx in range(n_views):
            feat_map = feature_maps[view_idx]  # (C, H', W')
            face_ids = face_id_maps[view_idx]  # (H, W)

            # Upsample feature map to match face_ids resolution
            feat_h, feat_w = feat_map.shape[1], feat_map.shape[2]
            face_h, face_w = face_ids.shape

            # Use nearest neighbor to upsample features
            scale_h = face_h / feat_h
            scale_w = face_w / feat_w

            # For each pixel in face_ids, get corresponding feature
            for i in range(face_h):
                for j in range(face_w):
                    face_idx = face_ids[i, j]
                    if face_idx < 0 or face_idx >= n_faces:
                        continue

                    # Map to feature map coordinates
                    fi = min(int(i / scale_h), feat_h - 1)
                    fj = min(int(j / scale_w), feat_w - 1)

                    # Accumulate feature
                    face_features[face_idx] += feat_map[:, fi, fj]
                    face_counts[face_idx] += 1

        # Average features
        mask = face_counts > 0
        face_features[mask] /= face_counts[mask, np.newaxis]

        # For faces with no observations, use global average
        if not mask.all():
            global_avg = face_features[mask].mean(axis=0) if mask.any() else np.zeros(feature_dim)
            face_features[~mask] = global_avg

        return face_features

    def project_features_fast(self, feature_maps, face_id_maps, n_faces):
        """
        Faster vectorized version of feature projection.

        Args:
            feature_maps: numpy array of shape (n_views, C, H', W')
            face_id_maps: numpy array of shape (n_views, H, W)
            n_faces: Total number of faces

        Returns:
            Face features array of shape (n_faces, C)
        """
        n_views = feature_maps.shape[0]
        feature_dim = feature_maps.shape[1]
        feat_h, feat_w = feature_maps.shape[2], feature_maps.shape[3]
        face_h, face_w = face_id_maps.shape[1], face_id_maps.shape[2]

        # Scale factors
        scale_h = feat_h / face_h
        scale_w = feat_w / face_w

        # Create coordinate maps
        i_coords = (np.arange(face_h) * scale_h).astype(int).clip(0, feat_h - 1)
        j_coords = (np.arange(face_w) * scale_w).astype(int).clip(0, feat_w - 1)

        # Initialize accumulators
        face_features = np.zeros((n_faces, feature_dim), dtype=np.float64)
        face_counts = np.zeros(n_faces, dtype=np.float64)

        for v in range(n_views):
            feat_map = feature_maps[v]  # (C, H', W')
            face_ids = face_id_maps[v]  # (H, W)

            # Sample features at downsampled locations
            sampled_features = feat_map[:, i_coords][:, :, j_coords]  # (C, H, W)
            sampled_features = sampled_features.transpose(1, 2, 0)  # (H, W, C)

            # Accumulate per face
            valid_mask = (face_ids >= 0) & (face_ids < n_faces)

            for face_idx in np.unique(face_ids[valid_mask]):
                mask = face_ids == face_idx
                face_features[face_idx] += sampled_features[mask].sum(axis=0)
                face_counts[face_idx] += mask.sum()

        # Average
        nonzero = face_counts > 0
        face_features[nonzero] /= face_counts[nonzero, np.newaxis]

        # Fill missing with global average
        if not nonzero.all():
            global_avg = face_features[nonzero].mean(axis=0) if nonzero.any() else np.zeros(feature_dim)
            face_features[~nonzero] = global_avg

        return face_features.astype(np.float32)


class MultiViewImportanceModel(nn.Module):
    """
    Full model for predicting face importance from multi-view CNN features.

    Pipeline:
    1. Render mesh from multiple views
    2. Extract CNN features from each view
    3. Project features to mesh faces
    4. Combine with geometric features
    5. Predict importance score
    """

    def __init__(self, cnn_feature_dim=256, geo_feature_dim=11, hidden_dim=128):
        """
        Initialize the multi-view importance model.

        Args:
            cnn_feature_dim: Dimension of CNN features per face
            geo_feature_dim: Dimension of geometric features per face
            hidden_dim: Hidden dimension for MLP
        """
        super().__init__()

        self.cnn_feature_dim = cnn_feature_dim
        self.geo_feature_dim = geo_feature_dim

        # CNN backbone (shared across views)
        self.cnn = CNNFeatureExtractor(backbone='resnet18', feature_dim=cnn_feature_dim)

        # Feature projector
        self.projector = FeatureProjector()

        # MLP for combining CNN + geometric features -> importance
        combined_dim = cnn_feature_dim + geo_feature_dim

        self.importance_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, cnn_features, geo_features):
        """
        Forward pass given pre-computed CNN features.

        Args:
            cnn_features: (N_faces, cnn_feature_dim)
            geo_features: (N_faces, geo_feature_dim)

        Returns:
            Importance scores (N_faces,)
        """
        # Concatenate features
        combined = torch.cat([cnn_features, geo_features], dim=-1)

        # Predict importance
        importance = self.importance_head(combined).squeeze(-1)

        return importance

    def extract_cnn_features(self, rendered_views):
        """
        Extract CNN features from rendered views and project to faces.

        Args:
            rendered_views: dict from MultiViewRenderer.render_views()

        Returns:
            Face features array (n_faces, cnn_feature_dim)
        """
        color_images = rendered_views['color']
        face_id_maps = rendered_views['face_ids']

        # Determine number of faces from face ID maps
        n_faces = max(fid.max() for fid in face_id_maps) + 1

        # Extract CNN features
        feature_maps = self.cnn.extract_features(color_images)

        # Project to faces
        face_features = self.projector.project_features_fast(
            feature_maps,
            np.stack(face_id_maps),
            n_faces
        )

        return face_features

    def predict(self, cnn_features_np, geo_features_np):
        """
        Predict importance scores from numpy features.

        Args:
            cnn_features_np: (N_faces, cnn_feature_dim) numpy array
            geo_features_np: (N_faces, geo_feature_dim) numpy array

        Returns:
            Importance scores (N_faces,) numpy array
        """
        device = next(self.parameters()).device

        cnn_features = torch.from_numpy(cnn_features_np).float().to(device)
        geo_features = torch.from_numpy(geo_features_np).float().to(device)

        with torch.no_grad():
            importance = self.forward(cnn_features, geo_features)

        return importance.cpu().numpy()


def save_model(model, path, epoch=None, loss=None, optimizer=None):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'cnn_feature_dim': model.cnn_feature_dim,
        'geo_feature_dim': model.geo_feature_dim,
    }
    if epoch is not None:
        checkpoint['epoch'] = epoch
    if loss is not None:
        checkpoint['loss'] = loss
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_model(path, device='cpu'):
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=device)

    model = MultiViewImportanceModel(
        cnn_feature_dim=checkpoint.get('cnn_feature_dim', 256),
        geo_feature_dim=checkpoint.get('geo_feature_dim', 11),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model


if __name__ == "__main__":
    # Test the model
    print("Testing MultiViewImportanceModel...")

    model = MultiViewImportanceModel()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    batch_size = 100
    cnn_features = torch.randn(batch_size, 256)
    geo_features = torch.randn(batch_size, 11)

    importance = model(cnn_features, geo_features)
    print(f"Output shape: {importance.shape}")
    print(f"Output range: [{importance.min():.3f}, {importance.max():.3f}]")

    print("Test passed!")
