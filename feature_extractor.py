"""
Feature Extractor for Mesh Retopology ML

Extracts per-face features from a mesh for use in ML-based importance prediction.
Features include curvature, geometry, and topology metrics.
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree


class FeatureExtractor:
    """Extract ML-relevant features from a mesh."""

    # Feature names for reference
    FEATURE_NAMES = [
        'gaussian_curvature',
        'mean_curvature',
        'face_area_normalized',
        'aspect_ratio',
        'max_dihedral_angle',
        'min_dihedral_angle',
        'avg_vertex_valence',
        'normal_variation',
        'edge_length_variance',
        'face_compactness',
        'local_density',
    ]

    def __init__(self, mesh):
        """
        Initialize with a trimesh mesh object.

        Args:
            mesh: trimesh.Trimesh object
        """
        self.mesh = mesh
        self._precompute_adjacency()

    def _precompute_adjacency(self):
        """Precompute face adjacency for efficient neighbor lookups."""
        # face_adjacency: pairs of adjacent faces
        # face_adjacency_edges: the edge index shared by each pair
        self.face_adj = self.mesh.face_adjacency
        self.face_adj_angles = self.mesh.face_adjacency_angles

        # Build face neighbor lookup
        n_faces = len(self.mesh.faces)
        self.face_neighbors = [[] for _ in range(n_faces)]
        self.face_neighbor_angles = [[] for _ in range(n_faces)]

        for i, (f1, f2) in enumerate(self.face_adj):
            angle = self.face_adj_angles[i]
            self.face_neighbors[f1].append(f2)
            self.face_neighbors[f2].append(f1)
            self.face_neighbor_angles[f1].append(angle)
            self.face_neighbor_angles[f2].append(angle)

    def extract_all_features(self):
        """
        Extract all features for every face in the mesh.

        Returns:
            np.ndarray: Shape (n_faces, n_features) feature matrix
        """
        n_faces = len(self.mesh.faces)
        n_features = len(self.FEATURE_NAMES)
        features = np.zeros((n_faces, n_features), dtype=np.float32)

        # Compute each feature type
        features[:, 0] = self._compute_gaussian_curvature()
        features[:, 1] = self._compute_mean_curvature()
        features[:, 2] = self._compute_face_area_normalized()
        features[:, 3] = self._compute_aspect_ratio()
        features[:, 4], features[:, 5] = self._compute_dihedral_angles()
        features[:, 6] = self._compute_vertex_valence()
        features[:, 7] = self._compute_normal_variation()
        features[:, 8] = self._compute_edge_length_variance()
        features[:, 9] = self._compute_face_compactness()
        features[:, 10] = self._compute_local_density()

        # Handle NaN/Inf values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

        return features

    def _compute_gaussian_curvature(self):
        """
        Compute Gaussian curvature per face.
        Uses discrete Gaussian curvature at vertices, averaged per face.
        """
        try:
            # Get vertex curvatures using trimesh's discrete curvature
            vertex_curvatures = trimesh.curvature.discrete_gaussian_curvature_measure(
                self.mesh, self.mesh.vertices, radius=0.0
            )
            # Average vertex curvatures for each face
            face_curvatures = vertex_curvatures[self.mesh.faces].mean(axis=1)
            # Normalize to reasonable range
            face_curvatures = np.clip(face_curvatures, -10, 10)
            return np.abs(face_curvatures)
        except Exception:
            return np.zeros(len(self.mesh.faces))

    def _compute_mean_curvature(self):
        """
        Compute mean curvature per face.
        Uses discrete mean curvature at vertices, averaged per face.
        """
        try:
            vertex_curvatures = trimesh.curvature.discrete_mean_curvature_measure(
                self.mesh, self.mesh.vertices, radius=0.0
            )
            face_curvatures = vertex_curvatures[self.mesh.faces].mean(axis=1)
            face_curvatures = np.clip(face_curvatures, -10, 10)
            return np.abs(face_curvatures)
        except Exception:
            return np.zeros(len(self.mesh.faces))

    def _compute_face_area_normalized(self):
        """Compute normalized face areas (relative to mesh total area)."""
        areas = self.mesh.area_faces
        total_area = self.mesh.area
        if total_area > 0:
            normalized = areas / total_area * len(self.mesh.faces)
        else:
            normalized = np.ones(len(self.mesh.faces))
        return normalized

    def _compute_aspect_ratio(self):
        """
        Compute aspect ratio for each face (max edge / min edge).
        Values close to 1.0 indicate equilateral triangles.
        """
        vertices = self.mesh.vertices
        faces = self.mesh.faces

        # Get edge lengths for each face
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        e0 = np.linalg.norm(v1 - v0, axis=1)
        e1 = np.linalg.norm(v2 - v1, axis=1)
        e2 = np.linalg.norm(v0 - v2, axis=1)

        edges = np.stack([e0, e1, e2], axis=1)
        max_edge = edges.max(axis=1)
        min_edge = edges.min(axis=1)

        # Avoid division by zero
        min_edge = np.maximum(min_edge, 1e-10)
        aspect_ratio = max_edge / min_edge

        return aspect_ratio

    def _compute_dihedral_angles(self):
        """
        Compute max and min dihedral angles for each face with its neighbors.
        Returns angles in radians.
        """
        n_faces = len(self.mesh.faces)
        max_angles = np.zeros(n_faces)
        min_angles = np.full(n_faces, np.pi)

        for i in range(n_faces):
            angles = self.face_neighbor_angles[i]
            if angles:
                max_angles[i] = max(angles)
                min_angles[i] = min(angles)
            else:
                max_angles[i] = 0
                min_angles[i] = 0

        return max_angles, min_angles

    def _compute_vertex_valence(self):
        """
        Compute average vertex valence for each face.
        Valence = number of edges connected to a vertex.
        """
        # Get vertex valences
        vertex_valences = np.array([
            len(neighbors) for neighbors in self.mesh.vertex_neighbors
        ])

        # Average valence of face's vertices
        face_valences = vertex_valences[self.mesh.faces].mean(axis=1)

        # Normalize: ideal valence for triangular mesh is 6
        return face_valences / 6.0

    def _compute_normal_variation(self):
        """
        Compute normal variation with neighboring faces.
        High values indicate feature edges or high curvature.
        """
        normals = self.mesh.face_normals
        n_faces = len(self.mesh.faces)
        variation = np.zeros(n_faces)

        for i in range(n_faces):
            neighbors = self.face_neighbors[i]
            if neighbors:
                # Compute dot products with neighbor normals
                neighbor_normals = normals[neighbors]
                dots = np.dot(neighbor_normals, normals[i])
                # Variation is 1 - average dot product (0 = identical, 2 = opposite)
                variation[i] = 1.0 - dots.mean()

        return variation

    def _compute_edge_length_variance(self):
        """Compute variance in edge lengths for each face."""
        vertices = self.mesh.vertices
        faces = self.mesh.faces

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        e0 = np.linalg.norm(v1 - v0, axis=1)
        e1 = np.linalg.norm(v2 - v1, axis=1)
        e2 = np.linalg.norm(v0 - v2, axis=1)

        edges = np.stack([e0, e1, e2], axis=1)
        variance = edges.var(axis=1)

        # Normalize by mean edge length squared
        mean_edge = edges.mean(axis=1)
        mean_edge = np.maximum(mean_edge, 1e-10)
        normalized_variance = variance / (mean_edge ** 2)

        return normalized_variance

    def _compute_face_compactness(self):
        """
        Compute compactness of each face (how close to equilateral).
        Uses ratio of area to perimeter squared, normalized.
        """
        vertices = self.mesh.vertices
        faces = self.mesh.faces

        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # Perimeter
        e0 = np.linalg.norm(v1 - v0, axis=1)
        e1 = np.linalg.norm(v2 - v1, axis=1)
        e2 = np.linalg.norm(v0 - v2, axis=1)
        perimeter = e0 + e1 + e2

        # Area
        areas = self.mesh.area_faces

        # Compactness: 4 * sqrt(3) * area / perimeter^2
        # For equilateral triangle, this equals 1
        perimeter_sq = perimeter ** 2
        perimeter_sq = np.maximum(perimeter_sq, 1e-10)
        compactness = 4 * np.sqrt(3) * areas / perimeter_sq

        return compactness

    def _compute_local_density(self):
        """
        Compute local face density (faces per unit area in neighborhood).
        High density indicates detailed regions.
        """
        # Use face centroids
        centroids = self.mesh.triangles_center

        # Build KD-tree for fast neighbor queries
        tree = cKDTree(centroids)

        # Query radius based on average edge length
        avg_edge = np.mean([
            np.linalg.norm(
                self.mesh.vertices[self.mesh.faces[:, 1]] -
                self.mesh.vertices[self.mesh.faces[:, 0]],
                axis=1
            ).mean()
        ])
        radius = avg_edge * 3

        # Count neighbors within radius for each face
        densities = np.array([
            len(tree.query_ball_point(c, radius))
            for c in centroids
        ], dtype=np.float32)

        # Normalize
        if densities.max() > 0:
            densities = densities / densities.max()

        return densities

    def normalize_features(self, features):
        """
        Normalize features to [0, 1] range using min-max scaling.

        Args:
            features: np.ndarray of shape (n_faces, n_features)

        Returns:
            Normalized feature matrix
        """
        min_vals = features.min(axis=0, keepdims=True)
        max_vals = features.max(axis=0, keepdims=True)
        range_vals = max_vals - min_vals
        range_vals = np.maximum(range_vals, 1e-10)  # Avoid division by zero

        normalized = (features - min_vals) / range_vals
        return normalized


def extract_features(mesh):
    """
    Convenience function to extract normalized features from a mesh.

    Args:
        mesh: trimesh.Trimesh object or path to mesh file

    Returns:
        np.ndarray: Normalized feature matrix (n_faces, n_features)
    """
    if isinstance(mesh, str):
        mesh = trimesh.load(mesh)

    extractor = FeatureExtractor(mesh)
    features = extractor.extract_all_features()
    normalized = extractor.normalize_features(features)

    return normalized


if __name__ == "__main__":
    # Test on a simple mesh
    print("Creating test mesh...")
    mesh = trimesh.creation.icosphere(subdivisions=3)

    print(f"Mesh has {len(mesh.faces)} faces")

    extractor = FeatureExtractor(mesh)
    features = extractor.extract_all_features()
    normalized = extractor.normalize_features(features)

    print(f"\nExtracted features shape: {features.shape}")
    print(f"Feature names: {extractor.FEATURE_NAMES}")

    print("\nFeature statistics (raw):")
    for i, name in enumerate(extractor.FEATURE_NAMES):
        print(f"  {name}: min={features[:, i].min():.4f}, "
              f"max={features[:, i].max():.4f}, "
              f"mean={features[:, i].mean():.4f}")

    print("\nFeature statistics (normalized):")
    for i, name in enumerate(extractor.FEATURE_NAMES):
        print(f"  {name}: min={normalized[:, i].min():.4f}, "
              f"max={normalized[:, i].max():.4f}, "
              f"mean={normalized[:, i].mean():.4f}")
