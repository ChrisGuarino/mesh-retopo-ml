"""
Dataset Generator for Mesh Retopology ML

Generates synthetic training pairs:
- High-poly meshes with various geometries (organic + hard surface)
- Ground truth simplifications
- Per-face importance labels based on which faces survive simplification
"""

import numpy as np
import trimesh
from pathlib import Path
import argparse
from feature_extractor import FeatureExtractor


class DatasetGenerator:
    """Generate synthetic mesh pairs for training importance prediction."""

    def __init__(self, output_dir="data"):
        """
        Initialize the generator.

        Args:
            output_dir: Directory to save generated data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_organic_meshes(self, count=10):
        """
        Generate organic-style meshes (smooth surfaces).

        Args:
            count: Number of meshes to generate

        Returns:
            List of trimesh.Trimesh objects
        """
        meshes = []

        for i in range(count):
            mesh_type = i % 5

            if mesh_type == 0:
                # Icosphere with varying subdivision
                subdiv = np.random.randint(3, 6)
                mesh = trimesh.creation.icosphere(subdivisions=subdiv)

            elif mesh_type == 1:
                # Torus
                major_r = np.random.uniform(1.0, 2.0)
                minor_r = np.random.uniform(0.2, 0.5)
                mesh = trimesh.creation.torus(
                    major_radius=major_r,
                    minor_radius=minor_r,
                    major_sections=64,
                    minor_sections=32
                )

            elif mesh_type == 2:
                # Capsule (cylinder with rounded ends)
                radius = np.random.uniform(0.3, 0.7)
                height = np.random.uniform(1.0, 3.0)
                mesh = trimesh.creation.capsule(radius=radius, height=height)

            elif mesh_type == 3:
                # Perlin noise deformed sphere
                mesh = trimesh.creation.icosphere(subdivisions=4)
                mesh = self._apply_noise_displacement(mesh, scale=0.1)

            else:
                # UV sphere
                mesh = trimesh.creation.uv_sphere(radius=1.0, count=[64, 32])

            # Random scaling
            scale = np.random.uniform(0.5, 2.0, size=3)
            mesh.apply_scale(scale)

            meshes.append(mesh)

        return meshes

    def generate_hard_surface_meshes(self, count=10):
        """
        Generate hard surface meshes (sharp edges, flat planes).

        Args:
            count: Number of meshes to generate

        Returns:
            List of trimesh.Trimesh objects
        """
        meshes = []

        for i in range(count):
            mesh_type = i % 5

            if mesh_type == 0:
                # Box with random dimensions
                extents = np.random.uniform(0.5, 2.0, size=3)
                mesh = trimesh.creation.box(extents=extents)
                # Subdivide to get more faces
                mesh = mesh.subdivide()

            elif mesh_type == 1:
                # Cylinder
                radius = np.random.uniform(0.3, 1.0)
                height = np.random.uniform(1.0, 3.0)
                mesh = trimesh.creation.cylinder(
                    radius=radius,
                    height=height,
                    sections=32
                )

            elif mesh_type == 2:
                # Cone
                radius = np.random.uniform(0.5, 1.5)
                height = np.random.uniform(1.0, 2.5)
                mesh = trimesh.creation.cone(
                    radius=radius,
                    height=height,
                    sections=32
                )

            elif mesh_type == 3:
                # Extruded polygon (prism) - use trimesh primitive instead
                # Create a box and scale it to make rectangular prisms
                extents = np.random.uniform(0.5, 2.0, size=3)
                mesh = trimesh.creation.box(extents=extents)
                # Apply random rotation for variety
                angle = np.random.uniform(0, np.pi / 4)
                rotation = trimesh.transformations.rotation_matrix(
                    angle, [0, 0, 1]
                )
                mesh.apply_transform(rotation)

            else:
                # Annulus (ring shape)
                mesh = trimesh.creation.annulus(
                    r_min=0.3,
                    r_max=1.0,
                    height=0.3,
                    sections=32
                )

            # Subdivide to increase face count
            if len(mesh.faces) < 500:
                mesh = mesh.subdivide()

            meshes.append(mesh)

        return meshes

    def _apply_noise_displacement(self, mesh, scale=0.1):
        """Apply Perlin-like noise displacement to mesh vertices."""
        vertices = mesh.vertices.copy()
        normals = mesh.vertex_normals

        # Simple noise based on position
        noise = np.sin(vertices[:, 0] * 5) * np.cos(vertices[:, 1] * 5) * np.sin(vertices[:, 2] * 5)
        noise += np.sin(vertices[:, 0] * 10) * 0.5
        noise += np.cos(vertices[:, 1] * 7) * 0.3

        # Displace along normals
        displacement = normals * noise[:, np.newaxis] * scale
        vertices += displacement

        return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)

    def compute_face_importance(self, original_mesh, simplified_mesh):
        """
        Compute importance labels by checking which regions survive simplification.

        Uses nearest-face mapping to determine if original faces are preserved.

        Args:
            original_mesh: High-poly trimesh object
            simplified_mesh: Simplified trimesh object

        Returns:
            np.ndarray: Importance scores (0-1) for each face in original mesh
        """
        # Get face centroids
        orig_centroids = original_mesh.triangles_center
        simp_centroids = simplified_mesh.triangles_center

        # For each original face, find distance to nearest simplified face
        from scipy.spatial import cKDTree
        tree = cKDTree(simp_centroids)
        distances, indices = tree.query(orig_centroids)

        # Normalize distances
        # Faces closer to simplified faces get higher importance
        max_dist = distances.max()
        if max_dist > 0:
            # Inverse distance -> importance
            importance = 1.0 - (distances / max_dist)
        else:
            importance = np.ones(len(orig_centroids))

        # Also consider if the simplified face normal matches
        orig_normals = original_mesh.face_normals
        simp_normals = simplified_mesh.face_normals

        # Dot product between original and nearest simplified normal
        matched_normals = simp_normals[indices]
        normal_similarity = np.sum(orig_normals * matched_normals, axis=1)
        normal_similarity = np.clip(normal_similarity, 0, 1)

        # Combine distance and normal similarity
        importance = importance * 0.5 + normal_similarity * 0.5

        return importance.astype(np.float32)

    def generate_training_sample(self, mesh, target_ratio=0.2):
        """
        Generate a single training sample from a mesh.

        Args:
            mesh: Original high-poly mesh
            target_ratio: Target face count as ratio of original

        Returns:
            dict with features, labels, and metadata
        """
        # Ensure mesh is a proper Trimesh (not Scene)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)

        original_faces = len(mesh.faces)
        target_faces = max(int(original_faces * target_ratio), 50)

        # Extract features from original mesh
        extractor = FeatureExtractor(mesh)
        features = extractor.extract_all_features()
        normalized_features = extractor.normalize_features(features)

        # Simplify mesh
        try:
            simplified = mesh.simplify_quadric_decimation(face_count=target_faces)
        except Exception as e:
            print(f"Simplification failed: {e}")
            return None

        # Compute importance labels
        importance = self.compute_face_importance(mesh, simplified)

        return {
            'features': normalized_features,
            'importance': importance,
            'original_faces': original_faces,
            'simplified_faces': len(simplified.faces),
            'target_ratio': target_ratio,
        }

    def generate_dataset(self, num_meshes=100, ratios=[0.1, 0.2, 0.3, 0.5]):
        """
        Generate a full training dataset.

        Args:
            num_meshes: Total number of unique meshes to generate
            ratios: List of simplification ratios to use for each mesh

        Returns:
            List of training samples
        """
        samples = []

        # Generate mesh types
        organic_count = num_meshes // 2
        hard_count = num_meshes - organic_count

        print(f"Generating {organic_count} organic meshes...")
        organic_meshes = self.generate_organic_meshes(organic_count)

        print(f"Generating {hard_count} hard surface meshes...")
        hard_meshes = self.generate_hard_surface_meshes(hard_count)

        all_meshes = organic_meshes + hard_meshes

        print(f"\nProcessing {len(all_meshes)} meshes with {len(ratios)} ratios each...")

        for i, mesh in enumerate(all_meshes):
            for ratio in ratios:
                sample = self.generate_training_sample(mesh, target_ratio=ratio)
                if sample is not None:
                    sample['mesh_id'] = i
                    sample['ratio_id'] = ratios.index(ratio)
                    samples.append(sample)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(all_meshes)} meshes")

        return samples

    def save_dataset(self, samples, filename="training_data.npz"):
        """
        Save dataset to disk.

        Args:
            samples: List of sample dictionaries
            filename: Output filename
        """
        # Concatenate all samples
        all_features = []
        all_importance = []
        all_mesh_ids = []

        for s in samples:
            n_faces = len(s['features'])
            all_features.append(s['features'])
            all_importance.append(s['importance'])
            all_mesh_ids.append(np.full(n_faces, s['mesh_id']))

        features = np.vstack(all_features)
        importance = np.concatenate(all_importance)
        mesh_ids = np.concatenate(all_mesh_ids)

        output_path = self.output_dir / filename
        np.savez_compressed(
            output_path,
            features=features,
            importance=importance,
            mesh_ids=mesh_ids,
            n_samples=len(samples)
        )

        print(f"\nDataset saved to {output_path}")
        print(f"  Total faces: {len(features):,}")
        print(f"  Feature dimensions: {features.shape[1]}")
        print(f"  Unique meshes: {len(set(mesh_ids))}")


def main():
    parser = argparse.ArgumentParser(description="Generate training dataset for mesh retopology ML")
    parser.add_argument("--num_meshes", type=int, default=100,
                        help="Number of meshes to generate")
    parser.add_argument("--output", type=str, default="data",
                        help="Output directory")
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5],
                        help="Simplification ratios to use")

    args = parser.parse_args()

    generator = DatasetGenerator(output_dir=args.output)

    print("=" * 50)
    print("Mesh Retopology ML - Dataset Generator")
    print("=" * 50)
    print(f"Generating {args.num_meshes} meshes")
    print(f"Simplification ratios: {args.ratios}")
    print(f"Output directory: {args.output}")
    print("=" * 50)

    samples = generator.generate_dataset(
        num_meshes=args.num_meshes,
        ratios=args.ratios
    )

    generator.save_dataset(samples)

    print("\nDone!")


if __name__ == "__main__":
    main()
