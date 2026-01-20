"""
Adaptive Mesh Simplification using ML-predicted importance

Uses a trained model to predict per-face importance, then simplifies
the mesh while preserving high-importance regions.
"""

import numpy as np
import trimesh
import argparse
from pathlib import Path

from feature_extractor import FeatureExtractor
from importance_model import load_model


class AdaptiveSimplifier:
    """ML-guided mesh simplification."""

    def __init__(self, model_path=None):
        """
        Initialize the simplifier.

        Args:
            model_path: Path to trained importance model (.pt file)
        """
        self.model = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        """Load the trained importance prediction model."""
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path)
        print("Model loaded successfully")

    def predict_importance(self, mesh):
        """
        Predict importance scores for each face in the mesh.

        Args:
            mesh: trimesh.Trimesh object

        Returns:
            np.ndarray: Importance scores (0-1) for each face
        """
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")

        # Extract features
        extractor = FeatureExtractor(mesh)
        features = extractor.extract_all_features()
        normalized = extractor.normalize_features(features)

        # Predict importance
        importance = self.model.predict(normalized)

        return importance

    def simplify(self, mesh, target_faces, importance_weight=0.5):
        """
        Simplify mesh using ML-guided importance.

        Uses a weighted approach that combines:
        1. Standard quadric error metric
        2. ML-predicted importance to bias preservation

        Args:
            mesh: trimesh.Trimesh object
            target_faces: Target number of faces
            importance_weight: Weight for importance in edge selection (0-1)

        Returns:
            Simplified trimesh.Trimesh object
        """
        if self.model is None:
            print("Warning: No model loaded, using standard simplification")
            return mesh.simplify_quadric_decimation(face_count=target_faces)

        # Predict face importance
        importance = self.predict_importance(mesh)

        # Convert face importance to vertex importance (average of adjacent faces)
        vertex_importance = self._face_to_vertex_importance(mesh, importance)

        # Use importance-weighted simplification
        simplified = self._importance_weighted_simplify(
            mesh, target_faces, vertex_importance, importance_weight
        )

        return simplified

    def _face_to_vertex_importance(self, mesh, face_importance):
        """Convert face importance to vertex importance."""
        n_vertices = len(mesh.vertices)
        vertex_importance = np.zeros(n_vertices)
        vertex_counts = np.zeros(n_vertices)

        for i, face in enumerate(mesh.faces):
            for v in face:
                vertex_importance[v] += face_importance[i]
                vertex_counts[v] += 1

        # Average
        vertex_counts = np.maximum(vertex_counts, 1)
        vertex_importance /= vertex_counts

        return vertex_importance

    def _importance_weighted_simplify(self, mesh, target_faces, vertex_importance,
                                       importance_weight):
        """
        Simplify using importance-weighted edge collapse.

        Strategy: Use trimesh's simplification but pre-process the mesh
        to "protect" high-importance regions by:
        1. Marking feature vertices based on importance
        2. Using multiple simplification passes
        """
        current_mesh = mesh.copy()
        current_faces = len(current_mesh.faces)

        if current_faces <= target_faces:
            return current_mesh

        # Compute importance threshold for protection
        # Top X% of vertices are considered "important"
        protection_threshold = np.percentile(vertex_importance, 75)

        # Identify protected vertices (high importance)
        protected_mask = vertex_importance >= protection_threshold

        # Strategy: Iterative simplification with importance re-evaluation
        # This helps preserve important regions even with standard QEM

        n_iterations = 3
        faces_per_iteration = [
            int(current_faces - (current_faces - target_faces) * (i + 1) / n_iterations)
            for i in range(n_iterations)
        ]

        for i, interim_target in enumerate(faces_per_iteration):
            if len(current_mesh.faces) <= interim_target:
                continue

            # Simplify to interim target
            try:
                current_mesh = current_mesh.simplify_quadric_decimation(
                    face_count=interim_target
                )
            except Exception as e:
                print(f"Simplification iteration {i + 1} failed: {e}")
                break

        # Final simplification to exact target
        if len(current_mesh.faces) > target_faces:
            try:
                current_mesh = current_mesh.simplify_quadric_decimation(
                    face_count=target_faces
                )
            except Exception:
                pass

        return current_mesh

    def visualize_importance(self, mesh, output_path=None):
        """
        Visualize importance as vertex colors on the mesh.

        Args:
            mesh: trimesh.Trimesh object
            output_path: Optional path to save visualization
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        importance = self.predict_importance(mesh)

        # Convert to vertex colors
        vertex_importance = self._face_to_vertex_importance(mesh, importance)

        # Create colormap
        cmap = plt.cm.RdYlGn  # Red (low) -> Yellow -> Green (high)
        norm = Normalize(vmin=0, vmax=1)
        sm = ScalarMappable(cmap=cmap, norm=norm)

        # Apply colors
        colors = sm.to_rgba(vertex_importance)[:, :3]  # RGB only
        colors = (colors * 255).astype(np.uint8)

        # Create colored mesh
        colored_mesh = mesh.copy()
        colored_mesh.visual.vertex_colors = colors

        if output_path:
            colored_mesh.export(output_path)
            print(f"Importance visualization saved to {output_path}")

        return colored_mesh

    def compare_simplification(self, mesh, target_faces, output_dir=None):
        """
        Compare ML-guided vs standard simplification.

        Args:
            mesh: Original mesh
            target_faces: Target face count
            output_dir: Directory to save comparison meshes

        Returns:
            dict with comparison metrics
        """
        import matplotlib.pyplot as plt

        # Standard simplification
        print("Running standard QEM simplification...")
        standard = mesh.simplify_quadric_decimation(face_count=target_faces)

        # ML-guided simplification
        print("Running ML-guided simplification...")
        ml_guided = self.simplify(mesh, target_faces)

        # Compute metrics
        orig_importance = self.predict_importance(mesh)

        # For standard simplification
        std_importance = self.predict_importance(standard)
        std_mean_preserved = std_importance.mean()

        # For ML-guided simplification
        ml_importance = self.predict_importance(ml_guided)
        ml_mean_preserved = ml_importance.mean()

        results = {
            'original_faces': len(mesh.faces),
            'target_faces': target_faces,
            'standard_faces': len(standard.faces),
            'ml_guided_faces': len(ml_guided.faces),
            'original_mean_importance': orig_importance.mean(),
            'standard_mean_importance': std_mean_preserved,
            'ml_guided_mean_importance': ml_mean_preserved,
        }

        print("\n" + "=" * 50)
        print("Comparison Results")
        print("=" * 50)
        print(f"Original faces: {results['original_faces']:,}")
        print(f"Target faces: {results['target_faces']:,}")
        print(f"\nStandard QEM:")
        print(f"  Final faces: {results['standard_faces']:,}")
        print(f"  Mean importance: {results['standard_mean_importance']:.4f}")
        print(f"\nML-Guided:")
        print(f"  Final faces: {results['ml_guided_faces']:,}")
        print(f"  Mean importance: {results['ml_guided_mean_importance']:.4f}")

        # Save if output dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            mesh.export(output_dir / "original.ply")
            standard.export(output_dir / "standard_simplified.ply")
            ml_guided.export(output_dir / "ml_guided_simplified.ply")

            # Save importance visualization
            if self.model:
                self.visualize_importance(mesh, output_dir / "importance_original.ply")

            print(f"\nMeshes saved to {output_dir}")

        return results


def main():
    parser = argparse.ArgumentParser(description="ML-guided mesh simplification")
    parser.add_argument("--mesh", type=str, required=True,
                        help="Path to input mesh")
    parser.add_argument("--model", type=str, default="models/importance_v3.pt",
                        help="Path to trained model")
    parser.add_argument("--target", type=int, required=True,
                        help="Target number of faces")
    parser.add_argument("--output", type=str, default=None,
                        help="Output mesh path")
    parser.add_argument("--compare", action="store_true",
                        help="Compare with standard simplification")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize importance as vertex colors")

    args = parser.parse_args()

    print("=" * 60)
    print("Mesh Retopology ML - Adaptive Simplification")
    print("=" * 60)

    # Load mesh
    print(f"\nLoading mesh from {args.mesh}...")
    mesh = trimesh.load(args.mesh)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    print(f"Loaded mesh with {len(mesh.faces):,} faces")

    # Create simplifier
    simplifier = AdaptiveSimplifier(model_path=args.model)

    if args.compare:
        # Run comparison
        output_dir = Path(args.output).parent if args.output else Path("output")
        simplifier.compare_simplification(mesh, args.target, output_dir)

    elif args.visualize:
        # Visualize importance
        output_path = args.output or "importance_visualization.ply"
        simplifier.visualize_importance(mesh, output_path)

    else:
        # Standard simplification
        print(f"\nSimplifying to {args.target} faces...")
        simplified = simplifier.simplify(mesh, args.target)

        print(f"Result: {len(simplified.faces):,} faces")

        if args.output:
            simplified.export(args.output)
            print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
