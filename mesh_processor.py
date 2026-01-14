import trimesh
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class MeshProcessor:
    def __init__(self, mesh_path=None):
        """Initialize with an optional mesh file path."""
        self.mesh = None
        if mesh_path:
            self.load_mesh(mesh_path)

    def load_mesh(self, mesh_path):
        """Load a mesh from file."""
        print(f"Loading mesh from: {mesh_path}")
        self.mesh = trimesh.load(mesh_path)
        print(f"Mesh loaded successfully")
        return self.mesh

    def get_stats(self):
        """Get basic statistics about the mesh."""
        if self.mesh is None:
            print("No mesh loaded")
            return None

        stats = {
            'vertices': len(self.mesh.vertices),
            'faces': len(self.mesh.faces),
            'edges': len(self.mesh.edges),
            'watertight': self.mesh.is_watertight,
            'volume': self.mesh.volume if self.mesh.is_watertight else None,
            'surface_area': self.mesh.area,
            'bounds': self.mesh.bounds,
        }

        return stats

    def print_stats(self):
        """Print mesh statistics."""
        stats = self.get_stats()
        if stats:
            print("\n=== Mesh Statistics ===")
            print(f"Vertices: {stats['vertices']:,}")
            print(f"Faces: {stats['faces']:,}")
            print(f"Edges: {stats['edges']:,}")
            print(f"Watertight: {stats['watertight']}")
            if stats['volume']:
                print(f"Volume: {stats['volume']:.4f}")
            print(f"Surface Area: {stats['surface_area']:.4f}")
            print(f"Bounds: {stats['bounds']}")

    def simplify(self, target_faces):
        """Simplify the mesh to a target number of faces."""
        if self.mesh is None:
            print("No mesh loaded")
            return None

        print(f"\nSimplifying mesh from {len(self.mesh.faces)} to {target_faces} faces...")
        simplified = self.mesh.simplify_quadric_decimation(face_count=target_faces)
        print(f"Simplified to {len(simplified.faces)} faces")

        return simplified

    def visualize(self, mesh=None):
        """Visualize the mesh using trimesh's viewer."""
        target_mesh = mesh if mesh is not None else self.mesh
        if target_mesh is None:
            print("No mesh to visualize")
            return

        target_mesh.show()

    def plot_topology_info(self):
        """Plot information about mesh topology."""
        if self.mesh is None:
            print("No mesh loaded")
            return

        # Analyze vertex valence (number of edges per vertex)
        vertex_neighbors = self.mesh.vertex_neighbors
        valences = [len(neighbors) for neighbors in vertex_neighbors]

        # Analyze face types (triangles, quads, n-gons)
        faces_per_face = [len(face) for face in self.mesh.faces]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Valence distribution
        axes[0].hist(valences, bins=range(min(valences), max(valences) + 2),
                     edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Vertex Valence')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Vertex Valence Distribution')
        axes[0].grid(True, alpha=0.3)

        # Face type distribution
        unique, counts = np.unique(faces_per_face, return_counts=True)
        axes[1].bar(unique, counts, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Vertices per Face')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Face Type Distribution')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('topology_analysis.png', dpi=150, bbox_inches='tight')
        print("Topology analysis saved to topology_analysis.png")
        plt.close()


def main():
    """Example usage of the MeshProcessor."""
    # Create an example mesh (subdivided sphere for testing)
    print("Creating example mesh (subdivided sphere)...")
    mesh = trimesh.creation.icosphere(subdivisions=4)

    # Save it locally for future use
    mesh.export('test_mesh.ply')
    print("Saved test_mesh.ply to working directory")

    # Create processor and analyze
    processor = MeshProcessor('test_meshes/bigfoot.stl')
    processor.print_stats()

    # Plot topology info
    processor.plot_topology_info()

    # Simplify the mesh
    simplified = processor.simplify(target_faces=100)

    # Show stats for simplified mesh
    print("\n=== Simplified Mesh ===")
    processor.mesh = simplified
    processor.print_stats()

    # Export simplified mesh
    simplified.export('test_mesh_simplified.ply')
    print("\nSimplified mesh saved to test_mesh_simplified.ply")

    # Visualize (uncomment to see 3D viewer)
    # processor.visualize()


if __name__ == "__main__":
    main()
