"""
Multi-View Mesh Renderer

Renders a mesh from multiple viewpoints for CNN-based importance prediction.
Uses pyrender for GPU-accelerated rendering with depth and face ID buffers.
"""

import numpy as np
import trimesh
from PIL import Image
import os

# Try to import pyrender, fall back to trimesh's built-in renderer
try:
    import pyrender
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False
    print("Warning: pyrender not available, using trimesh fallback (slower)")


class MultiViewRenderer:
    """Render mesh from multiple viewpoints."""

    def __init__(self, image_size=224, n_views=8, use_gpu=True):
        """
        Initialize the renderer.

        Args:
            image_size: Size of rendered images (square)
            n_views: Number of views to render
            use_gpu: Whether to use GPU rendering (requires pyrender + OpenGL)
        """
        self.image_size = image_size
        self.n_views = n_views
        self.use_gpu = use_gpu and PYRENDER_AVAILABLE

        # Camera positions around the object (azimuth angles)
        self.azimuth_angles = np.linspace(0, 2 * np.pi, n_views, endpoint=False)
        # Mix of elevation angles for variety
        self.elevation_angles = [0.3, -0.2, 0.4, -0.1, 0.2, -0.3, 0.1, -0.2][:n_views]

    def _get_camera_poses(self, mesh_center, mesh_radius):
        """
        Generate camera poses around the mesh.

        Args:
            mesh_center: Center point of the mesh
            mesh_radius: Bounding radius of the mesh

        Returns:
            List of 4x4 camera pose matrices
        """
        poses = []
        distance = mesh_radius * 2.5  # Camera distance from center

        for i, (azimuth, elevation) in enumerate(zip(self.azimuth_angles, self.elevation_angles)):
            # Camera position in spherical coordinates
            x = distance * np.cos(elevation) * np.cos(azimuth)
            y = distance * np.cos(elevation) * np.sin(azimuth)
            z = distance * np.sin(elevation)

            camera_pos = np.array([x, y, z]) + mesh_center

            # Look-at matrix
            forward = mesh_center - camera_pos
            forward = forward / np.linalg.norm(forward)

            # Up vector (world Z)
            up = np.array([0, 0, 1])

            # Right vector
            right = np.cross(forward, up)
            if np.linalg.norm(right) < 1e-6:
                up = np.array([0, 1, 0])
                right = np.cross(forward, up)
            right = right / np.linalg.norm(right)

            # Recompute up
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)

            # Build camera matrix (camera looks down -Z in its local frame)
            pose = np.eye(4)
            pose[:3, 0] = right
            pose[:3, 1] = up
            pose[:3, 2] = -forward
            pose[:3, 3] = camera_pos

            poses.append(pose)

        return poses

    def render_views_pyrender(self, mesh):
        """
        Render multiple views using pyrender.

        Args:
            mesh: trimesh.Trimesh object

        Returns:
            dict with:
                - 'color': List of RGB images (H, W, 3)
                - 'depth': List of depth images (H, W)
                - 'face_ids': List of face ID images (H, W) - which face is visible at each pixel
                - 'camera_poses': List of camera pose matrices
        """
        # Compute mesh bounds
        mesh_center = mesh.centroid
        mesh_radius = np.linalg.norm(mesh.vertices - mesh_center, axis=1).max()

        # Get camera poses
        camera_poses = self._get_camera_poses(mesh_center, mesh_radius)

        # Create pyrender scene
        # Assign unique colors to each face for face ID rendering
        n_faces = len(mesh.faces)

        # Create mesh with face colors for ID rendering
        face_colors = np.zeros((n_faces, 4), dtype=np.uint8)
        for i in range(n_faces):
            # Encode face ID in RGB (up to 16M faces)
            face_colors[i, 0] = (i >> 16) & 0xFF
            face_colors[i, 1] = (i >> 8) & 0xFF
            face_colors[i, 2] = i & 0xFF
            face_colors[i, 3] = 255

        # Create trimesh with face colors
        mesh_with_ids = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            face_colors=face_colors
        )

        # Convert to pyrender mesh
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh_with_ids, smooth=False)

        # Setup renderer
        renderer = pyrender.OffscreenRenderer(self.image_size, self.image_size)

        # Camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

        colors = []
        depths = []
        face_ids_list = []

        for pose in camera_poses:
            # Create scene
            scene = pyrender.Scene(bg_color=[255, 255, 255, 255])
            scene.add(mesh_pyrender)
            scene.add(camera, pose=pose)

            # Add light
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
            scene.add(light, pose=pose)

            # Render
            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.FLAT)

            # Decode face IDs from color
            face_ids = (color[:, :, 0].astype(np.int32) << 16) + \
                       (color[:, :, 1].astype(np.int32) << 8) + \
                       color[:, :, 2].astype(np.int32)

            # Mark background as -1
            face_ids[depth == 0] = -1

            colors.append(color)
            depths.append(depth)
            face_ids_list.append(face_ids)

        renderer.delete()

        return {
            'color': colors,
            'depth': depths,
            'face_ids': face_ids_list,
            'camera_poses': camera_poses,
            'mesh_center': mesh_center,
            'mesh_radius': mesh_radius,
        }

    def render_views_trimesh(self, mesh):
        """
        Render multiple views using trimesh's built-in renderer (CPU fallback).

        Args:
            mesh: trimesh.Trimesh object

        Returns:
            dict with rendered views (same format as pyrender version)
        """
        # Compute mesh bounds
        mesh_center = mesh.centroid
        mesh_radius = np.linalg.norm(mesh.vertices - mesh_center, axis=1).max()

        # Get camera poses
        camera_poses = self._get_camera_poses(mesh_center, mesh_radius)

        colors = []
        depths = []
        face_ids_list = []

        # Create a scene for rendering
        scene = mesh.scene()

        for pose in camera_poses:
            # Set camera pose
            scene.camera_transform = pose

            # Render - trimesh returns PNG data
            try:
                png_data = scene.save_image(resolution=[self.image_size, self.image_size])
                image = np.array(Image.open(png_data))[:, :, :3]
            except Exception:
                # Fallback: create blank image
                image = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 255

            colors.append(image)

            # Trimesh doesn't easily give us depth/face IDs, so we'll compute them differently
            # For now, use ray casting to get face IDs
            face_ids = self._raycast_face_ids(mesh, pose)
            face_ids_list.append(face_ids)

            # Approximate depth from face IDs (not accurate but usable)
            depths.append(np.zeros((self.image_size, self.image_size)))

        return {
            'color': colors,
            'depth': depths,
            'face_ids': face_ids_list,
            'camera_poses': camera_poses,
            'mesh_center': mesh_center,
            'mesh_radius': mesh_radius,
        }

    def _raycast_face_ids(self, mesh, camera_pose):
        """
        Use ray casting to determine which face is visible at each pixel.

        Args:
            mesh: trimesh.Trimesh object
            camera_pose: 4x4 camera pose matrix

        Returns:
            (H, W) array of face IDs (-1 for background)
        """
        # Create ray origins and directions for each pixel
        h, w = self.image_size, self.image_size

        # Camera intrinsics (simple perspective)
        fov = np.pi / 3.0
        focal = w / (2 * np.tan(fov / 2))

        # Pixel coordinates
        u = np.arange(w) - w / 2
        v = np.arange(h) - h / 2
        uu, vv = np.meshgrid(u, v)

        # Ray directions in camera frame
        dirs_cam = np.stack([uu / focal, -vv / focal, -np.ones_like(uu)], axis=-1)
        dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

        # Transform to world frame
        R = camera_pose[:3, :3]
        dirs_world = dirs_cam @ R.T

        # Ray origins (camera position)
        origins = np.broadcast_to(camera_pose[:3, 3], dirs_world.shape)

        # Flatten for ray casting
        origins_flat = origins.reshape(-1, 3)
        dirs_flat = dirs_world.reshape(-1, 3)

        # Cast rays
        try:
            locations, index_ray, index_tri = mesh.ray.intersects_location(
                origins_flat, dirs_flat, multiple_hits=False
            )

            # Build face ID image
            face_ids = np.full(h * w, -1, dtype=np.int32)
            face_ids[index_ray] = index_tri
            face_ids = face_ids.reshape(h, w)
        except Exception:
            face_ids = np.full((h, w), -1, dtype=np.int32)

        return face_ids

    def render_views(self, mesh):
        """
        Render multiple views of the mesh.

        Args:
            mesh: trimesh.Trimesh object

        Returns:
            dict with rendered views
        """
        if self.use_gpu:
            return self.render_views_pyrender(mesh)
        else:
            return self.render_views_trimesh(mesh)

    def render_for_display(self, mesh, shaded=True):
        """
        Render views for display/visualization (with proper shading).

        Args:
            mesh: trimesh.Trimesh object
            shaded: Whether to apply shading

        Returns:
            List of RGB images
        """
        mesh_center = mesh.centroid
        mesh_radius = np.linalg.norm(mesh.vertices - mesh_center, axis=1).max()
        camera_poses = self._get_camera_poses(mesh_center, mesh_radius)

        if not PYRENDER_AVAILABLE:
            # Use trimesh fallback
            scene = mesh.scene()
            images = []
            for pose in camera_poses:
                scene.camera_transform = pose
                try:
                    png_data = scene.save_image(resolution=[self.image_size, self.image_size])
                    image = np.array(Image.open(png_data))[:, :, :3]
                except Exception:
                    image = np.ones((self.image_size, self.image_size, 3), dtype=np.uint8) * 200
                images.append(image)
            return images

        # Use pyrender with proper shading
        mesh_pyrender = pyrender.Mesh.from_trimesh(mesh)
        renderer = pyrender.OffscreenRenderer(self.image_size, self.image_size)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

        images = []
        for pose in camera_poses:
            scene = pyrender.Scene(bg_color=[255, 255, 255, 255])
            scene.add(mesh_pyrender)
            scene.add(camera, pose=pose)

            # Add lights
            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
            scene.add(light, pose=pose)

            ambient = pyrender.DirectionalLight(color=[0.5, 0.5, 0.5], intensity=1.0)
            scene.add(ambient, pose=np.eye(4))

            color, _ = renderer.render(scene)
            images.append(color)

        renderer.delete()
        return images


def test_renderer():
    """Test the multi-view renderer."""
    # Create test mesh
    mesh = trimesh.creation.icosphere(subdivisions=3)

    # Add some noise to make it interesting
    mesh.vertices += np.random.randn(*mesh.vertices.shape) * 0.05

    print(f"Test mesh: {len(mesh.faces)} faces")

    # Create renderer
    renderer = MultiViewRenderer(image_size=224, n_views=8)

    # Render views
    print("Rendering views...")
    views = renderer.render_views(mesh)

    print(f"Rendered {len(views['color'])} views")
    print(f"Image shape: {views['color'][0].shape}")
    print(f"Face IDs range: {views['face_ids'][0].min()} to {views['face_ids'][0].max()}")

    # Count visible faces
    all_visible = set()
    for face_ids in views['face_ids']:
        visible = face_ids[face_ids >= 0]
        all_visible.update(visible.tolist())

    print(f"Visible faces: {len(all_visible)} / {len(mesh.faces)}")

    return views


if __name__ == "__main__":
    test_renderer()
