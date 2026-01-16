# Mesh Retopology ML

ML-powered mesh simplification that learns which faces to preserve based on geometric features.

> **Note:** This project was built with AI assistance (Claude).

## Overview

Traditional mesh simplification (like Quadric Error Metrics) treats all faces equally. This project trains a neural network to predict **face importance** based on geometric features, enabling smarter decimation that preserves important regions.

```
Input Mesh → Feature Extraction → ML Importance Prediction → Adaptive Simplification → Output
```

## Features

- **11 geometric features** extracted per face (curvature, area, angles, topology)
- **MLP-based importance prediction** (~3,600 parameters, trains in seconds)
- **Synthetic dataset generation** (organic + hard surface primitives)
- **Comparison tools** to evaluate ML-guided vs standard simplification

## Installation

```bash
git clone https://github.com/ChrisGuarino/mesh-retopo-ml.git
cd mesh-retopo-ml
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Generate training data
python dataset_generator.py --num_meshes 100 --output data/

# 2. Train the model
python train.py --data data/training_data.npz --epochs 50 --output models/importance_v1.pt

# 3. Simplify a mesh
python adaptive_simplify.py --mesh your_mesh.stl --model models/importance_v1.pt --target 1000 --output simplified.ply

# 4. Compare ML vs standard simplification
python adaptive_simplify.py --mesh your_mesh.stl --model models/importance_v1.pt --target 500 --compare
```

## Project Structure

```
mesh-retopo-ml/
├── feature_extractor.py    # Extract 11 geometric features per face
├── dataset_generator.py    # Generate synthetic training pairs
├── importance_model.py     # MLP neural network definition
├── train.py                # Training script
├── adaptive_simplify.py    # ML-guided mesh simplification
├── mesh_processor.py       # Basic mesh utilities
├── PLAN.md                 # Development roadmap
└── requirements.txt        # Dependencies
```

## How It Works

### Feature Extraction

For each face, we compute:

| Feature | Description |
|---------|-------------|
| Gaussian curvature | Surface type (peaks, valleys, saddles) |
| Mean curvature | Overall surface bending |
| Face area | Normalized relative to mesh |
| Aspect ratio | Triangle shape quality |
| Dihedral angles | Sharp edge detection |
| Vertex valence | Mesh topology quality |
| Normal variation | Feature edge indicator |
| Edge length variance | Triangle regularity |
| Face compactness | How close to equilateral |
| Local density | Detail concentration |

### Training Pipeline

1. **Generate meshes** - Synthetic primitives (spheres, tori, boxes, cylinders)
2. **Simplify each mesh** - Using standard QEM at various ratios
3. **Label faces** - Importance based on proximity to simplified mesh
4. **Train MLP** - Learn mapping from features → importance score

### Inference

1. Load mesh and trained model
2. Extract features for each face
3. Predict importance scores (0-1)
4. Simplify while preserving high-importance faces

## Limitations

The current system uses **geometric features only**. It doesn't understand:
- Semantic importance (faces, hands, joints)
- Animation requirements (edge loops for deformation)
- Artist intent

See [PLAN.md](PLAN.md) for future improvements including:
- ViT-based semantic importance
- Skeleton-aware simplification
- Quad remeshing with GNNs

## Supported Formats

Any format trimesh supports: `.stl`, `.obj`, `.ply`, `.off`, `.gltf`, `.glb`, and more.

## Dependencies

- trimesh - Mesh loading/processing
- numpy - Numerical operations
- torch - Neural network
- scipy - Scientific computing
- matplotlib - Visualization
- pyfqmr - Fast quadric mesh simplification

## License

MIT
