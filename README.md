# Mesh Retopology ML

ML-powered mesh simplification that learns which faces to preserve based on geometric and visual features.

> **Note:** This project was built with AI assistance (Claude).

## Overview

Traditional mesh simplification (like Quadric Error Metrics) treats all faces equally. This project trains neural networks to predict **face importance** based on geometric features and multi-view CNN analysis, enabling smarter decimation that preserves important regions.

```
Input Mesh → Feature Extraction → ML Importance Prediction → Adaptive Simplification → Output
```

## Features

- **11 geometric features** extracted per face (curvature, area, angles, topology)
- **MLP-based importance prediction** (~3,600 parameters, trains in seconds)
- **Multi-view CNN model** (optional) - Uses pretrained ResNet to understand global context
- **Geometry-based labeling** - Importance based on curvature, edges, and surface detail
- **Thingi10K dataset support** for training on real-world meshes
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

### Option 1: Simple MLP Model (Recommended for laptops)

```bash
# Generate training data from synthetic primitives
python dataset_generator.py --num_meshes 100 --output data/

# Or with Thingi10K (70% real meshes, 30% synthetic)
python dataset_generator.py --num_meshes 200 --thingi10k datasets/thingi10k --output data/

# Train the model
python train.py --data data/training_data.npz --epochs 50 --output models/importance.pt
```

### Option 2: Multi-View CNN Model (Requires more RAM/GPU)

```bash
# Preprocess meshes (extract CNN features + saliency labels)
python train_multiview.py preprocess --input datasets/thingi10k --output data/multiview_data.npz --max_meshes 50

# Train the multi-view model
python train_multiview.py train --data data/multiview_data.npz --output models/multiview_importance.pt
```

### Run Simplification

```bash
# Simplify a mesh
python adaptive_simplify.py --mesh your_mesh.stl --model models/importance.pt --target 1000 --output simplified.ply

# Compare ML vs standard simplification
python adaptive_simplify.py --mesh your_mesh.stl --model models/importance.pt --target 500 --compare

# Visualize importance as vertex colors
python adaptive_simplify.py --mesh your_mesh.stl --model models/importance.pt --target 500 --visualize --output importance.ply
```

## Project Structure

```
mesh-retopo-ml/
├── feature_extractor.py    # Extract 11 geometric features per face
├── dataset_generator.py    # Generate training pairs (synthetic + Thingi10K)
├── importance_model.py     # MLP neural network definition
├── train.py                # Training script for MLP model
├── adaptive_simplify.py    # ML-guided mesh simplification
├── multiview_renderer.py   # Multi-view mesh rendering
├── multiview_cnn.py        # CNN feature extractor + model
├── train_multiview.py      # Training script for multi-view CNN
├── mesh_processor.py       # Basic mesh utilities
└── requirements.txt        # Dependencies
```

## Two Model Architectures

### 1. Simple MLP (Default)

Fast, lightweight model that uses only geometric features:
- Input: 11 geometric features per face
- Architecture: `11 → 64 → 32 → 16 → 1`
- Output: Importance score (0-1)
- Training: ~seconds on CPU

**Best for:** Quick iterations, laptops, when geometric features are sufficient.

### 2. Multi-View CNN (Advanced)

Uses pretrained ResNet18 to extract visual features from rendered views:
- Renders mesh from 4-8 viewpoints
- Extracts CNN features from each view
- Projects features back to mesh faces
- Combines with geometric features
- Predicts importance with MLP head

**Best for:** Understanding global context, semantic importance, when you have GPU/RAM.

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

### Importance Labeling (Geometry-Based)

Faces are labeled as important based on:
- **High curvature** (50%) - Peaks, valleys, detailed regions
- **Normal variation** (25%) - Surface detail indicator
- **Sharp edges** (20%) - Dihedral angles > 30°
- **Flatness penalty** (5%) - Large flat faces are less important

### Training Pipeline

1. **Load meshes** - From Thingi10K and/or synthetic primitives
2. **Extract features** - Geometric features for each face
3. **Compute labels** - Geometry-based importance scores
4. **Train model** - Learn mapping from features → importance

### Inference

1. Load mesh and trained model
2. Extract features for each face
3. Predict importance scores (0-1)
4. Simplify while preserving high-importance faces

## Dataset Options

| Source | Command | Description |
|--------|---------|-------------|
| Synthetic only | `--num_meshes 100` | Fast, limited variety |
| Thingi10K | `--thingi10k PATH` | Real-world 3D printable objects |
| Mixed (default) | `--thingi10k PATH --thingi10k_ratio 0.7` | 70% real, 30% synthetic |

## Memory Optimization (Multi-View CNN)

If you run out of memory with the multi-view model:

```bash
# Reduce image size and views
python train_multiview.py preprocess --input datasets/thingi10k --image_size 64 --n_views 2 --max_meshes 30

# Limit mesh complexity
python train_multiview.py preprocess --input datasets/thingi10k --max_faces 20000
```

## Supported Formats

Any format trimesh supports: `.stl`, `.obj`, `.ply`, `.off`, `.gltf`, `.glb`, and more.

## Dependencies

- trimesh - Mesh loading/processing
- numpy - Numerical operations
- torch - Neural network
- torchvision - Pretrained CNN models (multi-view)
- scipy - Scientific computing
- matplotlib - Visualization
- pyfqmr - Fast quadric mesh simplification
- pyrender - GPU mesh rendering (optional)
- opencv-python - Image processing (multi-view)

## License

MIT
