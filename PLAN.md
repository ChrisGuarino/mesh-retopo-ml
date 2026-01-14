# ML-Integrated Mesh Retopology Pipeline

## Overview

Two-phase implementation:
1. **Phase 1**: Feature-aware adaptive simplification (ML predicts per-face importance)
2. **Phase 2**: Full quad remeshing network (ML generates optimal quad topology)

---

## Phase 1: Feature-Aware Adaptive Simplification

### Goal
Train a model to predict per-face importance scores, then use those scores to guide decimation (preserve high-importance faces, aggressively simplify low-importance regions).

### New Files to Create

```
mesh-retopo-ml/
├── mesh_processor.py          # Existing - will extend
├── feature_extractor.py       # NEW - extract curvature, geometry features
├── importance_model.py        # NEW - neural network for importance prediction
├── dataset_generator.py       # NEW - create synthetic training pairs
├── train.py                   # NEW - training script
├── adaptive_simplify.py       # NEW - ML-guided simplification
└── models/                    # NEW - saved model checkpoints
```

### Implementation Steps

#### Step 1: Feature Extractor (`feature_extractor.py`)
Extract per-face features from any mesh:
- Gaussian curvature (via trimesh)
- Mean curvature (via trimesh)
- Face area (normalized)
- Aspect ratio (min/max edge length)
- Max dihedral angle to neighbors
- Vertex valence statistics
- Normal variation with neighbors

Output: `(N_faces, ~11)` feature matrix

#### Step 2: Dataset Generator (`dataset_generator.py`)
Generate synthetic training pairs (mixed organic + hard surface):

**Organic shapes:**
- Icospheres with varying subdivision
- Tori, cylinders with smooth caps
- Noise-perturbed surfaces (Perlin noise displacement)

**Hard surface shapes:**
- Cubes, boxes with beveled edges
- Cylinders with sharp edges
- Boolean combinations (CSG operations)

**Pipeline:**
1. Generate mesh
2. Compute features for each face
3. Run multiple simplification methods at various target ratios
4. Label faces: importance = 1.0 if preserved, 0.0 if collapsed
5. Save as `.npz` files

#### Step 3: Importance Model (`importance_model.py`)
Simple MLP (chosen for faster iteration):
- Input: per-face features (11 floats)
- Output: importance score (0.0 - 1.0)
- Architecture: `11 -> 64 -> 32 -> 1` with ReLU + sigmoid
- Can upgrade to GCN later if needed

#### Step 4: Training Script (`train.py`)
- Load dataset
- Train model with BCE loss
- Save best checkpoint

#### Step 5: Adaptive Simplification (`adaptive_simplify.py`)
- Load trained model
- Predict importance for each face
- Sort faces by importance
- Use pyfqmr with face weighting OR iterative edge collapse prioritized by importance

### Dependencies to Add
```
torch
```
(torch-geometric not needed for MLP approach)

---

## Phase 2: Quad Remeshing Network (Future)

### Goal
Given a triangulated mesh, predict optimal quad layout with clean edge flow.

### Approach
- Graph Neural Network operating on mesh connectivity
- Predicts edge directions / quad patch boundaries
- Post-process to generate actual quad mesh

### Key Components
1. **Mesh encoder**: GCN layers on face graph
2. **Edge classifier**: Predict which edges form quad boundaries
3. **Quad generator**: Convert predictions to actual quad mesh

This phase is more research-oriented and will build on Phase 1's infrastructure.

---

## Files to Modify

| File | Changes |
|------|---------|
| `mesh_processor.py` | Add `extract_features()` method, add `ml_simplify()` method |
| `requirements.txt` | Add `torch`, `torch-geometric` |

---

## Verification Plan

1. **Feature extraction**: Run on test_mesh.ply, visualize curvature as vertex colors
2. **Dataset generation**: Generate 100 synthetic mesh pairs, inspect labels
3. **Training**: Train for 50 epochs, monitor loss convergence
4. **Evaluation**: Compare ML-guided simplification vs vanilla QEM on:
   - Visual quality (render comparison)
   - Feature preservation (curvature error)
   - Edge flow quality

### Test Commands
```bash
# Generate dataset
python dataset_generator.py --num_meshes 100 --output data/

# Train model
python train.py --data data/ --epochs 50 --output models/importance_v1.pt

# Run adaptive simplification
python adaptive_simplify.py --mesh test_mesh.ply --model models/importance_v1.pt --target 1000
```

---

## Summary

| Phase | Complexity | Deliverable |
|-------|------------|-------------|
| 1 | Medium | ML-guided decimation preserving important features |
| 2 | High | Full quad remeshing with learned edge flow |

Phase 1 is achievable in a focused session. Phase 2 will build on that foundation.
