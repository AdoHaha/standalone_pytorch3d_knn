# Standalone PyTorch3D KNN

A standalone library for K-Nearest Neighbors (KNN) operations on 3D point clouds, extracted from [PyTorch3D](https://github.com/facebookresearch/pytorch3d). This package provides the efficient `knn_points` function with support for both CPU and CUDA acceleration, without requiring the installation of the entire PyTorch3D library.

## Installation

### From GitHub
You can install the package directly from GitHub:
```bash
pip install git+https://github.com/AdoHaha/standalone_pytorch3d_knn
```

### From Source
Clone the repository and install:
```bash
git clone https://github.com/AdoHaha/standalone_pytorch3d_knn
cd standalone_pytorch3d_knn
pip install .
```

**Requirements:**
- Python 3.7+
- PyTorch
- C++ compiler (for CPU build)
- CUDA toolkit (nvcc) for GPU support

## Usage

The API is identical to PyTorch3D's `knn_points`, but imported from `knn_standalone`.

```python
import torch
from knn_standalone import knn_points

# Example data
p1 = torch.randn(1, 100, 3, device="cuda" if torch.cuda.is_available() else "cpu")
p2 = torch.randn(1, 50, 3, device="cuda" if torch.cuda.is_available() else "cpu")

# Find the 3 nearest neighbors in p2 for each point in p1
dists, idx, nn = knn_points(p1, p2, K=3, return_nn=True)

print(f"Distances shape: {dists.shape}")
print(f"Indices shape: {idx.shape}")
print(f"Nearest neighbors shape: {nn.shape}")
```

## Configuration

The package automatically detects if CUDA is available. To force a CPU-only build even if CUDA is present, set the environment variable `P3D_KNN_FORCE_CPU=1`:

```bash
P3D_KNN_FORCE_CPU=1 pip install .
```

## Testing & Benchmarks

Run the tests to verify installation:
```bash
python -m unittest knn_standalone.tests.test_knn_standalone
```

Run benchmarks:
```bash
# CPU benchmark
python -m knn_standalone.bench_knn --device cpu

# GPU benchmark (requires CUDA)
python -m knn_standalone.bench_knn --device cuda
```

## License

This code is extracted from PyTorch3D and is released under the [BSD License](LICENSE).