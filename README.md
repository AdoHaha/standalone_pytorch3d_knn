# Standalone PyTorch3D KNN

This folder isolates `knn_points` from PyTorch3D by building only the KNN
extension and wrapping it with the same Python API.

Usage:

```python
from knn_standalone import knn_points

dists, idx, _ = knn_points(p1, p2, K=3)
```

Notes:
- The extension builds on first import; you need a working C++ compiler.
- For GPU support, you also need CUDA/NVCC available.
- Set `P3D_KNN_FORCE_CPU=1` to skip the CUDA build and use CPU only.

Tests:

```bash
python -m unittest knn_standalone.tests.test_knn_standalone
```

Benchmark:

```bash
python -m knn_standalone.bench_knn --device cpu
python -m knn_standalone.bench_knn --device cuda --include-cdist
```
