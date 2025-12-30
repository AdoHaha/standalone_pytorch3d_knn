import os

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

_EXT = None


def _build_extension():
    this_dir = os.path.dirname(__file__)
    repo_root = os.path.abspath(os.path.join(this_dir, os.pardir))
    csrc_dir = os.path.join(repo_root, "pytorch3d", "csrc")

    sources = [
        os.path.join(this_dir, "knn_ext.cpp"),
        os.path.join(csrc_dir, "knn", "knn_cpu.cpp"),
    ]
    extra_include_paths = [csrc_dir]

    force_cpu = os.getenv("P3D_KNN_FORCE_CPU", "0") == "1"
    with_cuda = (not force_cpu) and torch.cuda.is_available() and CUDA_HOME is not None

    extra_cflags = ["-O3", "-std=c++17"]
    extra_cuda_cflags = ["-O3", "-std=c++17"]
    if with_cuda:
        sources.append(os.path.join(csrc_dir, "knn", "knn.cu"))
        extra_cflags.append("-DWITH_CUDA")
        extra_cuda_cflags.append("-DWITH_CUDA")
        extra_cuda_cflags.extend(
            [
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        )

    return load(
        name="p3d_knn_ext",
        sources=sources,
        extra_include_paths=extra_include_paths,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=with_cuda,
        verbose=False,
    )


try:
    _EXT = _build_extension()
except Exception as exc:
    raise RuntimeError(
        "Failed to build the standalone KNN extension. "
        "Check that a C++ compiler is available and CUDA/NVCC is set up "
        "if you want GPU support."
    ) from exc

knn_points_idx = _EXT.knn_points_idx
knn_points_backward = _EXT.knn_points_backward
if hasattr(_EXT, "knn_check_version"):
    knn_check_version = _EXT.knn_check_version
