import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

# Check if CUDA is available and if we want to force CPU build
FORCE_CPU = os.getenv("P3D_KNN_FORCE_CPU", "0") == "1"
WITH_CUDA = (not FORCE_CPU) and torch.cuda.is_available() and (CUDA_HOME is not None)

sources = [
    "knn_ext.cpp",
    "csrc/knn/knn_cpu.cpp",
]

extra_compile_args = {"cxx": ["-std=c++17", "-O3"]}
define_macros = []

if WITH_CUDA:
    sources.append("csrc/knn/knn.cu")
    define_macros.append(("WITH_CUDA", None))
    extra_compile_args["nvcc"] = [
        "-std=c++17",
        "-O3",
        "-DCUDA_HAS_FP16=1",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]

# Select extension type
Extension = CUDAExtension if WITH_CUDA else CppExtension

setup(
    name="knn_standalone",
    version="0.1.0",
    author="Meta Platforms, Inc.",
    description="Standalone KNN from PyTorch3D",
    packages=["knn_standalone"],
    package_dir={"knn_standalone": "."},
    ext_modules=[
        Extension(
            name="knn_standalone.p3d_knn_ext",
            sources=sources,
            include_dirs=[os.path.abspath("csrc")],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)
