# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from itertools import product

import torch
from knn_standalone import knn_gather, knn_points
from knn_standalone import _C as knn_C


def _get_random_cuda_device() -> str:
    num_devices = torch.cuda.device_count()
    device_id = (
        torch.randint(high=num_devices, size=(1,)).item() if num_devices > 1 else 0
    )
    return f"cuda:{device_id}"


class TestKNNStandalone(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        torch.manual_seed(1)

    @staticmethod
    def _knn_points_naive(p1, p2, lengths1, lengths2, K: int, norm: int = 2):
        """
        Naive PyTorch implementation of K-Nearest Neighbors.
        Returns always sorted results.
        """
        N, P1, D = p1.shape
        _N, P2, _D = p2.shape

        assert N == _N and D == _D

        if lengths1 is None:
            lengths1 = torch.full((N,), P1, dtype=torch.int64, device=p1.device)
        if lengths2 is None:
            lengths2 = torch.full((N,), P2, dtype=torch.int64, device=p1.device)

        dists = torch.zeros((N, P1, K), dtype=torch.float32, device=p1.device)
        idx = torch.zeros((N, P1, K), dtype=torch.int64, device=p1.device)

        for n in range(N):
            num1 = lengths1[n].item()
            num2 = lengths2[n].item()
            pp1 = p1[n, :num1].view(num1, 1, D)
            pp2 = p2[n, :num2].view(1, num2, D)
            diff = pp1 - pp2
            if norm == 2:
                diff = (diff * diff).sum(2)
            elif norm == 1:
                diff = diff.abs().sum(2)
            else:
                raise ValueError("No support for norm %d" % (norm))
            num2 = min(num2, K)
            for i in range(num1):
                dd = diff[i]
                srt_dd, srt_idx = dd.sort()

                dists[n, i, :num2] = srt_dd[:num2]
                idx[n, i, :num2] = srt_idx[:num2]

        return dists, idx

    @staticmethod
    def _cuda_extension_available() -> bool:
        return torch.cuda.is_available() and hasattr(knn_C, "knn_check_version")

    def _compare_knn(self, device, use_lengths):
        torch.manual_seed(2)
        N, P1, P2, D, K = 2, 7, 6, 3, 4
        norms = [1, 2]
        lengths1 = None
        lengths2 = None
        if use_lengths:
            lengths1 = torch.tensor([P1 - 1, P1], device=device)
            lengths2 = torch.tensor([P2 - 2, P2 - 1], device=device)

        for norm in norms:
            p1 = torch.randn((N, P1, D), device=device, requires_grad=True)
            p2 = torch.randn((N, P2, D), device=device, requires_grad=True)
            p1_ref = p1.clone().detach().requires_grad_(True)
            p2_ref = p2.clone().detach().requires_grad_(True)

            ref_dists, ref_idx = self._knn_points_naive(
                p1_ref, p2_ref, lengths1, lengths2, K=K, norm=norm
            )
            out = knn_points(
                p1, p2, lengths1=lengths1, lengths2=lengths2, K=K, norm=norm
            )

            self.assertTrue(torch.allclose(ref_dists, out.dists))
            self.assertTrue(torch.all(ref_idx == out.idx))

            grad_dist = torch.randn((N, P1, K), dtype=torch.float32, device=device)
            (ref_dists * grad_dist).sum().backward()
            (out.dists * grad_dist).sum().backward()

            self.assertTrue(torch.allclose(p1.grad, p1_ref.grad, atol=5e-6))
            self.assertTrue(torch.allclose(p2.grad, p2_ref.grad, atol=5e-6))

    def test_knn_vs_python_square_cpu(self):
        device = torch.device("cpu")
        self._compare_knn(device, use_lengths=False)

    def test_knn_vs_python_ragged_cpu(self):
        device = torch.device("cpu")
        self._compare_knn(device, use_lengths=True)

    def test_knn_vs_python_square_cuda(self):
        if not self._cuda_extension_available():
            self.skipTest("CUDA extension not available")
        device = torch.device(_get_random_cuda_device())
        self._compare_knn(device, use_lengths=False)

    def test_knn_vs_python_ragged_cuda(self):
        if not self._cuda_extension_available():
            self.skipTest("CUDA extension not available")
        device = torch.device(_get_random_cuda_device())
        self._compare_knn(device, use_lengths=True)

    def test_knn_gather_padding(self):
        device = torch.device("cpu")
        N, P1, P2, K, D = 2, 5, 3, 4, 3
        x = torch.rand((N, P1, D), device=device)
        y = torch.rand((N, P2, D), device=device)
        lengths2 = torch.tensor([2, 3], device=device)

        out = knn_points(x, y, lengths2=lengths2, K=K)
        y_nn = knn_gather(y, out.idx, lengths2)

        for n, p1, k in product(range(N), range(P1), range(K)):
            if k < lengths2[n]:
                self.assertTrue(torch.allclose(y_nn[n, p1, k], y[n, out.idx[n, p1, k]]))
            else:
                self.assertTrue(torch.all(y_nn[n, p1, k] == 0.0))

    def test_return_nn(self):
        device = torch.device("cpu")
        N, P1, P2, K, D = 2, 5, 6, 3, 3
        x = torch.rand((N, P1, D), device=device)
        y = torch.rand((N, P2, D), device=device)
        lengths2 = torch.tensor([P2 - 1, P2], device=device)

        out = knn_points(x, y, lengths2=lengths2, K=K, return_nn=True)
        self.assertIsNotNone(out.knn)
        self.assertTrue(torch.allclose(out.knn, knn_gather(y, out.idx, lengths2)))

    def test_invalid_norm(self):
        device = torch.device("cpu")
        N, P1, P2, K, D = 2, 5, 6, 3, 3
        x = torch.rand((N, P1, D), device=device)
        y = torch.rand((N, P2, D), device=device)
        with self.assertRaisesRegex(ValueError, "Support for 1 or 2 norm."):
            knn_points(x, y, K=K, norm=3)

    def test_knn_check_version(self):
        if not hasattr(knn_C, "knn_check_version"):
            self.skipTest("CUDA extension not available")
        for D in range(-2, 10):
            for K in range(-2, 10):
                v0 = True
                v1 = 1 <= D <= 32
                v2 = 1 <= D <= 8 and 1 <= K <= 32
                v3 = 1 <= D <= 8 and 1 <= K <= 4
                all_expected = [v0, v1, v2, v3]
                for version in range(-2, 5):
                    actual = knn_C.knn_check_version(version, D, K)
                    expected = False
                    if 0 <= version < len(all_expected):
                        expected = all_expected[version]
                    self.assertEqual(actual, expected)
