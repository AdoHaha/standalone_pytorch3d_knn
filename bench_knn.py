#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import time

import torch

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from knn_standalone import knn_points


def naive_knn_points(p1, p2, K: int, norm: int = 2):
    N, P1, D = p1.shape
    P2 = p2.shape[1]
    dists = torch.zeros((N, P1, K), dtype=torch.float32, device=p1.device)
    idx = torch.zeros((N, P1, K), dtype=torch.int64, device=p1.device)
    for n in range(N):
        for i1 in range(P1):
            diff = p1[n, i1] - p2[n]
            if norm == 2:
                dist = (diff * diff).sum(-1)
            elif norm == 1:
                dist = diff.abs().sum(-1)
            else:
                raise ValueError("Support for 1 or 2 norm.")
            k = min(P2, K)
            srt_dd, srt_idx = dist.sort()
            dists[n, i1, :k] = srt_dd[:k]
            idx[n, i1, :k] = srt_idx[:k]
    return dists, idx


def cdist_knn_points(p1, p2, K: int, norm: int = 2):
    dists = torch.cdist(p1, p2, p=norm)
    dists, idx = torch.topk(dists, k=K, dim=2, largest=False, sorted=True)
    if norm == 2:
        dists = dists * dists
    return dists, idx


def _time_fn(fn, runs: int, warmup: int, device: torch.device):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()
    return (end - start) / runs


def main():
    parser = argparse.ArgumentParser(description="Benchmark knn_points vs naive.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--N", type=int, default=1)
    parser.add_argument("--P1", type=int, default=128)
    parser.add_argument("--P2", type=int, default=128)
    parser.add_argument("--D", type=int, default=3)
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--norm", type=int, default=2, choices=[1, 2])
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--skip-naive", action="store_true")
    parser.add_argument("--include-cdist", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")

    p1 = torch.randn(args.N, args.P1, args.D, device=device)
    p2 = torch.randn(args.N, args.P2, args.D, device=device)

    def run_knn():
        knn_points(p1, p2, K=args.K, norm=args.norm)

    timings = []
    knn_time = _time_fn(run_knn, args.runs, args.warmup, device)
    timings.append(("knn_points", knn_time))

    if not args.skip_naive:
        def run_naive():
            naive_knn_points(p1, p2, K=args.K, norm=args.norm)

        naive_time = _time_fn(run_naive, args.runs, args.warmup, device)
        timings.append(("naive_python", naive_time))

    if args.include_cdist:
        def run_cdist():
            cdist_knn_points(p1, p2, K=args.K, norm=args.norm)

        cdist_time = _time_fn(run_cdist, args.runs, args.warmup, device)
        timings.append(("torch_cdist_topk", cdist_time))

    print(
        f"device={args.device} N={args.N} P1={args.P1} P2={args.P2} "
        f"D={args.D} K={args.K} norm={args.norm}"
    )
    for name, t in timings:
        print(f"{name:>18s}: {t * 1e3:.3f} ms/iter")


if __name__ == "__main__":
    main()
