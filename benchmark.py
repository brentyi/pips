"""Benchmarking script for forward pass times.

Requires an extra dependency for generating command-line interfaces:

    pip install tyro

To print help messages:

    python benchmark.py --help
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as onp
import torch
import tyro

from nets.pips import Pips


class Timer:
    """Timing helper.

    Usage:
        with Timer() as t:
            ... # Do something.
        print(t.elapsed) # seconds
    """

    elapsed: float

    def __enter__(self) -> Timer:
        self._start = time.time()
        return self

    def __exit__(self, *_unused_args):
        self.elapsed = time.time() - self._start


@dataclass(frozen=True)
class BenchmarkConfig:
    b: int = 1
    n: int = 256
    h: int = 360
    w: int = 640
    iters: int = 6
    num_forward_trials: int = 20


def benchmark(config: BenchmarkConfig, stride: int = 4, iters: int = 6) -> None:
    print(config)
    print(">>>>")

    # Set up model.
    model = Pips(stride=stride).cuda()
    model.eval()

    # Set up inputs.
    rgbs = torch.zeros(
        (config.b, model.S, 3, config.h, config.w), dtype=torch.uint8
    ).cuda()
    xys = torch.zeros((config.b, config.n, 2), dtype=torch.float32).cuda()

    # Run network!
    print("Forward pass...")
    runtimes = []
    torch.cuda.synchronize()
    for _ in range(config.num_forward_trials):
        with torch.inference_mode(), Timer() as t:
            model.forward(xys=xys, rgbs=rgbs, iters=iters)
            torch.cuda.synchronize()
        runtimes.append(t.elapsed)

    mean = onp.mean(runtimes)
    std_err = onp.std(runtimes, ddof=1) / onp.sqrt(len(runtimes))
    print(f"\t{mean:.05f} ± {std_err:.05f} seconds")


if __name__ == "__main__":
    tyro.cli(benchmark, description=__doc__)
