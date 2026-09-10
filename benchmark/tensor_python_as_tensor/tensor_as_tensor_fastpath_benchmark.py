#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Holoscan Tensor zero-copy Python protocol import and export paths."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import cupy as cp
except ImportError:  # pragma: no cover - benchmark environment dependent
    cp = None

try:
    import torch
except ImportError as exc:  # pragma: no cover - benchmark environment dependent
    raise SystemExit(f"PyTorch is required for this benchmark: {exc}") from exc

try:
    from holoscan.core import Tensor
    from holoscan.core._core import _test_dlpack_c_exchange_api
except ImportError as exc:  # pragma: no cover - benchmark environment dependent
    raise SystemExit(f"Holoscan Python bindings are required for this benchmark: {exc}") from exc


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    device: str
    iterations: int
    repeats: int
    median_ns: float
    min_ns: float
    max_ns: float
    samples_ns: list[float]


class PythonDLPackOnlyProxy:
    """Expose only the Python DLPack protocol, hiding C exchange support."""

    __slots__ = ("_tensor",)

    def __init__(self, tensor: Any):
        self._tensor = tensor

    def __dlpack__(self, *args: Any, **kwargs: Any) -> object:
        return self._tensor.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self) -> tuple[int, int]:
        return self._tensor.__dlpack_device__()


class ArrayInterfaceOnlyProxy:
    """Expose only the NumPy array interface, hiding DLPack support."""

    __slots__ = ("_array",)

    def __init__(self, array: np.ndarray):
        self._array = array

    @property
    def __array_interface__(self) -> dict[str, Any]:
        return self._array.__array_interface__


class CudaArrayInterfaceOnlyProxy:
    """Expose only the CUDA array interface, hiding DLPack support."""

    __slots__ = ("_array",)

    def __init__(self, array: Any):
        self._array = array

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        return self._array.__cuda_array_interface__


class HoloscanPythonDLPackOnlyProxy:
    """Expose only Python DLPack export for a Holoscan Tensor."""

    __slots__ = ("_tensor",)

    def __init__(self, tensor: Tensor):
        self._tensor = tensor

    def __dlpack__(self, *args: Any, **kwargs: Any) -> object:
        return self._tensor.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self) -> tuple[int, int]:
        return self._tensor.__dlpack_device__()


class HoloscanArrayInterfaceOnlyProxy:
    """Expose only __array_interface__ export for a Holoscan Tensor."""

    __slots__ = ("_tensor",)

    def __init__(self, tensor: Tensor):
        self._tensor = tensor

    @property
    def __array_interface__(self) -> dict[str, Any]:
        return self._tensor.__array_interface__


class HoloscanCudaArrayInterfaceOnlyProxy:
    """Expose only __cuda_array_interface__ export for a Holoscan Tensor."""

    __slots__ = ("_tensor",)

    def __init__(self, tensor: Tensor):
        self._tensor = tensor

    @property
    def __cuda_array_interface__(self) -> dict[str, Any]:
        return self._tensor.__cuda_array_interface__


def torch_version_at_least(major: int, minor: int) -> bool:
    match = re.match(r"^(\d+)\.(\d+)", torch.__version__)
    if match is None:
        return False
    return (int(match.group(1)), int(match.group(2))) >= (major, minor)


def parse_shape(value: str) -> tuple[int, ...]:
    shape = tuple(int(item) for item in value.replace("x", ",").split(",") if item)
    if not shape or any(dim <= 0 for dim in shape):
        raise argparse.ArgumentTypeError("shape must contain positive dimensions")
    return shape


def device_list(selection: str) -> list[torch.device]:
    cuda_device = None
    if torch.cuda.is_available():
        try:
            cuda_device = torch.cuda.current_device()
        except (RuntimeError, torch.cuda.DeferredCudaCallError):
            # Heterogeneous Tegra systems can expose a CUDA ordinal that the
            # installed PyTorch build cannot initialize. Keep the automatic
            # benchmark useful for CPU coverage on those systems.
            cuda_device = None

    if selection == "auto":
        devices = [torch.device("cpu")]
        if cuda_device is not None:
            devices.append(torch.device("cuda", cuda_device))
        return devices
    if selection == "cuda":
        if cuda_device is None:
            raise SystemExit("CUDA was requested but PyTorch could not initialize it")
        return [torch.device("cuda", cuda_device)]
    return [torch.device("cpu")]


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_source_tensor(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    numel = num_elements(shape)
    return torch.arange(numel, dtype=torch.float32, device=device).reshape(shape)


def make_source_array(shape: tuple[int, ...]) -> np.ndarray:
    return np.arange(num_elements(shape), dtype=np.float32).reshape(shape)


def make_source_cuda_array(shape: tuple[int, ...], device: torch.device) -> Any:
    if cp is None:
        raise SystemExit("CuPy is required for --device cuda array-interface benchmarks") from None

    with cp.cuda.Device(device.index or 0):
        return cp.arange(num_elements(shape), dtype=cp.float32).reshape(shape)


def num_elements(shape: tuple[int, ...]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def validate_zero_copy_pointer(expected_pointer: int, holoscan_tensor: Tensor, source: str) -> None:
    if holoscan_tensor.data != expected_pointer:
        raise RuntimeError(f"Holoscan tensor does not share the {source} pointer")


def validate_export_pointer(expected_pointer: int, actual_pointer: int, protocol: str) -> None:
    if actual_pointer != expected_pointer:
        raise RuntimeError(f"{protocol} export does not share the Holoscan tensor pointer")


def validate_zero_copy(torch_tensor: torch.Tensor, holoscan_tensor: Tensor) -> None:
    validate_zero_copy_pointer(torch_tensor.data_ptr(), holoscan_tensor, "PyTorch tensor")


def ns_per_iteration(elapsed_ns: int, iterations: int) -> float:
    return elapsed_ns / iterations if iterations else 0.0


def time_loop(operation: Callable[[], int], iterations: int) -> int:
    start = time.perf_counter_ns()
    for _ in range(iterations):
        data_pointer = operation()
        if data_pointer == 0:
            raise RuntimeError("unexpected null data pointer")
    return time.perf_counter_ns() - start


def measure(
    name: str,
    device: torch.device,
    iterations: int,
    repeats: int,
    warmup: int,
    operation: Callable[[], int],
) -> BenchmarkResult:
    if warmup:
        synchronize(device)
        time_loop(operation, warmup)
        synchronize(device)

    samples = []
    for _ in range(repeats):
        synchronize(device)
        elapsed_ns = time_loop(operation, iterations)
        synchronize(device)
        samples.append(ns_per_iteration(elapsed_ns, iterations))

    return BenchmarkResult(
        name=name,
        device=str(device),
        iterations=iterations,
        repeats=repeats,
        median_ns=statistics.median(samples),
        min_ns=min(samples),
        max_ns=max(samples),
        samples_ns=samples,
    )


def benchmark_device(
    device: torch.device, shape: tuple[int, ...], iterations: int, repeats: int, warmup: int
) -> list[BenchmarkResult]:
    source = make_source_tensor(shape, device)
    dlpack_proxy = PythonDLPackOnlyProxy(source)

    fast_tensor = Tensor.as_tensor(source)
    fallback_tensor = Tensor.as_tensor(dlpack_proxy)
    validate_zero_copy(source, fast_tensor)
    validate_zero_copy(source, fallback_tensor)
    dlpack_export_proxy = HoloscanPythonDLPackOnlyProxy(fast_tensor)
    validate_export_pointer(
        fast_tensor.data, torch.from_dlpack(dlpack_export_proxy).data_ptr(), "Python DLPack"
    )
    validate_export_pointer(
        fast_tensor.data,
        int(_test_dlpack_c_exchange_api(fast_tensor)["data"]),
        "DLPack C exchange",
    )

    cases: list[tuple[str, Callable[[], int]]] = [
        (
            "as_tensor_torch_c_exchange",
            lambda: Tensor.as_tensor(source).data,
        ),
        (
            "as_tensor_torch_dlpack_fallback",
            lambda: Tensor.as_tensor(dlpack_proxy).data,
        ),
        (
            "export_holoscan_c_exchange_helper",
            lambda: int(_test_dlpack_c_exchange_api(fast_tensor)["data"]),
        ),
        (
            "export_python_dlpack_to_torch",
            lambda: torch.from_dlpack(dlpack_export_proxy).data_ptr(),
        ),
    ]

    if device.type == "cpu":
        array = make_source_array(shape)
        array_proxy = ArrayInterfaceOnlyProxy(array)
        validate_zero_copy_pointer(array.ctypes.data, Tensor.as_tensor(array_proxy), "NumPy array")
        host_tensor = Tensor.as_tensor(array_proxy)
        host_export_proxy = HoloscanArrayInterfaceOnlyProxy(host_tensor)
        validate_export_pointer(
            host_tensor.data, np.asarray(host_export_proxy).ctypes.data, "NumPy array interface"
        )
        cases.append(
            (
                "as_tensor_numpy_array_interface",
                lambda: Tensor.as_tensor(array_proxy).data,
            )
        )
        cases.append(
            (
                "export_numpy_array_interface",
                lambda: np.asarray(host_export_proxy).ctypes.data,
            )
        )
    else:
        torch_cuda_array_proxy = CudaArrayInterfaceOnlyProxy(source)
        validate_zero_copy_pointer(
            source.data_ptr(), Tensor.as_tensor(torch_cuda_array_proxy), "PyTorch CUDA array"
        )
        cuda_array = make_source_cuda_array(shape, device)
        cuda_dlpack_tensor = Tensor.as_tensor(cuda_array)
        cuda_array_proxy = CudaArrayInterfaceOnlyProxy(cuda_array)
        validate_zero_copy_pointer(cuda_array.data.ptr, cuda_dlpack_tensor, "CuPy array")
        validate_zero_copy_pointer(
            cuda_array.data.ptr, Tensor.as_tensor(cuda_array_proxy), "CuPy array"
        )
        cuda_export_proxy = HoloscanCudaArrayInterfaceOnlyProxy(fast_tensor)
        validate_export_pointer(
            fast_tensor.data, cp.asarray(cuda_export_proxy).data.ptr, "CuPy CUDA array interface"
        )
        cases.append(
            (
                "as_tensor_torch_cuda_array_interface",
                lambda: Tensor.as_tensor(torch_cuda_array_proxy).data,
            )
        )
        cases.append(
            (
                "as_tensor_cupy_dlpack",
                lambda: Tensor.as_tensor(cuda_array).data,
            )
        )
        cases.append(
            (
                "as_tensor_cupy_cuda_array_interface",
                lambda: Tensor.as_tensor(cuda_array_proxy).data,
            )
        )
        cases.append(
            (
                "export_cupy_cuda_array_interface",
                lambda: cp.asarray(cuda_export_proxy).data.ptr,
            )
        )

    return [
        measure(
            name,
            device,
            iterations,
            repeats,
            warmup,
            operation,
        )
        for name, operation in cases
    ]


def print_results(results: list[BenchmarkResult]) -> None:
    print(
        f"{'benchmark':40} {'device':8} {'median ns/op':>14} {'min ns/op':>12} "
        f"{'max ns/op':>12} {'iters':>8}"
    )
    print("-" * 100)
    for result in results:
        print(
            f"{result.name:40} {result.device:8} {result.median_ns:14.1f} "
            f"{result.min_ns:12.1f} {result.max_ns:12.1f} {result.iterations:8d}"
        )

    by_device = {}
    for result in results:
        by_device.setdefault(result.device, {})[result.name] = result
    for device, device_results in by_device.items():
        fast = device_results.get("as_tensor_torch_c_exchange")
        fallback = device_results.get("as_tensor_torch_dlpack_fallback")
        if fast is not None and fallback is not None:
            speedup = fallback.median_ns / fast.median_ns
            print(f"{device}: {speedup:.1f}x faster than PyTorch Python DLPack fallback")


def write_benchmark_json(results: list[BenchmarkResult], path: str) -> None:
    payload = {
        "context": {
            "benchmark": "tensor_python_protocols",
            "torch_version": torch.__version__,
            "cupy_version": cp.__version__ if cp is not None else None,
        },
        "benchmarks": [result.__dict__ for result in results],
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--shape", type=parse_shape, default=(1024,), help="shape, e.g. 1024 or 32x32"
    )
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--benchmark_out", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--benchmark_out_format", default=None, help=argparse.SUPPRESS)
    args, extra_args = parser.parse_known_args()

    if extra_args:
        raise SystemExit(f"unsupported benchmark arguments: {' '.join(extra_args)}")
    if args.benchmark_out_format not in (None, "json"):
        raise SystemExit("--benchmark_out_format only supports json")
    if args.iterations <= 0 or args.warmup < 0 or args.repeats <= 0:
        raise SystemExit("iterations and repeats must be positive; warmup must be non-negative")
    if not torch_version_at_least(2, 10):
        raise SystemExit("PyTorch 2.10.0 or newer is required for the DLPack C exchange API")
    if not hasattr(torch.Tensor, "__dlpack_c_exchange_api__"):
        raise SystemExit("PyTorch Tensor does not expose __dlpack_c_exchange_api__")

    results = []
    for device in device_list(args.device):
        results.extend(
            benchmark_device(device, args.shape, args.iterations, args.repeats, args.warmup)
        )

    if args.json:
        print(json.dumps([result.__dict__ for result in results], indent=2))
    else:
        print_results(results)
    if args.benchmark_out is not None:
        write_benchmark_json(results, args.benchmark_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
