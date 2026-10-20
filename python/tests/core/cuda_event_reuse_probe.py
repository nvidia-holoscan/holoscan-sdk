#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assert retired imported-Tensor CUDA events are reused instead of created for every import."""

import argparse
import ctypes

try:
    import cupy as cp
except ImportError:
    raise SystemExit(77) from None

import holoscan as hs
from holoscan._core import _test_drain_tensor_releases

ROUNDS = 32
CONTEXT_ROUNDS = 6


class DeviceByte:
    """Expose one byte of a context's device memory through the CUDA array interface."""

    def __init__(self, pointer):
        self.__cuda_array_interface__ = {
            "shape": (1,),
            "typestr": "|u1",
            "data": (pointer, False),
            "version": 3,
        }


def import_and_export(stream):
    # The import records a readiness event and the CuPy view registers a consumer event.
    with stream:
        source = cp.arange(64, dtype=cp.float32)
        tensor = hs.Tensor.as_tensor(source, stream=stream.ptr)
        view = cp.from_dlpack(tensor)
        assert int(view[3]) == 3
    del view, tensor, source
    _test_drain_tensor_releases(5.0)


def driver_api():
    """Bind the CUDA driver calls that create, use, and destroy ordinary contexts."""
    driver = ctypes.CDLL("libcuda.so.1")
    names = (
        "cuInit",
        "cuDeviceGet",
        "cuCtxCreate_v2",
        "cuCtxDestroy_v2",
        "cuCtxPushCurrent_v2",
        "cuCtxPopCurrent_v2",
        "cuMemAlloc_v2",
        "cuMemFree_v2",
    )
    calls = {name: getattr(driver, name) for name in names}
    for call in calls.values():
        call.restype = ctypes.c_int

    def check(name, *arguments):
        result = calls[name](*arguments)
        if result != 0:
            raise RuntimeError(f"{name} failed with CUresult {result}")

    return check


def cycle_live_contexts(count, creations, destructions):
    """Import from each of `count` live contexts per round; return outstanding events per round."""
    check = driver_api()
    check("cuInit", 0)
    device = ctypes.c_int()
    check("cuDeviceGet", ctypes.byref(device), 0)
    popped = ctypes.c_void_p()
    contexts = []
    try:
        for _ in range(count):
            context = ctypes.c_void_p()
            check("cuCtxCreate_v2", ctypes.byref(context), 0, device)
            pointer = ctypes.c_uint64()
            try:
                check("cuMemAlloc_v2", ctypes.byref(pointer), ctypes.c_size_t(1))
            finally:
                check("cuCtxPopCurrent_v2", ctypes.byref(popped))
            contexts.append((context, pointer))
        outstanding = []
        for _ in range(CONTEXT_ROUNDS):
            for context, pointer in contexts:
                check("cuCtxPushCurrent_v2", context)
                try:
                    tensor = hs.Tensor.as_tensor(DeviceByte(pointer.value))
                    del tensor
                finally:
                    check("cuCtxPopCurrent_v2", ctypes.byref(popped))
                _test_drain_tensor_releases(5.0)
            outstanding.append(creations() - destructions())
        return outstanding
    finally:
        _test_drain_tensor_releases(5.0)
        for context, pointer in contexts:
            check("cuCtxPushCurrent_v2", context)
            check("cuMemFree_v2", pointer)
            check("cuCtxPopCurrent_v2", ctypes.byref(popped))
            check("cuCtxDestroy_v2", context)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cudart", required=True)
    parser.add_argument(
        "--live-contexts",
        type=int,
        default=0,
        help="instead, import from this many CUDA contexts that all stay alive",
    )
    args = parser.parse_args()
    try:
        if not cp.cuda.runtime.getDeviceCount():
            return 77
    except RuntimeError:
        return 77
    # The preload shim forwards to the next definition, which must be globally visible.
    ctypes.CDLL(args.cudart, mode=ctypes.RTLD_GLOBAL)
    shim = ctypes.CDLL(None)
    creations = shim.hsdk_test_event_creations
    creations.restype = ctypes.c_uint64
    destructions = shim.hsdk_test_event_destructions
    destructions.restype = ctypes.c_uint64
    create = shim.cudaEventCreateWithFlags
    create.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint)
    destroy = shim.cudaEventDestroy
    destroy.argtypes = (ctypes.c_void_p,)
    cp.cuda.runtime.free(0)

    before = (creations(), destructions())
    event = ctypes.c_void_p()
    assert create(ctypes.byref(event), 2) == 0
    assert destroy(event) == 0
    assert (creations(), destructions()) == (before[0] + 1, before[1] + 1), (
        "CUDA event counter did not observe its positive control"
    )

    if args.live_contexts:
        # Every context stays alive, so the pool must bound its events without dropping any.
        outstanding = cycle_live_contexts(args.live_contexts, creations, destructions)
        assert len(set(outstanding[1:])) == 1, (
            f"outstanding CUDA events kept growing across {args.live_contexts} live contexts: "
            f"{outstanding}"
        )
        print(f"{args.live_contexts} live contexts: outstanding events per round {outstanding}")
        return 0

    stream = cp.cuda.Stream(non_blocking=True)
    for _ in range(ROUNDS):
        import_and_export(stream)
    warm = (creations(), destructions())
    for _ in range(ROUNDS):
        import_and_export(stream)
    steady = (creations(), destructions())
    assert steady == warm, (
        f"{ROUNDS} retired imports created {steady[0] - warm[0]} and destroyed "
        f"{steady[1] - warm[1]} CUDA events after warmup"
    )
    print(f"{ROUNDS} retired imports reused their CUDA events; positive control passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
