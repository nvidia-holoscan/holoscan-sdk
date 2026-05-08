"""
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

import platform
import sys
from argparse import ArgumentParser

import cupy as cp

from holoscan.conditions import CountCondition, CudaStreamCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.logger import LogLevel, set_log_level
from holoscan.operators import PingTensorRxOp
from holoscan.resources import CudaGreenContext, CudaGreenContextPool, CudaStreamPool

MIN_GREEN_CONTEXT_CUDA_DRIVER_VERSION = 12040
GREEN_CONTEXT_REQUIREMENT_DOC_URL = "https://docs.nvidia.com/holoscan/sdk-user-guide/hsdk_faq.html"


def _cuda_driver_version():
    """Return ``cudaDriverGetVersion()`` as an integer (e.g. 12040 for 12.4), or None on failure."""
    try:
        return int(cp.cuda.runtime.driverGetVersion())
    except cp.cuda.runtime.CUDARuntimeError:
        return None


def _green_context_supported_by_cuda_driver():
    """Return True if the installed driver meets the minimum for Green Context (CUDA 12.4+ API)."""
    version = _cuda_driver_version()
    return version is not None and version >= MIN_GREEN_CONTEXT_CUDA_DRIVER_VERSION


def _green_context_min_sm_size_for_device_major(device_major):
    """Return a typical minimum SM block size for Green Context pools by architecture.

    This example uses the value for ``min_sm_size`` on ``CudaGreenContextPool`` together with
    ``sms_per_partition`` so partition sizes stay compatible with the driver's SM grouping.

    Args:
        device_major: CUDA compute capability major: the ``major`` field from CuPy
            ``getDeviceProperties(0)`` (same as ``cudaDeviceProp.major``):
            - 7: Volta/Turing (SM 7.x)
            - 8: Ampere (SM 8.x)
            - 9: Hopper (SM 9.x)
            - 10+: Blackwell and newer (SM 10.x+)

    Returns:
        int: Minimum SMs per block for the pool. Holoscan exposes this as ``min_sm_size``;
        GXF receives it as ``min_sm_count``.

    Note:
        For reference on some aarch64 boards when sizing partitions: Jetson Orin AGX has 16 SMs;
        Jetson Orin Nano has 8; Jetson AGX Thor has 20 (Blackwell sm_110).
    """
    if device_major == 7:
        return 2
    if device_major in (8, 9):
        return 4
    if device_major >= 10:
        return 8
    return 2


def _green_context_device_properties():
    """Return CUDA device properties needed by Green Context examples.

    Returns:
        dict | None: ``{"major": int, "sm_count": int, "min_sm_size": int}`` on success,
        otherwise ``None`` when CUDA is unavailable or properties cannot be queried.
    """
    try:
        props = cp.cuda.runtime.getDeviceProperties(0)
        major = int(props["major"])
        sm_count = int(props["multiProcessorCount"])
        return {
            "major": major,
            "sm_count": sm_count,
            "min_sm_size": _green_context_min_sm_size_for_device_major(major),
        }
    except (RuntimeError, cp.cuda.runtime.CUDARuntimeError) as exc:
        print(f"Unable to query CUDA device properties: {exc}", file=sys.stderr)
        return None


def _green_context_resolve_min_sm_size(partitions):
    """Pick the largest ``min_sm_size`` for which the CUDA driver accepts *partitions*.

    The architecture-default heuristic in
    :func:`_green_context_min_sm_size_for_device_major` is intentionally
    coarse and may not match every GPU's actual SM-grouping granularity.
    For example, on IGX Thor (compute capability 11.x, ``sm_count=20``)
    the Blackwell-class default of ``8`` is too large for the example's
    aarch64 partitioning of ``[4, 4]`` even though the driver itself
    accepts the smaller ``min_sm_size=4`` grouping just fine.

    To stay portable across such GPUs, this helper probes candidate
    ``min_sm_size`` values starting at the architecture default and
    halving down to ``2``, returning the largest value for which both
    the arithmetic constraints and
    :meth:`CudaGreenContextPool.is_partitioning_supported` succeed.

    Returns ``None`` when device properties cannot be read or no
    candidate is accepted.
    """
    props = _green_context_device_properties()
    if props is None:
        return None
    sm_count = props["sm_count"]
    total = sum(partitions)
    if sm_count < total:
        return None

    candidates = []
    candidate = props["min_sm_size"]
    while candidate >= 2:
        candidates.append(candidate)
        candidate //= 2

    for min_sm in candidates:
        if any(p < min_sm or p % min_sm != 0 for p in partitions):
            continue
        remainder = sm_count - total
        if remainder != 0 and remainder % min_sm != 0:
            continue
        if CudaGreenContextPool.is_partitioning_supported(0, min_sm, partitions):
            return min_sm
    return None


class CuPySourceOp(Operator):
    def __init__(
        self,
        fragment,
        *args,
        width=3840,
        height=2160,
        use_default_stream=False,
        **kwargs,
    ):
        self.height = height
        self.width = width
        self.use_default_stream = use_default_stream
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")

    def compute(self, op_input, op_output, context):
        if self.use_default_stream:
            cp_tensor = cp.linspace(0, 10, self.height * self.width, dtype=cp.float32)
            cp_tensor = cp_tensor.reshape((self.height, self.width))

            cp_tensor = cp.exp(cp_tensor)
        else:
            # create a stream managed by Holoscan
            stream = context.allocate_cuda_stream("my_stream")

            # Create a CuPy array and launch some kernel on it on the external stream
            with cp.cuda.ExternalStream(stream):
                cp_tensor = cp.linspace(0, 10, self.height * self.width, dtype=cp.float32)
                cp_tensor = cp_tensor.reshape((self.height, self.width))

                cp_tensor = cp.exp(cp_tensor)

            # configure to emit a stream ID component when emitting from the "out" port
            op_output.set_cuda_stream(stream, "out")

        # Emit the CuPy array (the stream ID will be emitted as well)
        op_output.emit(cp_tensor, "out")


class CuPyProcessOp(Operator):
    def __init__(self, fragment, *args, use_default_stream=False, **kwargs):
        self.use_default_stream = use_default_stream
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.input("in1")
        spec.input("in2")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        tensor1 = op_input.receive("in1")
        tensor2 = op_input.receive("in2")

        if self.use_default_stream:
            # CuPySourceOp emits CuPy array objects, but it is good practice to call
            # cp.asarray to ensure any array-like object is converted to a CuPy array.
            in1 = cp.asarray(tensor1)
            in2 = cp.asarray(tensor2)
            # elementwise multiplication
            prod1 = in1 * in2
            # sum along the last axis
            partial_sum = prod1.sum(axis=-1)
        else:
            # stream and stream2 will both be using the operator's internal stream
            # Any stream on in1 or in2 will have been synchronized to the same internal stream.
            stream = op_input.receive_cuda_stream("in1")
            stream2 = op_input.receive_cuda_stream("in2")
            assert stream2 == stream

            # Create a CuPy array on the stream
            with cp.cuda.ExternalStream(stream):
                # CuPySourceOp emits CuPy array objects, but it is good practice to call
                # cp.asarray to ensure any array-like object is converted to a CuPy array.
                in1 = cp.asarray(tensor1)
                in2 = cp.asarray(tensor2)

                # note that while within the cp.cuda.ExternalStream context
                #   in1.__cuda_array_interface__["stream"] == stream
                #   in2.__cuda_array_interface__["stream"] == stream

                # elementwise multiplication
                prod1 = in1 * in2
                # sum along the last axis
                partial_sum = prod1.sum(axis=-1)

        # Send the CuPy array to the output port
        # Use emitter_name="holoscan::Tensor" to output as a C++ holoscan::Tensor instead of a
        # default CuPy array object so that the output is compatible with wrapped C++ operators
        # like PingTensorRxOp.
        op_output.emit(partial_sum, "out", emitter_name="holoscan::Tensor")


class CuPyExampleApp(Application):
    def __init__(
        self,
        *args,
        count=10,
        use_default_stream=False,
        use_green_context=False,
        green_context_min_sm_size=None,
        **kwargs,
    ):
        self.count = count
        self.use_default_stream = use_default_stream
        self.use_green_context = use_green_context
        self.green_context_min_sm_size = green_context_min_sm_size

        super().__init__(*args, **kwargs)

    def compose(self):
        if self.use_green_context:
            arch = platform.machine().lower()
            if arch in ["x86_64", "amd64"]:
                partitions = [8, 8]
            elif arch in ["aarch64", "arm64"]:
                partitions = [4, 4]
            else:
                raise ValueError(f"Unsupported platform architecture: {arch}")
            min_sm_size = self.green_context_min_sm_size
            if min_sm_size is None:
                min_sm_size = _green_context_resolve_min_sm_size(partitions)
            if min_sm_size is None:
                raise RuntimeError(
                    "Green Context partitioning is not supported on this GPU; "
                    "the application launcher should have skipped before reaching compose()."
                )
            cuda_green_context_pool = CudaGreenContextPool(
                self,
                dev_id=0,
                num_partitions=2,
                sms_per_partition=partitions,
                min_sm_size=min_sm_size,
                name="cuda_green_context_pool",
            )
            cuda_green_context = CudaGreenContext(
                self,
                cuda_green_context_pool=cuda_green_context_pool,
                index=1,
                name="cuda_green_context",
            )

        else:
            cuda_green_context = None
            cuda_green_context_pool = None

        if self.use_default_stream:
            source1 = CuPySourceOp(
                self,
                CountCondition(self, self.count, name="source1_count"),
                use_default_stream=True,
                name="source1",
            )
            source2 = CuPySourceOp(
                self,
                CountCondition(self, self.count, name="source2_count"),
                use_default_stream=True,
                name="source2",
            )
            proc_op = CuPyProcessOp(
                self,
                use_default_stream=True,
                name="combine_and_sum",
            )
            rx = PingTensorRxOp(
                self,
                name="rx",
            )
        else:
            # Creat a cuda stream pool with green context if enabled
            stream_pool = CudaStreamPool(
                self,
                name="stream_pool",
                dev_id=0,
                stream_flags=0,
                stream_priority=0,
                reserved_size=1,
                max_size=5,
                cuda_green_context=cuda_green_context,
            )
            source1 = CuPySourceOp(
                self,
                stream_pool,
                CountCondition(self, self.count, name="source1_count"),
                use_default_stream=False,
                name="source1",
            )
            source2 = CuPySourceOp(
                self,
                stream_pool,
                CountCondition(self, self.count, name="source2_count"),
                use_default_stream=False,
                name="source2",
            )
            proc_op = CuPyProcessOp(
                self,
                stream_pool,
                use_default_stream=False,
                name="combine_and_sum",
            )
            stream_cond = CudaStreamCondition(self, receiver="in", name="stream_sync")
            rx = PingTensorRxOp(
                self,
                stream_cond,
                stream_pool,
                name="rx",
            )
        self.add_flow(source1, proc_op, {("out", "in1")})
        self.add_flow(source2, proc_op, {("out", "in2")})
        self.add_flow(proc_op, rx)


if __name__ == "__main__":
    parser = ArgumentParser(description="Operator stream handling example")
    parser.add_argument(
        "-d",
        "--default_stream",
        action="store_true",
        help=(
            "Sets the application to disable dedicated operator streams and just use the default "
            "stream for all kernels."
        ),
    )
    parser.add_argument(
        "-g",
        "--green_context",
        action="store_true",
        help=("Sets the application to use green context when creating cuda stream pool."),
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        default=10,
        help="The number of messages to transmit.",
    )
    args = parser.parse_args()
    if args.count < 1:
        raise ValueError("count must be a positive integer")

    # Honor HOLOSCAN_LOG_LEVEL for code that runs before the ``Application``
    # constructor -- notably ``CudaGreenContextPool.is_partitioning_supported``
    # called by ``_green_context_resolve_min_sm_size`` below.  Without this
    # call, ``Logger::set_level()`` is only invoked from the ``Application``
    # constructor, so any HOLOSCAN_LOG_DEBUG output emitted earlier is
    # silently dropped.
    #
    # Note: this does NOT force the level to INFO.  ``set_log_level()`` reads
    # HOLOSCAN_LOG_LEVEL first and overrides the passed-in level when the
    # env var is set to a recognized value (TRACE, DEBUG, INFO, WARN, ERROR,
    # CRITICAL, OFF; case-insensitive).  INFO is only the fallback when the
    # env var is unset.  See ``Logger::set_level`` in
    # ``src/logger/logger.cpp``.
    set_log_level(LogLevel.INFO)

    if args.green_context and not _green_context_supported_by_cuda_driver():
        version = _cuda_driver_version()
        version_display = "unknown" if version is None else str(version)
        print(
            "Green Context requires CUDA Driver API >= 12.4 "
            f"(cudaDriverGetVersion >= {MIN_GREEN_CONTEXT_CUDA_DRIVER_VERSION}, "
            f"detected: {version_display}). See {GREEN_CONTEXT_REQUIREMENT_DOC_URL}",
            file=sys.stderr,
        )
        sys.exit(77)

    green_context_min_sm_size = None
    if args.green_context:
        arch = platform.machine().lower()
        partitions = [8, 8] if arch in ["x86_64", "amd64"] else [4, 4]
        green_context_min_sm_size = _green_context_resolve_min_sm_size(partitions)
        if green_context_min_sm_size is None:
            props = _green_context_device_properties()
            sm_info = f" (sm_count={props['sm_count']})" if props else ""
            print(
                f"Green Context partitioning with {partitions} is not supported"
                f" on this GPU{sm_info}. See {GREEN_CONTEXT_REQUIREMENT_DOC_URL}",
                file=sys.stderr,
            )
            sys.exit(77)

    app = CuPyExampleApp(
        count=args.count,
        use_default_stream=args.default_stream,
        use_green_context=args.green_context,
        green_context_min_sm_size=green_context_min_sm_size,
    )
    app.run()
