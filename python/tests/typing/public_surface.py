# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Positive ty checks for the installed, provider-neutral public surface."""

from datetime import timedelta
from typing import Any, Never, assert_type

import holoscan as hs
from holoscan import channel, clocks, payload, sensor_io, topics
from holoscan.schema import Header, Image, Imu, Time
from holoscan.schema.geometry import Matrix3f
from holoscan.schema.image import ColorSpace, ImageEncoding, ImagePlaneLayout
from holoscan.schema.imu import ImuFlags

# REQ: PY-API-018
graph = hs.Graph("typing-public-surface")
assert_type(graph, hs.Graph)
assert_type(graph.default_clock(), clocks.ClockKey)

options = topics.TopicOptions(endpoint_name="frames")
assert_type(options, topics.TopicOptions)
assert_type(payload.uint32, payload.PayloadType)
assert_type(payload.schema.Time(), Never)

cuda_options = hs.execution.CudaExecutionCompileOptions(
    compute_stream_policy=hs.execution.CudaComputeStreamPolicy.SHARED_PROCESS_DEVICE
)
compile_options = hs.CompileOptions(cuda_execution=cuda_options)
support_binding = hs.execution.DeploymentBinding()
assert_type(
    support_binding.bind_support_thread_affinity(
        hs.execution.SupportThreadRole.CONTROLLER,
        hs.execution.CPUAffinity([0]),
    ),
    hs.execution.DeploymentBinding,
)
assert_type(
    compile_options.cuda_execution.compute_stream_policy,
    hs.execution.CudaComputeStreamPolicy,
)

plan = hs.compile(graph, compile_options)
assert_type(plan.cuda_compute_stream_policy, hs.execution.CudaComputeStreamPolicy)
for partition in plan.partitions:
    assert_type(partition, hs.execution.ExecutionPartition)
assert_type(list(plan.partitions), list[hs.execution.ExecutionPartition])

contract = hs.Contract().trigger(hs.OnClock(period=timedelta(milliseconds=1)))
assert_type(contract, hs.Contract)
assert_type(hs.LatestWithin(max_age=hs.time.ms(5)).max_age, hs.time.Duration)

header = Header(stamp=Time(time_ns=7), frame_id="camera")
assert_type(header, Header)
assert_type(header.stamp, Time | None)
assert_type(Matrix3f().m, tuple[float, ...])

image = Image(
    header=header,
    height=2,
    width=3,
    encoding=ImageEncoding.MONO8,
    color_space=ColorSpace.UNSPECIFIED,
    plane_layouts=[ImagePlaneLayout()],
)
assert_type(image, Image)
assert_type(image.header, Header | None)
assert_type(image.plane_layouts, list[ImagePlaneLayout])

imu = Imu(flags=ImuFlags.ORIENTATION_VALID, orientation_covariance=Matrix3f())
assert_type(imu, Imu)
assert_type(imu.flags, ImuFlags)
assert_type(imu.orientation_covariance, Matrix3f | None)


def tensor_projection_surface(output: hs.Port, input_: hs.Port, spec_port: hs.OperatorPort) -> None:
    """Type-check descriptor Tensor-field projection authoring."""
    projection = output.project("data")
    assert_type(projection, hs.tensor.TensorProjection)
    assert_type(projection.source, hs.Port)
    assert_type(graph.add_flow(projection, input_), hs.Flow)
    assert_type(spec_port.accepts_projected_tensor(), hs.OperatorPort)


# REQ: PY-API-021
def callback_surface(
    ctx: hs.ExecutionContext,
    port: hs.OperatorPort,
    sample: hs.Sample[int],
    batch: hs.SelectedBatch[int],
) -> None:
    """Type-check callback context, selection, and publication value objects."""
    point = hs.time.TimePoint(1_000, 1)
    options = hs.EmitOptions(capture_time=point, frame_id=2, trace_id=3)
    selection = hs.InputSelection(
        state=hs.InputState.SELECTED,
        reason=hs.SelectionReason.NONE,
    )
    clock = hs.OnClock(period=hs.time.ms(1), start_at=point)

    assert_type(ctx.activation_time, hs.time.TimePoint)
    assert_type(ctx.tag_time, hs.time.TagTime)
    assert_type(ctx.activation, hs.ActivationStatus)
    assert_type(selection, hs.InputSelection)
    assert_type(clock.start_at, hs.time.TimePoint | None)
    assert_type(batch[0], hs.Sample[int])
    assert_type(batch[1::2], list[hs.Sample[int]])
    port.emit(sample.data, options=options)
    port.forward(sample, options=options)


channel_options = channel.ChannelOptions(enable_event_wake=True)
assert_type(channel_options, channel.ChannelOptions)


def check_channel_types(subscriber: channel.Subscriber) -> None:
    assert_type(subscriber.try_receive(), hs.Sample[Any] | None)
    assert_type(subscriber.receive(1.0), hs.Sample[Any])
    assert_type(subscriber.readiness(), channel.ReadinessHandle)


async def check_channel_aio_types(
    subscriber: channel.Subscriber, publisher: channel.Publisher
) -> None:
    assert_type(await channel.aio.Subscriber(subscriber).receive(), hs.Sample[Any])
    assert_type(await channel.aio.Publisher(publisher).emit(1, timeout=1.0), int)


def check_sensor_io_types(sample: hs.Sample[int]) -> None:
    tracker = sensor_io.SequenceTracker()
    assert_type(tracker.observe(1), sensor_io.SequenceObservation)
    accumulator = sensor_io.ProvenanceAccumulator()
    accumulator.observe(sample.metadata)
    assert_type(
        accumulator.emit_options(sensor_io.CaptureTimePolicy.EARLIEST),
        hs.EmitOptions,
    )
