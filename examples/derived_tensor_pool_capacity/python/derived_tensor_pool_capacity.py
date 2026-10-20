# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compile native Python operators with Parameter- and input-derived Tensor pools."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import holoscan as hs

DEFAULT_OUTPUT_SHAPE = (1080, 1920, 4)
BYTE_DTYPE = hs.tensor.DLDataType(hs.tensor.DLDataTypeCode.DLUINT, 8, 1)
FLOAT_DTYPE = hs.tensor.DLDataType(hs.tensor.DLDataTypeCode.DLFLOAT, 32, 1)
CAMERA_BYTE_SPAN = 3840 * 2160 * 4


class CapacityCameraSource(hs.Operator):
    """Declare the fixed-rate bounded camera output used by build_graph."""

    output = hs.Output("output", hs.payload.Tensor)

    def setup(self, spec):
        spec.output(self.output).nominal_rate(60).produces_tensor(
            hs.tensor.OutputSpec(
                storage=hs.tensor.OutputStorage.RUNTIME_POOL,
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=BYTE_DTYPE,
                rank=3,
                bounds=hs.tensor.TensorBounds(max_byte_span=CAMERA_BYTE_SPAN),
            )
        )

    def compute(self, context):
        pass


class UnboundedTensorSource(hs.Operator):
    """Omit a finite output bound for the build_unbounded_plan failure example."""

    output = hs.Output("output", hs.payload.Tensor)

    def setup(self, spec):
        spec.output(self.output)

    def compute(self, context):
        pass


class ConfiguredResize(hs.Operator):
    """Derive output capacity, storage, and rate from frozen declarations in build_graph."""

    input = hs.Input("input", hs.payload.Tensor)
    output = hs.Output("output", hs.payload.Tensor)
    output_shape = hs.Param([720, 1280, 4], max_elements=3, recompile_required=True)
    pinned_output = hs.Param(False, recompile_required=True)

    def setup(self, spec):
        spec.input(self.input).expected_rate(60).expects_tensor(
            hs.tensor.InputSpec(
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=BYTE_DTYPE,
                rank=3,
            )
        )
        # Keep this choice as compiler-visible evidence tied to the declared parameter.
        storage = hs.tensor.parameter_memory_kind(
            type(self).pinned_output,
            {
                False: hs.tensor.MemoryKind.HOST,
                True: hs.tensor.MemoryKind.PINNED_HOST,
            },
        )
        spec.output(self.output).follows(self.input, numerator=1, denominator=4).produces_tensor(
            hs.tensor.OutputSpec(
                storage=hs.tensor.OutputStorage.RUNTIME_POOL, dtype=BYTE_DTYPE, rank=3
            ),
            hs.tensor.parameter_contiguous_byte_span(type(self).output_shape),
            memory_kind=storage,
        )

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        pass


class ProjectedBatchOutput(hs.Operator):
    """Exercise a checked closed width/height/batch projection in build_graph."""

    output = hs.Output("output", hs.payload.Tensor)
    width = hs.Param(1280, range=(1, 16384), recompile_required=True)
    height = hs.Param(720, range=(1, 16384), recompile_required=True)
    max_batch_size = hs.Param(2, range=(1, 64), recompile_required=True)

    def setup(self, spec):
        capacity = (
            hs.tensor.parameter_byte_span(type(self).width)
            .multiplied_by(hs.tensor.parameter_byte_span(type(self).height))
            .multiplied_by(hs.tensor.parameter_byte_span(type(self).max_batch_size))
            .multiplied_by(hs.tensor.byte_span(4 * 4))
        )
        spec.output(self.output).produces_tensor(
            hs.tensor.OutputSpec(
                storage=hs.tensor.OutputStorage.RUNTIME_POOL,
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=FLOAT_DTYPE,
                rank=4,
            ),
            capacity,
        )

    def compute(self, context):
        pass


class ProjectedBatchSink(hs.Operator):
    """Require the projected rank-four Tensor in build_graph."""

    input = hs.Input("input", hs.payload.Tensor)

    def setup(self, spec):
        spec.input(self.input).expects_tensor(
            hs.tensor.InputSpec(
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=FLOAT_DTYPE,
                rank=4,
            )
        )

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        pass


class PackedGrayscale(hs.Operator):
    """Use the exact packed-RGBA reduction in build_graph."""

    input = hs.Input("input", hs.payload.Tensor)
    output = hs.Output("output", hs.payload.Tensor)
    pinned_input = hs.Param(False, recompile_required=True)

    def setup(self, spec):
        # This frozen setup-time branch selects a literal contract; no derivation is needed here.
        input_memory = (
            hs.tensor.MemoryKind.PINNED_HOST if self.pinned_input else hs.tensor.MemoryKind.HOST
        )
        spec.input(self.input).expects_tensor(
            hs.tensor.InputSpec(memory_kind=input_memory, dtype=BYTE_DTYPE, rank=3)
        )
        spec.output(self.output).produces_tensor(
            hs.tensor.OutputSpec(
                storage=hs.tensor.OutputStorage.RUNTIME_POOL,
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=BYTE_DTYPE,
                rank=2,
            ),
            hs.tensor.input_max_byte_span(self.input).ceiling_divided_by(hs.tensor.byte_span(4)),
        )

    def contract(self):
        return (
            hs.Contract()
            .trigger(hs.OnClock(period=hs.time.ms(1000)))
            .input(self.input, hs.Exact())
            .on_missing(self.input, hs.Skip())
        )

    def compute(self, context):
        pass


class ConservativeGrayscale(hs.Operator):
    """Preserve the full input capacity for conservative sizing in build_graph."""

    input = hs.Input("input", hs.payload.Tensor)
    output = hs.Output("output", hs.payload.Tensor)
    pinned_input = hs.Param(False, recompile_required=True)

    def setup(self, spec):
        # This frozen setup-time branch selects a literal contract; no derivation is needed here.
        input_memory = (
            hs.tensor.MemoryKind.PINNED_HOST if self.pinned_input else hs.tensor.MemoryKind.HOST
        )
        spec.input(self.input).expects_tensor(
            hs.tensor.InputSpec(memory_kind=input_memory, dtype=BYTE_DTYPE, rank=3)
        )
        spec.output(self.output).produces_tensor(
            hs.tensor.OutputSpec(
                storage=hs.tensor.OutputStorage.RUNTIME_POOL,
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=BYTE_DTYPE,
                rank=2,
            ),
            hs.tensor.input_max_byte_span(self.input),
        )

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.input))

    def compute(self, context):
        pass


def build_graph(
    output_shape: Sequence[int] = DEFAULT_OUTPUT_SHAPE, output_storage: str = "host"
) -> hs.Graph:
    """Build the native Python graph with compiler-resolved structural declarations."""
    shape = list(output_shape)
    if len(shape) != 3 or any(extent <= 0 for extent in shape):
        raise ValueError("output shape must contain three positive extents")
    if shape[2] != 4:
        raise ValueError("CHANNELS must be 4 for the packed RGBA example")
    if output_storage not in {"host", "pinned"}:
        raise ValueError("output storage must be host or pinned")

    graph = hs.Graph("derived-tensor-pool-capacity")
    pinned = output_storage == "pinned"
    camera = graph.op(CapacityCameraSource, "camera")
    resize = graph.op(
        ConfiguredResize,
        "resize",
        output_shape=shape,
        pinned_output=pinned,
    )
    projected_batch = graph.op(
        ProjectedBatchOutput,
        "projected_batch",
        width=shape[1],
        height=shape[0],
        max_batch_size=2,
    )
    projected_batch_sink = graph.op(ProjectedBatchSink, "projected_batch_sink")
    packed_grayscale = graph.op(PackedGrayscale, "packed_grayscale", pinned_input=pinned)
    conservative_grayscale = graph.op(
        ConservativeGrayscale, "conservative_grayscale", pinned_input=pinned
    )
    graph.add_flow(camera.output, resize.input)
    graph.add_flow(projected_batch.output, projected_batch_sink.input)
    graph.add_flow(resize.output, packed_grayscale.input)
    graph.add_flow(resize.output, conservative_grayscale.input, queue_depth=4)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "app-clock"))
    return graph


def build_plan(
    output_shape: Sequence[int] = DEFAULT_OUTPUT_SHAPE, output_storage: str = "host"
) -> hs.Plan:
    """Compile the example graph and return its inspectable plan."""
    return hs.compile(build_graph(output_shape, output_storage))


def build_unbounded_plan() -> hs.Plan:
    """Compile the shipped failure case for an input-derived bound without producer authority."""
    graph = hs.Graph("derived-tensor-pool-capacity-unbounded")
    source = graph.op(UnboundedTensorSource, "source")
    grayscale = graph.op(ConservativeGrayscale, "grayscale", pinned_input=False)
    graph.add_flow(source.output, grayscale.input)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "app-clock"))
    return hs.compile(graph)


def parse_args() -> argparse.Namespace:
    """Parse an optional HWC shape and the machine-readable plan mode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("shape", metavar="DIMENSION", nargs="*", type=int)
    parser.add_argument("--output-storage", choices=("host", "pinned"), default="host")
    parser.add_argument("--plan-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Compile the native graph and report its resolved derived capacities."""
    args = parse_args()
    shape = args.shape or DEFAULT_OUTPUT_SHAPE
    try:
        plan = build_plan(shape, args.output_storage)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not plan:
        raise SystemExit(plan.summary())
    if args.plan_json:
        print(plan.json(), end="")
        return

    capacities = {}
    for operator_name in (
        "resize",
        "packed_grayscale",
        "conservative_grayscale",
        "projected_batch",
    ):
        evidence = plan.find_tensor_byte_span_derivation(operator_name, "output")
        if evidence is None:
            raise RuntimeError(f"missing derived-capacity evidence for {operator_name}.output")
        capacities[operator_name] = evidence.resolved_max_byte_span

    memory_evidence = plan.find_tensor_memory_kind_derivation("resize", "output")
    if memory_evidence is None:
        raise RuntimeError("missing derived-memory evidence for resize.output")
    expected_memory = (
        hs.tensor.MemoryKind.PINNED_HOST
        if args.output_storage == "pinned"
        else hs.tensor.MemoryKind.HOST
    )
    if memory_evidence.resolved_memory_kind is not expected_memory:
        raise RuntimeError("resolved resize memory kind does not match the selected parameter")
    memory_label = "pinned_host" if expected_memory is hs.tensor.MemoryKind.PINNED_HOST else "host"

    print(
        "derived tensor pool capacity Python: "
        + " ".join(f"{name}_bytes={value}" for name, value in capacities.items())
        + f" output_storage={args.output_storage} resize_memory={memory_label} "
        + "native Python expressions verified"
    )


if __name__ == "__main__":
    main()
