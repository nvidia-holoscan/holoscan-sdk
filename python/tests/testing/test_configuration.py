# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native configuration ownership, controlled clocks and configured port declarations."""

import ctypes
import json

import pytest
from observation import OBSERVATION_MS

import holoscan as hs
from holoscan.testing import (
    ClockTestOutcome,
    ControlledClockRun,
    FrameRelease,
    OperatorTestHarness,
    sample_at,
)
from holoscan.testing import TestRunOptions as RunOptions


# REQ: PY-API-030
def test_controlled_clock_uses_native_selection_and_exact_integer_units():
    class Latest(hs.Operator):
        inp = hs.Input("in", hs.payload.uint32)
        out = hs.Output("out", hs.payload.uint32)

        def contract(self):
            return (
                hs.Contract()
                .trigger(hs.OnClock(period=hs.time.ms(10)))
                .input(self.inp, hs.LatestWithin(max_age=hs.time.ms(100)))
                .on_missing(self.inp, hs.Skip())
            )

        def compute(self, context):
            self.out.emit(self.inp.receive_data() * 2)

    options = ControlledClockRun(tick_ns=10_000_000)
    assert options.tick_ns == 10_000_000
    assert options.timeout_ms == 30_000
    with OperatorTestHarness(Latest) as harness:
        harness.push_samples("in", [sample_at(7, capture_timestamp_ns=0)])
        result = harness.run_controlled_clock(options)
    assert result.value("out") == 14
    stamp = result.samples("out")[0].metadata.publish_timestamp_ns
    assert stamp in (0, options.tick_ns)
    expected = ClockTestOutcome.SUPPRESSED if stamp == 0 else ClockTestOutcome.RETIRED
    assert result.clock_outcome == expected
    assert result.clock_selection("in").state == (
        hs.InputState.ABSENT if stamp == 0 else hs.InputState.SELECTED
    )


# REQ: PY-API-030
def test_params_refresh_automatic_output_selection_and_reject_incompatible_pending_input():
    class Configured(hs.Operator):
        inp = hs.Input("in", hs.payload.uint32)
        out = hs.Output("out", hs.payload.uint32)
        extra = hs.Output("extra", hs.payload.uint32)
        enabled = hs.Param(False)

        def setup(self, spec):
            spec.input(self.inp)
            spec.output(self.out)
            if self.enabled:
                spec.output(self.extra)

        def compute(self, context):
            value = self.inp.receive_data()
            self.out.emit(value)
            if self.enabled:
                self.extra.emit(value + 1)

    with OperatorTestHarness(Configured) as harness:
        harness.push("in", 7)
        harness.set_params({"enabled": True})
        result = harness.run()
        assert result.value("out") == 7
        assert result.value("extra") == 8
        harness.set_params({"enabled": False})
        harness.push("in", 9)
        result = harness.run()
        assert result.value("out") == 9
        with pytest.raises(ValueError, match="unknown or unselected"):
            result.values("extra")

    # An explicit capture remains a selector even when construction selected all outputs.
    with OperatorTestHarness(Configured, params={"enabled": True}) as harness:
        harness.capture("extra", capacity=1)
        harness.push("in", 13)
        with pytest.raises(ValueError, match="selected output absent"):
            harness.set_params({"enabled": False})
        result = harness.run()
        assert result.value("out") == 13
        assert result.value("extra") == 14
        harness.push_frame({"in": 1})
        harness.push_frame({"in": 2})
        with harness.run_async(RunOptions(frames=FrameRelease.MANUAL)) as running:
            running.release_frame(timeout_ms=OBSERVATION_MS)
            assert running.wait_for_output("extra", 1, timeout_ms=OBSERVATION_MS)
            running.release_frame(timeout_ms=OBSERVATION_MS)
            overflow = running.finish(check=False)
        assert not overflow.ok
        assert len(overflow.values("extra")) == 1


# REQ: PY-API-030
def test_connection_options_are_owned_and_native_overrides_can_be_cleared():
    class Echo(hs.Operator):
        inp = hs.Input("in", hs.payload.uint32)
        out = hs.Output("out", hs.payload.uint32)

        def compute(self, context):
            self.out.emit(self.inp.receive_data())

    with OperatorTestHarness(Echo) as harness:
        options = hs.ConnectionOptions(queue_depth=3, topic_name="owned-temporary-name")
        assert options.queue_depth == 3
        assert options.topic_name == "owned-temporary-name"
        harness.set_input_flow("in", queue_depth=options.queue_depth, topic_name=options.topic_name)
        del options
        harness.set_output_flow("out", queue_depth=3)
        harness.push("in", 7)
        assert harness.run().value("out") == 7
        harness.set_input_queue_depth("in", 3)
        harness.set_input_queue_depth("in", None)
        harness.clear_input_flow("in")
        harness.clear_output_flow("out")
        with pytest.raises(ValueError, match="does not carry Tensor"):
            harness.set_input_tensor_spec("in", None)
        compile_options = hs.CompileOptions()
        harness.set_compile_options(compile_options)
        harness.push("in", 11)
        assert harness.run().value("out") == 11


class HostArray:
    def __init__(self):
        self.storage = (ctypes.c_uint8 * 4)(1, 2, 3, 4)
        self.__array_interface__ = {
            "shape": (4,),
            "typestr": "|u1",
            "version": 3,
            "data": (ctypes.addressof(self.storage), False),
        }


class TensorCopy(hs.Operator):
    inp = hs.Input("in", hs.payload.Tensor)
    out = hs.Output("out", hs.payload.Tensor)

    def setup(self, spec):
        spec.input(self.inp)
        # Literal capacity permits graph overrides; derived allocation expressions forbid them.
        spec.output(self.out).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=1,
                bounds=hs.tensor.TensorBounds(max_byte_span=4),
            ),
        )

    def compute(self, context):
        value = self.inp.receive_data()
        with self.out.allocate_tensor(shape=(4,), dtype=hs.tensor.DLDataType(1, 8, 1)) as loan:
            ctypes.memmove(loan.tensor.data, value.data, 4)


@pytest.mark.parametrize(
    ("setter", "port", "spec_type"),
    [
        ("set_input_tensor_spec", "in", hs.tensor.InputSpec),
        ("set_output_tensor_spec", "out", hs.tensor.OutputSpec),
        ("set_input_source_tensor_spec", "in", hs.tensor.OutputSpec),
    ],
)
# REQ: PY-API-030
# REQ: PY-API-031
def test_tensor_overrides_copy_replace_clear_and_reject_active_mutation(setter, port, spec_type):
    def specification(*, rank=1, shape=(4,), byte_span=4):
        options = {
            "memory_kind": hs.tensor.MemoryKind.HOST,
            "dtype": hs.tensor.DLDataType(1, 8, 1),
            "rank": rank,
            "bounds": hs.tensor.TensorBounds(max_shape=shape, max_byte_span=byte_span),
        }
        if spec_type is hs.tensor.OutputSpec:
            options["storage"] = "runtime_pool"
        return spec_type(**options)

    value = hs.Tensor.as_tensor(HostArray())
    with OperatorTestHarness(TensorCopy) as harness:
        override = getattr(harness, setter)
        override(port, specification(shape=(2,), byte_span=2))
        owned = specification()
        override(port, owned)
        # Mutating the actual bound spec after setting must not shrink pending native bounds.
        owned.bounds.max_shape = [1]
        owned.bounds.max_byte_span = 1
        harness.push("in", value)
        assert harness.run().snapshot("out").bytes == bytes((1, 2, 3, 4))

        # Clearing a restrictive override restores the configured capacity without consuming
        # the already staged sample, whose four bytes exceed the temporary two-byte bound.
        harness.push("in", value)
        override(port, specification(shape=(2,), byte_span=2))
        override(port, None)
        assert harness.run().snapshot("out").bytes == bytes((1, 2, 3, 4))

        harness.push_frame({"in": value})
        with harness.run_async(RunOptions(frames=FrameRelease.MANUAL)) as running:
            for replacement in (specification(), None):
                with pytest.raises(RuntimeError, match="active attempt"):
                    override(port, replacement)
            running.release_frame(timeout_ms=OBSERVATION_MS)
            assert running.finish().snapshot("out").bytes == bytes((1, 2, 3, 4))


# REQ: PY-API-030
@pytest.mark.parametrize("direction,port", [("input", "in"), ("output", "out")])
def test_flow_setters_use_graph_keywords_and_preserve_state_after_invalid_updates(direction, port):
    class Echo(hs.Operator):
        inp = hs.Input("in", hs.payload.uint32)
        out = hs.Output("out", hs.payload.uint32)

        def compute(self, context):
            self.out.emit(self.inp.receive_data())

    with OperatorTestHarness(Echo) as harness:
        setter = getattr(harness, "set_" + direction + "_flow")
        clear = getattr(harness, "clear_" + direction + "_flow")
        setter(port, queue_depth=3, max_inflight_samples=3)
        harness.push_frame({"in": 17})
        with pytest.raises(hs.HoloscanError, match="GRAPH_CONNECTION_OPTION_TYPE_MISMATCH"):
            setter(port, queue_depth=3, max_inflight_samples="invalid")
        with harness.run_async(RunOptions(frames=FrameRelease.MANUAL)) as running:
            with pytest.raises(RuntimeError, match="active attempt"):
                setter(port)
            with pytest.raises(RuntimeError, match="active attempt"):
                clear(port)
            running.release_frame(timeout_ms=OBSERVATION_MS)
            assert running.finish().value("out") == 17
        # Default options remove explicit edge sizing; clearing restores the fixture's
        # input-count or capture-capacity sizing. Observe the compiled contract, not timing.
        edge_id = 1 if direction == "input" else 2
        setter(port)
        harness.push("in", 23)
        default_result = harness.run()
        assert default_result.value("out") == 23
        default_recipe = next(
            recipe
            for recipe in json.loads(default_result.plan.json())["queue_sizing_recipes"]
            if recipe["edge"] == edge_id
        )
        assert default_recipe["explicit_depth"] is None
        clear(port)
        harness.push("in", 29)
        cleared_result = harness.run()
        assert cleared_result.value("out") == 29
        cleared_recipe = next(
            recipe
            for recipe in json.loads(cleared_result.plan.json())["queue_sizing_recipes"]
            if recipe["edge"] == edge_id
        )
        assert cleared_recipe["explicit_depth"] == (1 if direction == "input" else 1024)
