# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite native operator testing through the optional Python testing component."""

import holoscan_test_support  # noqa: F401 -- registers unchanged native test operators
import pytest

import holoscan as hs
from holoscan.errors import HoloscanError
from holoscan.testing import OperatorTestHarness, sample_at


class Scale(hs.Operator):
    inp = hs.Input("in", hs.payload.uint32)
    out = hs.Output("out", hs.payload.uint32)
    factor = hs.Param(2, range=(1, 8))

    def compute(self, context):
        self.out.emit(self.inp.receive_data() * self.factor)


class CountCallbacks(hs.Operator):
    inp = hs.Input("in", hs.payload.uint32)
    out = hs.Output("out", hs.payload.uint32)

    def compute(self, context):
        self.inp.receive_data()
        self.count = getattr(self, "count", 0) + 1
        self.out.emit(self.count)


class Fail(hs.Operator):
    inp = hs.Input("in", hs.payload.uint32)
    out = hs.Output("out", hs.payload.uint32)

    def compute(self, context):
        raise RuntimeError("intentional operator test failure")


class ObjectEcho(hs.Operator):
    inp = hs.Input("in", hs.pyobject)
    out = hs.Output("out", hs.pyobject)

    def compute(self, context):
        self.out.emit(self.inp.receive_data())


class CaptureCoordinate(hs.Operator):
    inp = hs.Input("in", hs.payload.uint32)
    out = hs.Output("out", hs.payload.uint64)

    def compute(self, context):
        self.out.emit(self.inp.receive().metadata.capture_timestamp_ns)


# REQ: PY-API-030
@pytest.mark.parametrize("operator", [Scale, "holoscan.test.DoubleIt"])
# REQ: PY-API-030
def test_registered_and_python_operators_share_the_harness(operator):
    with OperatorTestHarness(operator) as harness:
        harness.push_many("in", [1, 2, 3])
        first = harness.run()
        harness.push("in", 21)
        second = harness.run()
    assert first.ok
    assert first.values("out") == [2, 4, 6]
    assert second.value("out") == 42
    assert [sample.data for sample in first.samples("out")] == [2, 4, 6]


# REQ: PY-API-030
def test_callbacks_share_an_instance_and_a_new_run_starts_fresh():
    with OperatorTestHarness(CountCallbacks) as harness:
        harness.push_many("in", [0, 0, 0])
        first = harness.run()
        harness.push("in", 0)
        second = harness.run()
    assert first.values("out") == [1, 2, 3]
    assert second.value("out") == 1


# REQ: PY-API-030
def test_params_use_the_canonical_constructor():
    with OperatorTestHarness(Scale, params={"factor": 3}) as harness:
        harness.push("in", 7)
        assert harness.run().value("out") == 21
    with pytest.raises(HoloscanError):
        OperatorTestHarness(Scale, params={"factor": 0})


# REQ: PY-API-030
def test_atomic_batch_failure_preserves_prior_staging():
    with OperatorTestHarness(Scale) as harness:
        harness.push("in", 1)
        with pytest.raises(
            hs.errors.InvalidArgumentError, match="AUTHORING_PAYLOAD_VALUE_TYPE_MISMATCH"
        ):
            harness.push_many("in", [2, "invalid uint32"])
        assert harness.run().values("out") == [2]


# REQ: PY-API-030
def test_failed_attempt_is_checked_or_returned_without_rerunning():
    with OperatorTestHarness(Fail) as harness:
        harness.push("in", 1)
        with pytest.raises(HoloscanError) as raised:
            harness.run()
        failed = raised.value.result
        assert not failed.ok
        assert failed.values("out") == []
        assert "intentional operator test failure" in failed.describe_failure()
        harness.push("in", 2)
        unchecked = harness.run(check=False)
        assert not unchecked.ok
        assert unchecked.values("out") == []


# REQ: PY-API-030
def test_python_object_identity_container_and_none_remain_data():
    value = [1, 2, 3]
    with OperatorTestHarness(ObjectEcho) as harness:
        harness.push("in", value)
        harness.push("in", None)
        result = harness.run()
    assert result.values("out")[0] is value
    assert result.values("out")[1] is None
    value.append(4)
    assert result.values("out")[0] == [1, 2, 3, 4]


# REQ: PY-API-030
def test_explicit_timestamps_preserve_native_selection():
    with OperatorTestHarness(CaptureCoordinate) as harness:
        harness.push_samples(
            "in", [sample_at(1, capture_timestamp_ns=100), sample_at(2, capture_timestamp_ns=0)]
        )
        result = harness.run()
    # Native Runtime selects by capture coordinate, as in the C++ harness parity test.
    # Staging publishes the caller's order and does not impose a separate FIFO policy.
    assert result.values("out") == [0, 100]


# REQ: PY-API-030
def test_empty_capture_unknown_output_and_singular_access_are_distinct():
    with OperatorTestHarness(Scale) as harness:
        harness.push_many("in", [])
        empty = harness.run()
        assert empty.values("out") == []
        with pytest.raises(ValueError, match="out.*0.*values"):
            empty.value("out")
        with pytest.raises(ValueError, match="unknown or unselected.*unknown"):
            empty.values("unknown")
        harness.push_many("in", [1, 2])
        multiple = harness.run()
        with pytest.raises(ValueError, match="out.*2.*values"):
            multiple.value("out")


# REQ: PY-API-030
def test_explicit_output_subset_and_capture_limit():
    with OperatorTestHarness(Scale, outputs=[]) as harness:
        harness.push("in", 1)
        # Omitting a collector preserves native topology admission and emit failures.
        result = harness.run(check=False)
        with pytest.raises(ValueError, match="unknown or unselected.*out"):
            result.values("out")
    with OperatorTestHarness(Scale) as harness:
        harness.capture("out", capacity=1)
        harness.set_output_flow("out", queue_depth=2)
        harness.push_many("in", [1, 2])
        result = harness.run(check=False)
        assert not result.ok
        assert result.values("out") == [2]


# REQ: PY-API-030
def test_grouped_inputs_are_atomic_and_none_is_data():
    class Pair(hs.Operator):
        left = hs.Input("left", hs.pyobject)
        right = hs.Input("right", hs.payload.uint32)
        out = hs.Output("out", hs.pyobject)

        def contract(self):
            return (
                hs.Contract()
                .trigger(hs.OnAligned(ports=[self.left, self.right], window=hs.time.ns(1)))
                .input(self.left, hs.Exact())
                .input(self.right, hs.Exact())
            )

        def compute(self, context):
            self.out.emit((self.left.receive_data(), self.right.receive_data()))

    with OperatorTestHarness(Pair) as harness:
        harness.push_frame([("left", None), ("right", 7)], capture_timestamp_ns=0)
        with pytest.raises(
            hs.errors.InvalidArgumentError, match="AUTHORING_PAYLOAD_VALUE_TYPE_MISMATCH"
        ):
            harness.push_frame([("left", "must not stage"), ("right", "invalid")])
        assert harness.run().value("out") == (None, 7)


# REQ: PY-API-030
def test_set_params_preserves_staging_and_failed_update_is_atomic():
    with OperatorTestHarness(Scale) as harness:
        harness.push("in", 7)
        harness.set_params({"factor": 3})
        with pytest.raises(HoloscanError):
            harness.set_params({"factor": 0})
        assert harness.run().value("out") == 21


# REQ: PY-API-030
@pytest.mark.parametrize(
    ("payload", "values", "invalid"),
    [
        (hs.payload.boolean, [False, True], 2),
        (hs.payload.int8, [-128, 127], 128),
        (hs.payload.uint8, [0, 255], 256),
        (hs.payload.int16, [-32768, 32767], 32768),
        (hs.payload.uint16, [0, 65535], 65536),
        (hs.payload.int32, [-(2**31), 2**31 - 1], 2**31),
        (hs.payload.uint32, [0, 2**32 - 1], 2**32),
        (hs.payload.int64, [-(2**63), 2**63 - 1], 2**63),
        (hs.payload.uint64, [0, 2**64 - 1], 2**64),
        (hs.payload.float32, [-1.25, 2.5], "invalid"),
        (hs.payload.float64, [-1.25, 2.5], "invalid"),
    ],
)
def test_all_registered_scalar_widths_preserve_values_and_atomic_rejection(
    payload, values, invalid
):
    class Echo(hs.Operator):
        inp = hs.Input("in", payload)
        out = hs.Output("out", payload)

        def compute(self, context):
            self.out.emit(self.inp.receive_data())

    with OperatorTestHarness(Echo) as harness:
        harness.push_many("in", values)
        with pytest.raises(hs.errors.InvalidArgumentError, match="AUTHORING_PAYLOAD_VALUE_"):
            harness.push_many("in", [values[0], invalid])
        assert harness.run().values("out") == values
