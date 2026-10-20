# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical schema conversion and independently owned Tensor captures."""

import ctypes

import control_payload
import pytest
import stereo_payload
from observation import OBSERVATION_SECONDS

import holoscan as hs
from holoscan.testing import OperatorTestHarness


class HostArray:
    def __init__(self, values):
        self.storage = (ctypes.c_uint8 * len(values))(*values)
        self.__array_interface__ = {
            "shape": (len(values),),
            "typestr": "|u1",
            "version": 3,
            "data": (ctypes.addressof(self.storage), False),
        }


class TensorSource(hs.Operator):
    out = hs.Output("out", hs.payload.Tensor)

    def setup(self, spec):
        spec.output(self.out).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=1,
            ),
            hs.tensor.byte_span(4),
        )

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        with self.out.allocate_tensor(shape=(4,), dtype=hs.tensor.DLDataType(1, 8, 1)) as loan:
            (ctypes.c_uint8 * 4).from_address(loan.tensor.data)[:] = (1, 2, 3, 4)


class StereoSource(hs.Operator):
    out = hs.Output("out", stereo_payload.StereoPayload)

    def setup(self, spec):
        output = spec.output(self.out)
        tensor_spec = hs.tensor.OutputSpec(
            storage="runtime_pool",
            memory_kind=hs.tensor.MemoryKind.HOST,
            dtype=hs.tensor.DLDataType(1, 8, 1),
            rank=1,
        )
        for field in ("left", "right"):
            output.produces_tensor(tensor_spec, hs.tensor.byte_span(4), field=field)

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        request = hs.tensor.LoanRequest(shape=(4,), dtype=hs.tensor.DLDataType(1, 8, 1))
        with self.out.allocate_tensors(
            stereo_payload.StereoPayload(
                header=stereo_payload.FrameHeader(sequence=9, source="testing")
            ),
            {"left": request, "right": request},
        ) as loan:
            for field, values in (("left", (1, 2, 3, 4)), ("right", (5, 6, 7, 8))):
                (ctypes.c_uint8 * 4).from_address(loan.tensors[field].data)[:] = values


# REQ: PY-API-030
def test_generated_schema_value_copy_and_atomic_validation():
    class Echo(hs.Operator):
        inp = hs.Input("in", control_payload.ControlPayload)
        out = hs.Output("out", control_payload.ControlPayload)

        def compute(self, context):
            self.out.emit(self.inp.receive_data())

    source = control_payload.ControlPayload(sequence=7, gain=2.5, labels=["left"])
    with OperatorTestHarness(Echo) as harness:
        harness.push("in", source)
        source.labels.append("later")
        with pytest.raises(hs.errors.HoloscanError):
            harness.push_many("in", [{"sequence": 8}, {"sequence": -1}])
        result = harness.run()
    value = result.value("out")
    assert isinstance(value, control_payload.ControlPayload)
    assert value.sequence == 7
    assert value.labels == ["left"]


# REQ: PY-API-030
def test_tensor_snapshots_survive_close_and_enforce_access_policy():
    with OperatorTestHarness(TensorSource) as harness:
        result = harness.run()
    snapshot = result.snapshot("out")
    assert snapshot.bytes == bytes((1, 2, 3, 4))
    assert snapshot.shape == [4]
    assert snapshot.dtype.code == 1
    assert snapshot.dtype.bits == 8
    assert snapshot.dtype.lanes == 1
    assert snapshot.source_device.device_type == hs.tensor.DLDeviceType.DLCPU
    assert snapshot.source_byte_strides == [1]
    assert snapshot.source_byte_offset == 0
    assert snapshot.sample_metadata.publish_timestamp_ns >= 0
    assert result.capture_failure("out") is None
    assert result.plan is not None
    assert result.diagnostics == []
    with pytest.raises(ValueError, match="out.*snapshot.*snapshots"):
        result.values("out")


# REQ: PY-API-030
def test_multi_tensor_schema_snapshot_captures_every_field_and_budget():
    with OperatorTestHarness(StereoSource) as harness:
        result = harness.run()
        harness.capture_snapshot("out", maximum_bytes=7)
        failed = harness.run(check=False)
    snapshot = result.snapshot("out")
    assert snapshot.metadata.header.sequence == 9
    assert snapshot.tensor_fields == ["left", "right"]
    assert snapshot.tensor("left").bytes == bytes((1, 2, 3, 4))
    assert snapshot.tensor("right").bytes == bytes((5, 6, 7, 8))
    assert not failed.ok
    assert failed.snapshots("out") == []
    failure = failed.capture_failure("out")
    assert failure is not None
    assert failure.error.code == hs.errors.ErrorCode.RESOURCE_EXHAUSTED
    assert failure.error.message
    assert failure.metadata.publish_timestamp_ns >= 0


# REQ: PY-API-030
def test_staged_tensor_and_all_schema_tensor_fields_use_native_loans():
    class TensorSum(hs.Operator):
        inp = hs.Input("in", hs.payload.Tensor)
        out = hs.Output("out", hs.payload.uint32)

        def compute(self, context):
            tensor = self.inp.receive_data()
            self.out.emit(sum((ctypes.c_uint8 * tensor.nbytes).from_address(tensor.data)))

    class StereoSum(hs.Operator):
        inp = hs.Input("in", stereo_payload.StereoPayload)
        out = hs.Output("out", hs.payload.uint32)

        def compute(self, context):
            value = self.inp.receive_data()
            total = 0
            for tensor in (value.left, value.right):
                total += sum((ctypes.c_uint8 * tensor.nbytes).from_address(tensor.data))
            self.out.emit(total)

    left = hs.Tensor.as_tensor(HostArray((1, 2, 3, 4)))
    right = hs.Tensor.as_tensor(HostArray((5, 6, 7, 8)))
    with OperatorTestHarness(TensorSum) as harness:
        harness.push("in", left)
        assert harness.run().value("out") == 10
    with OperatorTestHarness(StereoSum) as harness:
        harness.push("in", stereo_payload.StereoPayload(left=left, right=right))
        assert harness.run().value("out") == 36


@pytest.mark.parametrize("operator", [TensorSource, StereoSource])
# REQ: PY-API-030
def test_native_retention_and_plan_replay_preserve_the_original_capture(operator):
    with OperatorTestHarness(operator) as harness:
        harness.capture_native("out")
        result = harness.run()
    value = result.value("out")
    tensor = value if isinstance(value, hs.Tensor) else value.left
    assert bytes((ctypes.c_uint8 * 4).from_address(tensor.data)) == bytes((1, 2, 3, 4))
    with hs.run_async(result.plan) as replay:
        assert replay.wait(timeout=OBSERVATION_SECONDS)
        assert replay.diagnostics == []
    assert bytes((ctypes.c_uint8 * 4).from_address(tensor.data)) == bytes((1, 2, 3, 4))
