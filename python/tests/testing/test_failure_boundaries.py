# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hosted lifecycle and result-materialization failures retain their native attempt evidence."""

import control_payload
import holoscan_constructor_handle_provider as provider
import pytest

import holoscan as hs
from holoscan.testing import OperatorTestHarness


@pytest.mark.parametrize("callback", ["start", "stop"])
@pytest.mark.parametrize("check", [False, True])
# REQ: PY-API-030
def test_hosted_startup_and_cleanup_failures_preserve_result_and_allow_fresh_retry(callback, check):
    calls = []
    fail = True

    def invoke(name):
        calls.append(name)
        if fail and name == callback:
            error = hs.errors.InvalidArgumentError(f"testing {callback} failure")
            error.code = hs.errors.ErrorCode.INVALID_ARGUMENT
            raise error

    class LifecycleFailure(hs.Operator):
        inp = hs.Input("in", hs.payload.uint32)
        out = hs.Output("out", hs.payload.uint32)

        def start(self):
            invoke("start")

        def compute(self, context):
            calls.append("compute")
            self.out.emit(self.inp.receive_data())

        def stop(self):
            invoke("stop")

    with OperatorTestHarness(LifecycleFailure) as harness:
        harness.push("in", 7)
        if check:
            with pytest.raises(hs.errors.InvalidArgumentError) as caught:
                harness.run()
            result = caught.value.result
        else:
            result = harness.run(check=False)
        assert not result.ok
        assert result.termination == hs.RunTermination.FAILED
        prefix = [] if callback == "start" else [7]
        assert result.values("out") == prefix
        errors = [
            item.detail.error
            for item in result.diagnostics
            if item.detail is not None
            and f"testing {callback} failure" in item.detail.error.message
        ]
        assert errors
        assert all(error.code == hs.errors.ErrorCode.INVALID_ARGUMENT for error in errors)
        assert all("test_failure_boundaries.py" in error.message for error in errors)
        observed_calls = list(calls)
        assert f"testing {callback} failure" in result.describe_failure()
        assert result.values("out") == prefix
        assert calls == observed_calls

        # An executed failure consumes its input. A fresh attempt gets only the newly staged
        # value, while the old result stays inspectable after the operator has run again.
        fail = False
        harness.push("in", 9)
        assert harness.run().values("out") == [9]
        assert result.values("out") == prefix
    assert result.values("out") == prefix


# REQ: PY-API-030
def test_primary_compute_failure_keeps_its_code_when_cleanup_also_fails():
    class MultipleFailures(hs.Operator):
        inp = hs.Input("in", hs.payload.uint32)
        out = hs.Output("out", hs.payload.uint32)

        def compute(self, context):
            error = hs.errors.InvalidArgumentError("primary compute sentinel")
            error.code = hs.errors.ErrorCode.INVALID_ARGUMENT
            raise error

        def stop(self):
            raise RuntimeError("secondary cleanup sentinel")

    with OperatorTestHarness(MultipleFailures) as harness:
        harness.push("in", 1)
        with pytest.raises(hs.errors.InvalidArgumentError) as caught:
            harness.run()
    result = caught.value.result
    assert not result.ok
    assert result.values("out") == []
    details = [item.detail for item in result.diagnostics if item.detail is not None]
    assert any(
        detail.role == hs.diagnostics.DiagnosticFailureRole.PRIMARY
        and detail.error.code == hs.errors.ErrorCode.INVALID_ARGUMENT
        and "primary compute sentinel" in detail.error.message
        for detail in details
    )
    assert any(
        detail.role == hs.diagnostics.DiagnosticFailureRole.TEARDOWN
        and detail.error.code == hs.errors.ErrorCode.FAILURE
        and "secondary cleanup sentinel" in detail.error.message
        for detail in details
    )


# REQ: PY-API-030
def test_schema_result_materialization_failure_preserves_native_capture_without_rerunning(
    monkeypatch,
):
    calls = []

    class SchemaSource(hs.Operator):
        out = hs.Output("out", control_payload.ControlPayload)

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

        def compute(self, context):
            calls.append("compute")
            self.out.emit(control_payload.ControlPayload(sequence=7, gain=2.5, labels=["retained"]))

    with OperatorTestHarness(SchemaSource) as harness:
        result = harness.run()
    assert result.ok
    assert calls == ["compute"]

    def fail_materialization(fields):
        assert fields["sequence"] == 7
        raise ValueError("result materialization sentinel")

    # Conversion happens when a captured result is inspected. A provider failure here must
    # neither destroy the native capture nor trigger execution to reconstruct the result.
    with monkeypatch.context() as patch:
        patch.setattr(
            control_payload.ControlPayload,
            "_from_holoscan_schema",
            staticmethod(fail_materialization),
        )
        with pytest.raises(ValueError, match="result materialization sentinel"):
            result.values("out")
        assert result.ok
        assert result.diagnostics == []
        assert calls == ["compute"]
    recovered = result.value("out")
    assert recovered.sequence == 7
    assert recovered.labels == ["retained"]
    assert calls == ["compute"]


# REQ: PY-API-030
def test_python_constructor_callbacks_reject_before_mutation_and_preserve_configuration():
    class SharedChannel:
        """Explicitly share the test provider across the fixture's parameter copies."""

        def __init__(self):
            self.channel = provider.TestChannel(42)

        def __deepcopy__(self, memo):
            return self

        def __holoscan_constructor_handle__(self, type_name):
            return self.channel.__holoscan_constructor_handle__(type_name)

    receiver = hs.registry.resolve("holoscan.test.py.CallbackReceiver", kind="operator")
    params = {"data_channel": SharedChannel()}
    calls = []
    callback_params = {**params, "device_start": lambda: calls.append("started")}
    before = provider.receiver_constructions()
    token = "PYTHON_CONSTRUCTOR_CALLBACK_OWNER_REQUIRED"
    with pytest.raises(hs.errors.NotSupportedError, match=token):
        OperatorTestHarness(receiver, params=callback_params)
    assert provider.receiver_constructions() == before

    with OperatorTestHarness(receiver, params=params) as harness:
        before = provider.receiver_constructions()
        with pytest.raises(hs.errors.NotSupportedError, match=token):
            harness.set_params(callback_params)
        assert provider.receiver_constructions() == before
        attempt = harness.run_async()
        attempt.request_stop()
        result = attempt.finish(check=False)
        assert result.termination == hs.RunTermination.STOPPED
        assert result.diagnostics == []
        assert result.values("output") == []
    assert calls == []
