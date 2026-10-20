# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise ordinary test-author imports, staging and configuration failures."""

import subprocess
import sys

import pytest
from observation import OBSERVATION_MS

import holoscan as hs
import holoscan.testing as testing


class Echo(hs.Operator):
    inp = hs.Input("in", hs.pyobject)
    out = hs.Output("out", hs.pyobject)

    def compute(self, context):
        self.out.emit(self.inp.receive_data())


# REQ: PY-API-026
def test_public_test_helpers_do_not_enter_pytest_collection(tmp_path):
    names = [name for name in testing.__all__ if name.startswith("Test")]
    source = tmp_path / "test_imports.py"
    source.write_text(
        f"from holoscan.testing import {', '.join(names)}\ndef test_example():\n    assert True\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "-W",
            "error::pytest.PytestCollectionWarning",
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 test collected" in result.stdout


# REQ: PY-API-030
def test_single_output_name_selects_one_port():
    with testing.OperatorTestHarness(Echo, outputs="out") as harness:
        harness.push("in", "payload")
        assert harness.run().value("out") == "payload"


# REQ: PY-API-030
def test_explicit_subset_captures_only_selected_output():
    class TwoOutputs(Echo):
        unused = hs.Output("unused", hs.pyobject)

    with testing.OperatorTestHarness(TwoOutputs, outputs=["out"]) as harness:
        harness.push("in", "selected")
        result = harness.run()
    assert result.value("out") == "selected"
    with pytest.raises(ValueError, match="unused"):
        result.values("unused")


# REQ: PY-API-030
@pytest.mark.parametrize("values", ["abc", b"abc", bytearray(b"abc"), {"key": 1}])
def test_ambiguous_batches_reject_without_changing_pending_input(values):
    with testing.OperatorTestHarness(Echo) as harness:
        harness.push("in", "before")
        with pytest.raises(TypeError, match="push"):
            harness.push_many("in", values)
        assert harness.run().values("out") == ["before"]


# REQ: PY-API-031
def test_output_wait_requires_an_explicit_timeout():
    with testing.OperatorTestHarness(Echo) as harness:
        harness.push_frame({"in": "value"})
        options = testing.TestRunOptions(frames=testing.FrameRelease.MANUAL)
        with harness.run_async(options) as running:
            assert running.is_running
            with pytest.raises(TypeError, match="timeout_ms"):
                running.wait_for_output("out", 1)
            assert not running.wait_for_output("out", 1, timeout_ms=0)
            running.release_frame(timeout_ms=OBSERVATION_MS)
            assert running.wait_for_output("out", 1, timeout_ms=OBSERVATION_MS)
            assert running.finish().value("out") == "value"
            assert not running.is_running


# REQ: PY-API-030
def test_parameter_update_rejects_a_removed_flow_selector_atomically():
    class Configured(Echo):
        extra = hs.Output("extra", hs.pyobject)
        enabled = hs.Param(True)

        def setup(self, spec):
            spec.input(self.inp)
            spec.output(self.out)
            if self.enabled:
                spec.output(self.extra)

        def compute(self, context):
            value = self.inp.receive_data()
            self.out.emit(value)
            if self.enabled:
                self.extra.emit(value)

    with testing.OperatorTestHarness(Configured) as harness:
        harness.set_output_flow("extra", queue_depth=2)
        harness.push("in", "preserved")
        with pytest.raises(hs.errors.NotFoundError, match="extra"):
            harness.set_params({"enabled": False})
        result = harness.run()
        assert result.value("out") == "preserved"
        assert result.value("extra") == "preserved"


# REQ: PY-API-030
def test_singular_and_capture_policy_errors_name_the_output():
    with testing.OperatorTestHarness(Echo) as harness:
        harness.push_many("in", [1, 2])
        result = harness.run()
    with pytest.raises(ValueError, match="out.*2.*values"):
        result.value("out")
    with pytest.raises(ValueError, match="out.*values"):
        result.snapshot("out")


# REQ: PY-API-030
@pytest.mark.parametrize("timestamp", [True, 1.5, "1"])
def test_timestamps_reject_non_integer_coordinates(timestamp):
    with pytest.raises(TypeError, match="integer"):
        testing.sample_at("value", capture_timestamp_ns=timestamp)


# REQ: PY-API-030
@pytest.mark.parametrize("timestamp", [-(2**63) - 1, 2**63])
def test_timestamps_reject_out_of_range_coordinates(timestamp):
    with pytest.raises(OverflowError, match="64-bit"):
        testing.sample_at("value", capture_timestamp_ns=timestamp)


# REQ: PY-API-030
def test_closed_builder_rejects_staging_with_a_specific_message():
    harness = testing.OperatorTestHarness(Echo)
    harness.close()
    with pytest.raises(RuntimeError, match="harness is closed"):
        harness.push("in", 1)


# REQ: PY-API-030
@pytest.mark.parametrize("setter", ["input_queue_depth", "input_flow"])
def test_removed_input_override_rejects_before_staging(setter):
    class Configured(Echo):
        enabled = hs.Param(True)

        def setup(self, spec):
            if self.enabled:
                spec.input(self.inp)
            spec.output(self.out)

    with testing.OperatorTestHarness(Configured) as harness:
        override = getattr(harness, "set_" + setter)
        if setter == "input_queue_depth":
            override("in", 2)
        else:
            override("in", queue_depth=2)
        with pytest.raises(hs.errors.NotFoundError, match="in"):
            harness.set_params({"enabled": False})
        harness.push("in", "still usable")
        assert harness.run().value("out") == "still usable"
        getattr(harness, "clear_" + setter)("in")
        harness.set_params({"enabled": False})


# REQ: PY-API-030
@pytest.mark.parametrize(
    "setter,port",
    [
        ("set_input_tensor_spec", "in"),
        ("set_output_tensor_spec", "out"),
        ("set_input_source_tensor_spec", "in"),
        ("set_input_source_device", "in"),
        ("set_output_device", "out"),
    ],
)
def test_removed_tensor_override_rejects_atomically(setter, port):
    class Configured(hs.Operator):
        inp = hs.Input("in", hs.payload.Tensor)
        out = hs.Output("out", hs.payload.Tensor)
        enabled = hs.Param(True)

        def setup(self, spec):
            if self.enabled:
                spec.input(self.inp)
                spec.output(self.out).produces_tensor(
                    hs.tensor.OutputSpec(
                        storage="runtime_pool",
                        rank=1,
                        dtype=hs.tensor.DLDataType(1, 8, 1),
                        memory_kind=hs.tensor.MemoryKind.HOST,
                        bounds=hs.tensor.TensorBounds(max_byte_span=4),
                    )
                )

        def compute(self, context):
            pass

    with testing.OperatorTestHarness(Configured) as harness:
        override = getattr(harness, setter)
        if setter.endswith("device"):
            value = hs.execution.DeviceId(0)
        elif setter == "set_input_tensor_spec":
            value = hs.tensor.InputSpec(rank=1, dtype=hs.tensor.DLDataType(1, 8, 1))
        else:
            value = hs.tensor.OutputSpec(
                storage="runtime_pool",
                rank=1,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                memory_kind=hs.tensor.MemoryKind.HOST,
                bounds=hs.tensor.TensorBounds(max_byte_span=4),
            )
        override(port, value)
        with pytest.raises(hs.errors.NotFoundError, match=port):
            harness.set_params({"enabled": False})
        # The previous declaration and override remain addressable after rejection.
        override(port, None)
        harness.set_params({"enabled": False})
