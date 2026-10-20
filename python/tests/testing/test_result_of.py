# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed access to the original joined failure result without executing work."""

import pytest
from observation import OBSERVATION_MS
from test_operator_test_harness import Fail

from holoscan.errors import HoloscanError
from holoscan.testing import FrameRelease, OperatorTestHarness, TestRunOptions, result_of


# REQ: PY-API-029
@pytest.mark.parametrize("managed", [False, True])
def test_result_of_preserves_failed_attempt_identity(managed):
    with OperatorTestHarness(Fail) as harness:
        harness.push("in", 1)
        if managed:
            with harness.run_async() as running:
                with pytest.raises(HoloscanError) as caught:
                    running.finish()
                result = running.finish(check=False)
                with pytest.raises(HoloscanError) as repeated:
                    running.finish()
                assert result_of(repeated.value) is result
        else:
            with pytest.raises(HoloscanError) as caught:
                harness.run()
            result = caught.value.result
    assert result_of(caught.value) is result
    assert result_of(caught.value) is result_of(caught.value)
    assert not result.ok
    assert "intentional operator test failure" in result.describe_failure()
    unrelated = RuntimeError("wrapper")
    unrelated.__cause__ = caught.value
    assert result_of(unrelated) is None


# REQ: PY-API-029
@pytest.mark.parametrize(
    "error", [RuntimeError("ordinary"), HoloscanError("ordinary"), KeyboardInterrupt()]
)
def test_result_of_ignores_absent_or_unrelated_attributes(error):
    assert result_of(error) is None
    for value in (None, 3, object()):
        error.result = value
        assert result_of(error) is None


# REQ: PY-API-029
def test_result_of_preserves_partial_output_without_reexecuting():
    calls = []

    class FailsSecond(Fail):
        def compute(self, context):
            value = self.inp.receive_data()
            calls.append(value)
            if value == 2:
                raise RuntimeError("second frame failed")
            self.out.emit(value)

    with OperatorTestHarness(FailsSecond) as harness:
        harness.push_frame({"in": 1})
        harness.push_frame({"in": 2})
        with harness.run_async(TestRunOptions(frames=FrameRelease.MANUAL)) as running:
            running.release_frame(timeout_ms=OBSERVATION_MS)
            assert running.wait_for_output("out", 1, timeout_ms=OBSERVATION_MS)
            running.release_frame(timeout_ms=OBSERVATION_MS)
            with pytest.raises(HoloscanError) as caught:
                running.finish()
            result = result_of(caught.value)
            assert result is not None
            assert result is running.finish(check=False)
    assert result.values("out") == [1]
    assert "second frame failed" in result.describe_failure()
    assert "compute" in result.describe_failure()
    assert result_of(caught.value) is result
    assert calls == [1, 2]


# REQ: PY-API-029
def test_result_of_ignores_inherited_and_computed_results():
    with OperatorTestHarness(Fail) as harness:
        harness.push("in", 1)
        failed = harness.run(check=False)

    class InheritedResultError(Exception):
        result = failed

    class ComputedResultError(Exception):
        @property
        def result(self):
            raise AssertionError("inspection must not execute a result descriptor")

    assert result_of(InheritedResultError()) is None
    assert result_of(ComputedResultError()) is None
