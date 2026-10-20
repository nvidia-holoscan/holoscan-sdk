# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed attempts own native work and admit joins only outside Runtime callbacks."""

import gc
import threading
import time

import pytest
from observation import CALLBACK_RELEASE_SECONDS, OBSERVATION_MS, OBSERVATION_SECONDS

import holoscan as hs
from holoscan.testing import ControlledClockRun, FrameRelease, OperatorTestHarness
from holoscan.testing import TestRunOptions as RunOptions


class Echo(hs.Operator):
    inp = hs.Input("in", hs.payload.uint32)
    out = hs.Output("out", hs.payload.uint32)

    def compute(self, context):
        self.out.emit(self.inp.receive_data())


def manual(harness):
    return harness.run_async(RunOptions(frames=FrameRelease.MANUAL, expect_stop=True))


# REQ: PY-API-031
def test_owner_survives_builder_and_finish_is_cached():
    harness = OperatorTestHarness(Echo)
    harness.push_frame({"in": 7})
    running = manual(harness)
    harness.close()
    del harness
    gc.collect()
    running.release_frame(timeout_ms=OBSERVATION_MS)
    assert running.wait_for_output("out", 1, timeout_ms=OBSERVATION_MS)
    running.request_stop()
    result = running.finish()
    assert result.value("out") == 7
    assert running.finish() is result
    running.close()
    assert result.value("out") == 7


# REQ: PY-API-031
def test_active_attempt_rejects_changes_and_releases_builder_after_join():
    with OperatorTestHarness(Echo) as harness:
        harness.push_frame({"in": 8})
        with manual(harness) as running:
            with pytest.raises(RuntimeError, match="active attempt"):
                harness.push("in", 9)
            with pytest.raises(RuntimeError, match="active attempt"):
                harness.run_async()
            running.release_frame(timeout_ms=OBSERVATION_MS)
            assert running.wait_for_output("out", 1, timeout_ms=OBSERVATION_MS)
            assert running.output_count("out") == 1
        harness.push("in", 10)
        assert harness.run().value("out") == 10


# REQ: PY-API-031
def test_callback_join_rejected_and_gil_released_during_outer_join():
    ready = threading.Event()
    owners = []
    errors = []

    class Callback(Echo):
        def compute(self, context):
            assert ready.wait(OBSERVATION_SECONDS)
            try:
                owners[0].finish()
            except hs.HoloscanError as error:
                errors.append(error)
            self.out.emit(self.inp.receive_data())
            owners[0].request_stop()

    with OperatorTestHarness(Callback) as harness:
        harness.push_frame({"in": 3})
        running = manual(harness)
        owners.append(running)
        running.release_frame(timeout_ms=OBSERVATION_MS)
        ready.set()
        result = running.finish()
        assert errors
        assert result.ok
        owners.clear()
        errors.clear()


# REQ: PY-API-031
def test_failed_managed_finish_raises_with_same_result():
    class Fail(Echo):
        def compute(self, context):
            raise ValueError("managed callback failure")

    with OperatorTestHarness(Fail) as harness:
        harness.push("in", 1)
        running = harness.run_async()
        result = running.finish(check=False)
        assert not result.ok
        with pytest.raises(hs.HoloscanError) as raised:
            running.finish()
        assert raised.value.result is result
        assert "managed callback failure" in result.describe_failure()


# REQ: PY-API-031
def test_last_owner_finalized_inside_callback_uses_shared_cleanup():
    completed = threading.Event()
    owners = []

    class Finalizer(Echo):
        def compute(self, context):
            owner = owners.pop()
            del owner
            gc.collect()
            completed.set()

    with OperatorTestHarness(Finalizer) as harness:
        harness.push_frame({"in": 1})
        owners.append(manual(harness))
        owners[0].release_frame(timeout_ms=OBSERVATION_MS)
        assert completed.wait(OBSERVATION_SECONDS)
        # The reaper releases the native attempt gate only after complete retirement.
        deadline = time.monotonic() + OBSERVATION_SECONDS
        while True:
            try:
                harness.set_params({})
                break
            except RuntimeError as error:
                if str(error) != "operator test already has an active attempt":
                    raise
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                threading.Event().wait(min(0.01, remaining))


# REQ: PY-API-031
def test_blocking_runs_from_runtime_callbacks_reject_before_admission():
    errors = []

    class Nested(Echo):
        def compute(self, context):
            with OperatorTestHarness(Echo) as nested:
                nested.push("in", 1)
                for run in (
                    nested.run,
                    lambda: nested.run_controlled_clock(ControlledClockRun(tick_ns=1)),
                ):
                    try:
                        run()
                    except hs.HoloscanError as error:
                        errors.append(str(error))
            self.out.emit(self.inp.receive_data())

    with OperatorTestHarness(Nested) as harness:
        harness.push("in", 2)
        assert harness.run().value("out") == 2
    assert len(errors) == 2
    assert all("CALLBACK" in error for error in errors)


# REQ: PY-API-031
def test_concurrent_finish_and_close_join_one_attempt():
    with OperatorTestHarness(Echo) as harness:
        harness.push_frame({"in": 6})
        running = manual(harness)
        running.release_frame(timeout_ms=OBSERVATION_MS)
        assert running.wait_for_output("out", 1, timeout_ms=OBSERVATION_MS)
        results = []
        worker = threading.Thread(target=lambda: results.append(running.finish()))
        worker.start()
        running.close()
        worker.join(OBSERVATION_SECONDS)
        assert not worker.is_alive()
        assert results[0] is running.finish()
        assert results[0].value("out") == 6


# REQ: PY-API-031
def test_builder_context_preserves_primary_error_during_concurrent_run():
    entered = threading.Event()
    release = threading.Event()

    class Waiting(Echo):
        def compute(self, context):
            entered.set()
            assert release.wait(CALLBACK_RELEASE_SECONDS)
            super().compute(context)

    harness = OperatorTestHarness(Waiting)
    harness.push("in", 1)
    results = []
    worker = threading.Thread(target=lambda: results.append(harness.run()))
    worker.start()
    try:
        assert entered.wait(OBSERVATION_SECONDS)
        with pytest.raises(ValueError, match="primary body failure") as caught, harness:
            raise ValueError("primary body failure")
        assert any("cleanup also failed" in note for note in caught.value.__notes__)
    finally:
        release.set()
        worker.join(OBSERVATION_SECONDS)
        harness.close()
    assert not worker.is_alive()
    assert results[0].value("out") == 1
