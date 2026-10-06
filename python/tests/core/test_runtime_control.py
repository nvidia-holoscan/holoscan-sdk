# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for run-scoped parameter and lifecycle control."""

import datetime
import enum
import gc
import os
import signal
import threading
import time

import holoscan_test_support
import pytest

import holoscan as hs


def _control_compile_options():
    """Return intentionally roomy finite capacities for the live-control test."""
    parameters = hs.params.ParameterCompileOptions()
    parameters.runtime_updates = hs.params.ParamRuntimeUpdatePolicy.ENABLED_IF_ADMITTED
    parameters.max_event_cursors = 2
    lifecycle = hs.lifecycle.LifecycleTransactionOptions(
        max_pending_requests=8,
        max_pending_transactions=8,
        max_receipts=16,
        max_fenced_operators=8,
        max_candidate_generations=8,
        max_events=128,
        max_audit_records=32,
        max_quiesce_records=32,
        max_rollback_records=64,
    )
    return hs.CompileOptions(parameters=parameters, lifecycle_transactions=lifecycle)


RuntimeControlMode = enum.IntEnum("RuntimeControlMode", {"automatic": 0, "manual": 1})


# REQ: PY-API-020
def test_live_parameter_and_lifecycle_control_of_registered_cpp_operator():
    # Key setup: Run a clock-driven registered C++ operator with runtime updates and restart
    # authority enabled.
    #
    #   (1 ms clock) --> probe
    holoscan_test_support._register_runtime_control_probe_for_test()
    holoscan_test_support._reset_runtime_control_observation()
    graph = hs.Graph("runtime-control")
    probe = graph.op("holoscan.tests.RuntimeControlProbe", "probe")
    graph.lifecycle().allow_external_restart(probe)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    plan = hs.compile(graph, _control_compile_options())
    plan.require_no_errors()

    session = hs.run_async(plan)
    try:
        assert holoscan_test_support._wait_for_runtime_control_generation(0, 5.0)
        policy = hs.params.ParamAccessPolicy(
            permissions=(
                hs.params.ParamPermission.READ
                | hs.params.ParamPermission.UPDATE
                | hs.params.ParamPermission.AWAIT
                | hs.params.ParamPermission.OBSERVE
            ),
            max_event_cursors=1,
        )
        control = session.parameter_control(policy)
        events = control.events()
        assert not hasattr(events, "__next__")
        initial = control.read("probe/scalar")
        assert initial.value == 1
        assert initial.generation == 0
        assert control.read("probe/interval").value.nanoseconds == 1_000_000

        # Key assertion: Equal runtime string, sequence, object, and empty-optional patches retain
        # generation zero instead of publishing a spurious successor.
        runtime_no_change = control.submit(
            [
                hs.params.ParamPatch("probe/runtime_label", "initial"),
                hs.params.ParamPatch("probe/runtime_levels", [1]),
                hs.params.ParamPatch("probe/runtime_sensor", {"channels": 1}),
                hs.params.ParamPatch("probe/runtime_optional_count", None),
            ]
        )
        assert runtime_no_change.committed
        assert runtime_no_change.status is hs.params.ParamUpdateStatus.ACCEPTED_NO_CHANGE
        assert runtime_no_change.target_generation == 0
        runtime_no_change_terminal = control.await_applied(
            runtime_no_change.update_sequence, timeout=5.0
        )
        assert runtime_no_change_terminal.state is hs.params.ParamUpdateState.ACCEPTED_NO_CHANGE

        before_no_change = holoscan_test_support._runtime_control_observation()
        no_change = control.submit(
            [
                hs.params.ParamPatch("probe/scalar", 1),
                hs.params.ParamPatch("probe/levels", [1]),
                hs.params.ParamPatch("probe/optional_count", None),
                hs.params.ParamPatch("probe/sensor", {"channels": 1}),
                hs.params.ParamPatch("probe/backup_sensor", None),
                hs.params.ParamPatch("probe/label", "initial"),
            ]
        )
        assert no_change.committed
        assert no_change.status is hs.params.ParamUpdateStatus.ACCEPTED_NO_CHANGE
        no_change_terminal = control.await_applied(no_change.update_sequence, timeout=5.0)
        assert no_change_terminal.state is hs.params.ParamUpdateState.ACCEPTED_NO_CHANGE
        assert no_change_terminal.target_generation == 0
        after_no_change = holoscan_test_support._runtime_control_observation()
        assert after_no_change["generation"] == 0
        assert after_no_change["start_count"] == before_no_change["start_count"]

        update = control.submit(
            [
                hs.params.ParamPatch("probe/scalar", 5),
                hs.params.ParamPatch("probe/interval", datetime.timedelta(microseconds=2500)),
                hs.params.ParamPatch("probe/levels", [2, 3]),
                hs.params.ParamPatch("probe/optional_count", None),
                hs.params.ParamPatch("probe/mode", RuntimeControlMode.manual),
                hs.params.ParamPatch("probe/modes", ["manual", "automatic"]),
                hs.params.ParamPatch("probe/sensor", {"channels": 3}),
                hs.params.ParamPatch("probe/backup_sensor", {"channels": 4}),
                hs.params.ParamPatch("probe/label", "live"),
            ]
        )
        assert update.committed
        assert update.status is hs.params.ParamUpdateStatus.ACCEPTED
        applied = control.await_applied(update.update_sequence, timeout=5.0)
        assert applied.state in (
            hs.params.ParamUpdateState.PUBLISHED,
            hs.params.ParamUpdateState.OBSERVED,
        )
        assert holoscan_test_support._wait_for_runtime_control_generation(1, 5.0)

        observation = holoscan_test_support._runtime_control_observation()
        assert observation["generation"] == 1
        assert observation["scalar"] == 5
        assert observation["interval_ns"] == 2_500_000
        assert observation["level_count"] == 2
        assert observation["level_sum"] == 5
        assert observation["optional_count"] == -1
        assert observation["mode"] == 1
        assert observation["mode_count"] == 2
        assert observation["sensor_channels"] == 3
        assert observation["backup_channels"] == 4
        assert observation["label_is_live"] is True
        assert observation["start_count"] == before_no_change["start_count"] + 1

        optional_update = control.submit(
            [
                hs.params.ParamPatch("probe/optional_count", 9, expected_generation=1),
                hs.params.ParamPatch("probe/interval", hs.time.ms(3), expected_generation=1),
            ]
        )
        assert optional_update.committed
        control.await_applied(optional_update.update_sequence, timeout=5.0)
        assert (
            control.await_applied(optional_update.update_sequence, timeout=0.0).update_sequence
            == optional_update.update_sequence
        )
        assert holoscan_test_support._wait_for_runtime_control_generation(2, 5.0)
        assert holoscan_test_support._runtime_control_observation()["optional_count"] == 9
        assert control.read("probe/interval").value.nanoseconds == 3_000_000

        rejected = control.submit(
            [
                hs.params.ParamPatch("probe/scalar", 8),
                hs.params.ParamPatch("probe/output_path", "not-supported"),
            ]
        )
        assert not rejected.committed
        assert rejected.status is hs.params.ParamUpdateStatus.TYPE_MISMATCH
        assert rejected.failed_patch == 1
        assert control.read("probe/scalar").value == 5
        with pytest.raises(hs.errors.NotSupportedError) as unsupported:
            control.read("probe/label")
        assert unsupported.value.code is hs.errors.ErrorCode.NOT_SUPPORTED
        observed_events = events.drain()
        assert any(event.update_sequence == update.update_sequence for event in observed_events)
        assert any(event.operator_instance.run_generation > 0 for event in observed_events)

        lifecycle_policy = hs.lifecycle.LifecycleAccessPolicy(
            targets=[probe], from_stages=[hs.lifecycle.LifecycleStage.CONFIGURE]
        )
        lifecycle = session.lifecycle_control(lifecycle_policy)
        receipt = lifecycle.request_restart(probe)
        terminal = lifecycle.await_terminal(receipt, timeout=5.0)
        assert (
            lifecycle.await_terminal(receipt, timeout=0.0).receipt.receipt_id == receipt.receipt_id
        )

        # Key assertion: One Python capability atomically updates every supported C++ field,
        # rejects an unsupported mixed batch without partial application, and restarts its target.
        assert terminal.state is hs.lifecycle.LifecycleTransactionState.PUBLISHED
        assert terminal.receipt.receipt_id == receipt.receipt_id
        assert holoscan_test_support._runtime_control_observation()["start_count"] >= 2
    finally:
        session.request_stop()
        assert session.wait(timeout=5.0)

    assert session.diagnostics == []


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires POSIX fork")
def test_inherited_parameter_event_cursor_destruction_is_inert():
    """A child must not run native cursor cleanup against inherited process locks."""
    holoscan_test_support._register_runtime_control_probe_for_test()
    holoscan_test_support._reset_runtime_control_observation()
    graph = hs.Graph("runtime-control-cursor-fork")
    probe = graph.op("holoscan.tests.RuntimeControlProbe", "probe")
    graph.lifecycle().allow_external_restart(probe)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    plan = hs.compile(graph, _control_compile_options())
    plan.require_no_errors()

    session = hs.run_async(plan)
    try:
        control = session.parameter_control(
            hs.params.ParamAccessPolicy(
                permissions=hs.params.ParamPermission.OBSERVE,
                max_event_cursors=1,
            )
        )
        events = control.events()
        child = os.fork()
        if child == 0:
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            signal.alarm(5)
            del events
            gc.collect()
            os._exit(0)

        deadline = time.monotonic() + 10.0
        status = None
        while time.monotonic() < deadline:
            waited, child_status = os.waitpid(child, os.WNOHANG)
            if waited == child:
                status = child_status
                break
            time.sleep(0.01)
        if status is None:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)
            pytest.fail("child cursor destruction did not complete")
        assert os.waitstatus_to_exitcode(status) == 0

        with pytest.raises(hs.errors.ResourceExhaustedError):
            control.events()
        del events
        gc.collect()
        replacement = control.events()
        assert replacement.drain() == []
    finally:
        session.request_stop()
        assert session.wait(timeout=5.0)

    assert session.diagnostics == []


# REQ: PY-API-020
def test_lifecycle_access_policy_rejects_non_operator_targets_eagerly():
    # Key setup: Supply an invalid target while constructing a lifecycle access policy.

    # Key assertion: The lifecycle-specific type error is raised at the offending constructor.
    with pytest.raises(TypeError, match="LIFECYCLE_CONTROL_OPERATOR_REQUIRED"):
        hs.lifecycle.LifecycleAccessPolicy(
            targets=["probe"], from_stages=[hs.lifecycle.LifecycleStage.CONFIGURE]
        )


# REQ: PY-API-020
def test_immediate_control_mint_with_native_operator_does_not_hold_the_gil():
    # Key setup: Mint controls immediately after launching a graph whose worker must acquire the
    # GIL to reconstruct its native Python operator.
    #
    #   (1 ms clock) -+-> probe[Python]
    #                 +-> params[C++]
    holoscan_test_support._register_runtime_control_probe_for_test()
    holoscan_test_support._reset_runtime_control_observation()

    class NativeProbe(hs.Operator):
        """Keep a hosted run alive while exposing one authorized restart target."""

        configured = threading.Event()
        computed = threading.Event()
        run_ref = None

        def setup(self, spec):
            spec.lifecycle().stage(hs.lifecycle.LifecycleStage.CONFIGURE, "configure")

        def configure(self, context):
            type(self).run_ref = context.restart_target("probe")
            type(self).configured.set()

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

        def compute(self, context):
            type(self).computed.set()

    NativeProbe.configured.clear()
    NativeProbe.computed.clear()
    NativeProbe.run_ref = None
    graph = hs.Graph("runtime-control-native-startup")
    probe = graph.op(NativeProbe, "probe")
    graph.op("holoscan.tests.RuntimeControlProbe", "params")
    graph.lifecycle().allow_self_restart(probe)
    graph.lifecycle().allow_external_restart(probe)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    session = hs.run_async(hs.compile(graph, _control_compile_options()))
    try:
        parameter_control = session.parameter_control(
            hs.params.ParamAccessPolicy(permissions=hs.params.ParamPermission.READ)
        )
        assert NativeProbe.configured.wait(timeout=5.0)
        assert NativeProbe.computed.wait(timeout=5.0)
        lifecycle_control = session.lifecycle_control(
            hs.lifecycle.LifecycleAccessPolicy(
                targets=[probe], from_stages=[hs.lifecycle.LifecycleStage.CONFIGURE]
            )
        )
        receipt = lifecycle_control.request_restart(NativeProbe.run_ref)
        terminal = lifecycle_control.await_terminal(receipt, timeout=5.0)

        # Key assertion: Startup completes without a GIL cycle, and the callback-resolved run
        # reference can be consumed directly by lifecycle control.
        assert parameter_control is not None
        assert terminal.state is hs.lifecycle.LifecycleTransactionState.PUBLISHED
    finally:
        session.request_stop()
        assert session.wait(timeout=5.0)

    assert session.diagnostics == []


# REQ: PY-API-020
def test_runtime_callbacks_reject_general_control_operations_without_aborting():
    # Key setup: Let a native Python callback attempt parameter-capability minting and a lifecycle
    # restart against a concurrently running registered C++ target.
    #
    #   (1 ms clock) -+-> guard[Python]
    #                 +-> probe[C++]
    holoscan_test_support._register_runtime_control_probe_for_test()
    state = {}
    callback_finished = threading.Event()

    graph = hs.Graph("runtime-control-callback-guard")
    probe = graph.op("holoscan.tests.RuntimeControlProbe", "probe")
    graph.lifecycle().allow_external_restart(probe)

    parameter_policy = hs.params.ParamAccessPolicy(
        permissions=hs.params.ParamPermission.READ | hs.params.ParamPermission.UPDATE
    )
    lifecycle_policy = hs.lifecycle.LifecycleAccessPolicy(
        targets=[probe], from_stages=[hs.lifecycle.LifecycleStage.CONFIGURE]
    )

    class CallbackGuard(hs.Operator):
        """Exercise general control APIs from a forbidden runtime callback context.

        Used by test_runtime_callbacks_reject_general_control_operations_without_aborting.
        """

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1)))

        def compute(self, context):
            if callback_finished.is_set():
                return
            deadline = time.monotonic() + 5.0
            while "session" not in state and time.monotonic() < deadline:
                time.sleep(0.001)
            if "session" not in state:
                state["setup_error"] = RuntimeError("session publication timed out")
                callback_finished.set()
                return
            try:
                state["session"].parameter_control(parameter_policy)
            except Exception as error:  # noqa: BLE001 - the exact public type is asserted below
                state["parameter_error_type"] = type(error)
            try:
                lifecycle = state["session"].lifecycle_control(lifecycle_policy)
                lifecycle.request_restart(probe)
            except Exception as error:  # noqa: BLE001 - the exact public type is asserted below
                state["lifecycle_error_type"] = type(error)
            callback_finished.set()

    graph.op(CallbackGuard, "guard")
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    plan = hs.compile(graph, _control_compile_options())
    assert plan.ok, [diagnostic.token for diagnostic in plan.diagnostics]

    session = hs.run_async(plan)
    state["session"] = session
    try:
        assert callback_finished.wait(timeout=5.0)
        state.pop("session", None)

        # Key assertion: Both Core callback guards surface as typed Python exceptions and the
        # runtime remains responsive enough to stop and join normally.
        assert "setup_error" not in state
        assert issubclass(state["parameter_error_type"], hs.errors.InvalidArgumentError)
        assert issubclass(
            state["lifecycle_error_type"], hs.errors.LifecycleInvalidCallingContextError
        )
    finally:
        state.pop("session", None)
        session.request_stop()
        assert session.wait(timeout=5.0)

    assert session.diagnostics == []
