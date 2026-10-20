# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python callables passed as SDK callback constructor handles."""

import gc
import importlib.util
import os
import subprocess
import sys
import textwrap
import threading
import weakref

import holoscan_constructor_handle_provider as provider
import holoscan_test_support  # noqa: F401
import pytest

import holoscan as hs

provider.register_rows()
CallbackReceiver = hs.registry.resolve("holoscan.test.py.CallbackReceiver", kind="operator")
Sink = hs.registry.resolve("holoscan.test.Sink", kind="operator")


class PythonSensor:
    """A sensor implemented entirely in Python, like the HSB Imx274Cam and Vb1940Cam classes."""

    def __init__(self, fail_on_start=False):
        self.fail_on_start = fail_on_start
        self.started = threading.Event()
        self.stopped = threading.Event()
        self.start_thread = None

    def start(self):
        self.start_thread = threading.get_ident()
        self.started.set()
        if self.fail_on_start:
            raise ValueError("sensor refused to start")

    def stop(self):
        self.stopped.set()


def _graph(name, sensor):
    graph = hs.Graph(name)
    receiver = CallbackReceiver(
        graph,
        "receiver",
        data_channel=provider.TestChannel(10),
        device_start=sensor.start,
        device_stop=sensor.stop,
    )
    sink = Sink(graph, "sink")
    graph.add_flow(receiver.output, sink.in_)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    return graph, receiver


# REQ: PY-API-025
def test_callback_inputs_are_described_and_rendered():
    # Key setup: Author a receiver whose sensor actions are Python bound methods.
    sensor = PythonSensor()
    graph, receiver = _graph("callback-repr", sensor)

    # Key assertion: The callback inputs are typed in discovery, and repr marks their Python
    # dependence without calling or showing the callables.
    handles = CallbackReceiver.constructor_handles
    assert handles["device_start"].type_name == "holoscan.VoidCallback"
    assert not handles["device_start"].required
    rendered = repr(receiver)
    assert "device_start=<holoscan.VoidCallback: Python callable, process-local>" in rendered
    assert "PythonSensor" not in rendered
    assert not sensor.started.is_set()


# REQ: PY-API-025
def test_receiver_lifecycle_invokes_python_callbacks_under_the_runtime():
    # Key setup: Run the graph so the receiver's start and stop invoke the Python sensor.
    #
    #   receiver --> sink
    sensor = PythonSensor()
    graph, _ = _graph("callback-run", sensor)
    plan = hs.compile(graph)
    plan.require_no_errors()

    session = hs.run_async(plan)
    try:
        assert sensor.started.wait(timeout=5.0)
    finally:
        session.request_stop()
        assert session.wait(timeout=5.0)

    # Key assertion: Both callbacks ran through the receiver's lifecycle, and the run completed.
    assert sensor.stopped.is_set()
    assert session.termination in (hs.RunTermination.REQUESTED_COMPLETED, hs.RunTermination.STOPPED)


# REQ: PY-API-025
def test_python_exception_fails_the_run_with_a_bounded_token(monkeypatch):
    # Key setup: Let the Python sensor raise from its start callback.
    reported = []
    monkeypatch.setattr(sys, "unraisablehook", reported.append)
    sensor = PythonSensor(fail_on_start=True)
    graph, _ = _graph("callback-failure", sensor)
    plan = hs.compile(graph)

    session = hs.run_async(plan)
    try:
        assert session.wait(timeout=5.0)
    finally:
        session.request_stop()
        session.wait(timeout=5.0)

    # Key assertion: The run fails with the SDK token, and the Python traceback is reported
    # through sys.unraisablehook instead of escaping into C++ lifecycle handling.
    assert session.termination == hs.RunTermination.FAILED
    assert "CONSTRUCTOR_CALLBACK_FAILED" in session.summary()
    assert any(isinstance(item.exc_value, ValueError) for item in reported)


# REQ: PY-API-025
def test_callback_copies_do_not_touch_python_reference_counts():
    # Key setup: Hold the bound methods so their reference counts can be observed.
    sensor = PythonSensor()
    start, stop = sensor.start, sensor.stop
    graph = hs.Graph("callback-refcounts")
    receiver = CallbackReceiver(
        graph,
        "receiver",
        data_channel=provider.TestChannel(11),
        device_start=start,
        device_stop=stop,
    )
    sink = Sink(graph, "sink")
    graph.add_flow(receiver.output, sink.in_)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    authored = sys.getrefcount(start)

    plans = [hs.compile(graph), hs.compile(graph)]

    # Key assertion: Compilation replays and copies the factory without touching reference counts;
    # each Plan object holds exactly one reference of its own, and the callables are released once
    # the Graph and plans are gone.
    assert all(plan.ok for plan in plans)
    assert sys.getrefcount(start) == authored + len(plans)
    sensor_ref = weakref.ref(sensor)
    del sensor, start, stop, receiver, sink
    gc.collect()
    assert sensor_ref() is not None
    del graph, plans
    gc.collect()
    assert sensor_ref() is None


# REQ: PY-API-025
def test_callback_cycle_through_the_graph_is_collectable():
    # Key setup: The sensor refers back to the Graph whose receiver holds its bound methods.
    sensor = PythonSensor()
    graph, _ = _graph("callback-graph-cycle", sensor)
    sensor.graph = graph
    sensor_ref = weakref.ref(sensor)
    graph_ref = weakref.ref(graph)

    del sensor, graph
    gc.collect()

    # Key assertion: Cyclic GC sees the Graph's ownership of the callbacks and reclaims the cycle.
    assert sensor_ref() is None
    assert graph_ref() is None


# REQ: PY-API-025
def test_callback_cycle_through_a_plan_is_collectable():
    # Key setup: The sensor refers to a compiled Plan, which owns the callbacks of its Graph.
    sensor = PythonSensor()
    graph, _ = _graph("callback-plan-cycle", sensor)
    sensor.plan = hs.compile(graph)
    sensor_ref = weakref.ref(sensor)

    del sensor, graph
    gc.collect()

    # Key assertion: The Plan's ownership is also visible, so the cycle is reclaimed.
    assert sensor_ref() is None


# REQ: PY-API-025
def test_a_retained_plan_keeps_callbacks_usable_after_its_graph_is_collected():
    # Key setup: Keep only the Plan; the Graph and the caller's sensor reference are dropped, and
    # the sensor's events are kept to observe the calls.
    #
    #   receiver --> sink
    sensor = PythonSensor()
    started, stopped = sensor.started, sensor.stopped
    graph, _ = _graph("callback-plan-owner", sensor)
    plan = hs.compile(graph)
    sensor_ref = weakref.ref(sensor)
    graph_ref = weakref.ref(graph)
    del sensor, graph
    gc.collect()
    assert graph_ref() is None
    assert sensor_ref() is not None

    session = hs.run_async(plan)
    try:
        assert started.wait(timeout=5.0)
    finally:
        session.request_stop()
        assert session.wait(timeout=5.0)

    # Key assertion: The Plan alone kept both callbacks valid for the run; dropping it releases
    # the sensor.
    assert stopped.is_set()
    assert session.termination in (hs.RunTermination.REQUESTED_COMPLETED, hs.RunTermination.STOPPED)
    del plan
    gc.collect()
    assert sensor_ref() is not None  # the joined session still holds its root until released
    assert session.wait()
    gc.collect()
    assert sensor_ref() is None


# REQ: PY-API-025
def test_a_running_session_owns_its_callbacks_independently_of_the_plan():
    # Key setup: Start a run, then drop the Graph, the Plan, and the caller's sensor reference
    # while it is still running.
    sensor = PythonSensor()
    started, stopped = sensor.started, sensor.stopped
    graph, _ = _graph("callback-session-owner", sensor)
    session = hs.run_async(hs.compile(graph))
    sensor_ref = weakref.ref(sensor)
    del sensor, graph
    assert started.wait(timeout=5.0)
    gc.collect()

    # Key assertion: The running session keeps the sensor alive, so stop still reaches it.
    assert sensor_ref() is not None
    session.request_stop()
    assert session.wait()
    assert stopped.is_set()
    gc.collect()
    assert sensor_ref() is None


# REQ: PY-API-025
def test_abandoning_a_running_session_still_stops_the_sensor():
    # Key setup: Drop every Python reference, including the session, while the run is active.
    sensor = PythonSensor()
    started, stopped = sensor.started, sensor.stopped
    graph, _ = _graph("callback-abandoned-session", sensor)
    session = hs.run_async(hs.compile(graph))
    del sensor, graph
    assert started.wait(timeout=5.0)

    del session
    gc.collect()

    # Key assertion: Abandonment stops and joins the run, and the stop callback was still valid.
    assert stopped.wait(timeout=5.0)


# REQ: PY-API-025
def test_a_plan_without_python_owners_fails_with_a_bounded_token():
    # Key setup: Remove the Plan's own ownership after the Graph is gone, so nothing owns the
    # callables when the run starts.
    sensor = PythonSensor()
    graph, _ = _graph("callback-released", sensor)
    plan = hs.compile(graph)
    del graph
    plan.__dict__.clear()
    gc.collect()

    session = hs.run_async(plan)
    try:
        assert session.wait(timeout=5.0)
    finally:
        session.request_stop()
        session.wait(timeout=5.0)

    # Key assertion: The released callback fails the run with an SDK token instead of calling a
    # freed object.
    assert session.termination == hs.RunTermination.FAILED
    assert "CONSTRUCTOR_CALLBACK_RELEASED" in session.summary()
    assert not sensor.started.is_set()


# REQ: PY-API-025
def test_non_callable_callback_is_rejected_before_mutation():
    # Key setup: Pass an object that is neither callable nor a handle provider.
    graph = hs.Graph("callback-not-callable")
    with pytest.raises(hs.errors.IncompatibleError) as caught:
        CallbackReceiver(graph, "receiver", data_channel=provider.TestChannel(12), device_start=42)

    # Key assertion: The field-qualified token names the input, and the name stays reusable.
    assert caught.value.token == "AUTHORING_CONSTRUCTOR_HANDLE_TYPE_MISMATCH"
    assert "field=device_start" in str(caught.value)
    assert CallbackReceiver(graph, "receiver", data_channel=provider.TestChannel(13)).graph_valid


# REQ: PY-API-025
def test_interpreter_exit_with_live_python_callbacks_is_clean():
    # Key setup: Exit the interpreter while a compiled plan still owns Python callbacks.
    script = textwrap.dedent(
        """
        import holoscan_constructor_handle_provider as provider
        import holoscan_test_support
        import holoscan as hs

        class Sensor:
            def start(self):
                pass

            def stop(self):
                pass

        sensor = Sensor()
        graph = hs.Graph("callback-exit")
        receiver = hs.registry.resolve("holoscan.test.py.CallbackReceiver", kind="operator")(
            graph, "receiver", data_channel=provider.TestChannel(14),
            device_start=sensor.start, device_stop=sensor.stop,
        )
        sink = hs.registry.resolve("holoscan.test.Sink", kind="operator")(graph, "sink")
        graph.add_flow(receiver.output, sink.in_)
        graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
        PLAN = hs.compile(graph)
        assert PLAN.ok
        print("authored")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        env=os.environ.copy(),
        check=False,
    )

    # Key assertion: Finalization releases or skips the callbacks without crashing or hanging.
    assert completed.returncode == 0, completed.stderr
    assert "authored" in completed.stdout


# REQ: PY-API-025
def test_generated_stubs_type_callback_keywords_without_a_mapping():
    # Key setup: Render the callback row's stub with no explicit annotation mapping.
    path = os.environ.get("HOLOSCAN_OPERATOR_STUB_GENERATOR")
    if path is None:
        pytest.skip("operator stub generator path is unavailable")
    spec = importlib.util.spec_from_file_location("_generate_operator_stubs", path)
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)

    rendered = generator.render_operator_proxies(
        {"CallbackReceiver": CallbackReceiver},
        {"holoscan.test.py.TestChannel": "holoscan_constructor_handle_provider.TestChannel"},
    )

    # Key assertion: The SDK's built-in callback type has a built-in annotation.
    assert "import collections.abc" in rendered
    assert "device_start: collections.abc.Callable[[], object] | None = None" in rendered


class _CallableHandleProvider:
    """A callable that also implements the handle protocol but cannot export a callback."""

    def __call__(self):
        pass

    def __holoscan_constructor_handle__(self, type_name):
        return NotImplemented


# REQ: PY-API-025
def test_handle_protocol_takes_precedence_over_callable_adaptation():
    # Key setup: Pass an object that is callable and also implements the provider protocol.
    graph = hs.Graph("callback-protocol-precedence")
    with pytest.raises(hs.errors.IncompatibleError) as caught:
        CallbackReceiver(
            graph,
            "receiver",
            data_channel=provider.TestChannel(15),
            device_start=_CallableHandleProvider(),
        )

    # Key assertion: The provider's own answer wins, so a provider can supply a C++ callback and
    # Holoscan never wraps such an object as a Python callable behind its back.
    assert caught.value.token == "AUTHORING_CONSTRUCTOR_HANDLE_TYPE_MISMATCH"
