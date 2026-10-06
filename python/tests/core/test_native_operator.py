# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Python Operator hosting, parameters, scalar edges, and callback failures."""

import json
import subprocess
import sys
import textwrap
import threading

import holoscan_test_support
import pytest

import holoscan as hs

COUNTER = "holoscan.tests.CounterSource"
SINK = "holoscan.tests.Sink"


class Increment(hs.Operator):
    """Adds a declared scalar parameter; used by the mixed-edge and replay tests."""

    inp = hs.Input("in", hs.payload.uint32)
    out = hs.Output("out", hs.payload.uint32)
    amount = hs.Param(1, description="Amount added to each sample.", range=(1, 8))
    instance_count = 0
    activations = []
    samples = []

    def setup(self, spec):
        spec.input(self.inp, queue_depth=4)
        spec.output(self.out)

    def compute(self, context):
        if not hasattr(self, "instance"):
            type(self).instance_count += 1
            self.instance = type(self).instance_count
        type(self).activations.append(self.instance)
        sample = self.inp.receive()
        type(self).samples.append(sample)
        self.out.emit(sample.data + self.amount)


# Compose the representative C++ → Python → C++ graph used by native hosting tests.
def increment_plan(name="native-increment", **params):
    graph = hs.Graph(name)
    source = graph.op(COUNTER, "source", limit=3)
    increment = graph.op(Increment, "increment", **params)
    sink = graph.op(SINK, "sink")
    graph.add_flow(source.out, increment["in"], queue_depth=4)
    graph.add_flow(increment.out, sink["in"], queue_depth=4)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    input_payload_type = increment["in"].payload_type
    return hs.compile(graph), input_payload_type


# Run a finite-observation graph and stop it after the C++ sink receives output.
def run_until_observed(plan, count=3):
    holoscan_test_support._reset_pipeline_observation()
    with hs.run_async(plan) as session:
        assert holoscan_test_support._wait_for_pipeline_observation(count, 5.0)
        session.request_stop()
        assert session.wait(timeout=5.0)
    return session


def diagnostic_error_messages(diagnostics):
    """Return owned error messages from structured diagnostics that carry details."""
    return [item.detail.error.message for item in diagnostics if item.detail is not None]


# REQ: PY-OP-001
def test_execution_context_operator_path_is_owned_encoded_and_callback_scoped():
    # Key setup: Run root and nested Python operators whose decoded names require percent encoding.
    #
    #   source[C++] --> root/%[Python] --> nested/%/child/%[Python] --> sink[C++]
    class PathRecorder(hs.Operator):
        inp = hs.Input("in", hs.payload.uint32)
        out = hs.Output("out", hs.payload.uint32)
        paths = []
        contexts = []

        def compute(self, ctx):
            type(self).paths.append(ctx.operator_path)
            type(self).contexts.append(ctx)
            self.out.emit(self.inp.receive_data())

    class NestedPathGraph(hs.Graph):
        input = hs.Input("in", hs.payload.uint32)
        output = hs.Output("out", hs.payload.uint32)

        def compose(self):
            self.clock_import = self.import_clock("timeline")
            self.set_default_clock(self.clock_import)
            self.member = self.op(PathRecorder, "child/%")
            self.export_input(self.input, self.member["in"])
            self.export_output(self.output, self.member["out"])

    PathRecorder.paths.clear()
    PathRecorder.contexts.clear()
    graph = hs.Graph("execution-context-path")
    source = graph.op(COUNTER, "source", limit=1)
    root = graph.op(PathRecorder, "root/%")
    nested = graph.embed(NestedPathGraph, "nested/%")
    sink = graph.op(SINK, "sink")
    graph.add_flow(source.out, root["in"], queue_depth=1)
    graph.add_flow(root["out"], nested.input, queue_depth=1)
    graph.add_flow(nested.output, sink["in"], queue_depth=1)
    clock = graph.add_clock(hs.RealtimeClock, "clock")
    graph.set_default_clock(clock)
    graph.bind_clock(nested, nested.clock_import, clock)

    session = run_until_observed(hs.compile(graph), 1)

    # Key assertion: Each callback gets an owned canonical path and retained contexts fail closed.
    assert session.diagnostics == []
    assert PathRecorder.paths == ["root%2F%25", "nested%2F%25/child%2F%25"]
    for context in PathRecorder.contexts:
        with pytest.raises(hs.errors.NotReadyError, match="PYTHON_EXECUTION_CONTEXT_EXPIRED"):
            _ = context.operator_path


# REQ: PY-OP-001
def test_execution_context_exposes_its_live_runtime_clock():
    # Key setup: Read the activation clock and retain both the callback context and returned
    # reference after a finite C++ -> Python -> C++ pipeline stops.
    class ClockRecorder(hs.Operator):
        inp = hs.Input("in", hs.payload.uint32)
        out = hs.Output("out", hs.payload.uint32)
        contexts = []
        clocks = []
        points = []

        def compute(self, ctx):
            clock = ctx.clock()
            type(self).contexts.append(ctx)
            type(self).clocks.append(clock)
            type(self).points.append(clock.now())
            self.out.emit(self.inp.receive_data())

    ClockRecorder.contexts.clear()
    ClockRecorder.clocks.clear()
    ClockRecorder.points.clear()
    graph = hs.Graph("execution-context-clock")
    source = graph.op(COUNTER, "source", limit=1)
    recorder = graph.op(ClockRecorder, "recorder")
    sink = graph.op(SINK, "sink")
    graph.add_flow(source.out, recorder["in"], queue_depth=1)
    graph.add_flow(recorder.out, sink["in"], queue_depth=1)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))

    session = run_until_observed(hs.compile(graph), 1)

    # Key assertion: The compute context returns the shared facade while live, then expires its
    # own minting authority without erasing retained reference diagnostics.
    assert session.diagnostics == []
    assert len(ClockRecorder.clocks) == 1
    clock = ClockRecorder.clocks[0]
    assert isinstance(clock, hs.clocks.ClockRef)
    assert ClockRecorder.points[0].clock_id == clock.id
    assert clock.at_ns(91) == hs.time.TimePoint(91, clock.id)
    with pytest.raises(hs.errors.NotReadyError, match="PYTHON_EXECUTION_CONTEXT_EXPIRED"):
        ClockRecorder.contexts[0].clock()
    with pytest.raises(hs.errors.StaleEndpointError):
        clock.now()


# REQ: PY-OP-001
def test_operator_ports_reject_other_threads_and_retained_callback_access():
    # Key setup: Attempt input and output operations from a helper while the callback is live,
    # then retain both ports beyond callback completion.
    class ThreadAuthority(hs.Operator):
        inp = hs.Input("in", hs.payload.uint32)
        out = hs.Output("out", hs.payload.uint32)
        failures = []
        retained = []

        def compute(self, context):
            def use_ports():
                for operation in (self.inp.receive_data, lambda: self.out.emit(7)):
                    try:
                        operation()
                    except Exception as error:
                        type(self).failures.append(error)

            worker = threading.Thread(target=use_ports)
            worker.start()
            worker.join()
            sample = self.inp.receive_data()
            self.out.emit(sample)
            type(self).retained = [self.inp, self.out]

    ThreadAuthority.failures.clear()
    ThreadAuthority.retained.clear()
    graph = hs.Graph("operator-port-thread-authority")
    source = graph.op(COUNTER, "source", limit=1)
    authority = graph.op(ThreadAuthority, "authority")
    sink = graph.op(SINK, "sink")
    graph.add_flow(source.out, authority["in"], queue_depth=1)
    graph.add_flow(authority["out"], sink["in"], queue_depth=1)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))

    session = run_until_observed(hs.compile(graph), 1)

    # Key assertion: Only the owning callback thread may touch Core I/O, and escaped ports retain
    # no authority after the callback even though their run-specific references remain bound.
    assert session.diagnostics == []
    assert len(ThreadAuthority.failures) == 2
    assert all(isinstance(error, hs.errors.NotReadyError) for error in ThreadAuthority.failures)
    assert all(
        "PYTHON_OPERATOR_PORT_UNAVAILABLE" in str(error) for error in ThreadAuthority.failures
    )
    with pytest.raises(hs.errors.NotReadyError, match="PYTHON_OPERATOR_PORT_UNAVAILABLE"):
        ThreadAuthority.retained[0].receive_data()
    with pytest.raises(hs.errors.NotReadyError, match="PYTHON_OPERATOR_PORT_UNAVAILABLE"):
        ThreadAuthority.retained[1].emit(7)


# REQ: PY-OP-001
def test_native_operator_runs_on_mixed_scalar_edges_with_fresh_instances():
    # Key setup: Compile one C++ source → Python operator → C++ sink graph and replay it twice.
    #
    #   source[C++] --uint32--> increment[Python] --uint32--> sink[C++]
    Increment.activations.clear()
    Increment.samples.clear()
    plan, input_payload_type = increment_plan(amount=2)
    assert plan.ok

    first = run_until_observed(plan)
    first_instances = set(Increment.activations)
    first_sample = Increment.samples[0]
    Increment.activations.clear()
    Increment.samples.clear()
    second = run_until_observed(plan)
    second_instances = set(Increment.activations)

    # Key assertion: Scalar conversion is exact and each plan run owns a fresh Python instance.
    assert first.diagnostics == []
    assert second.diagnostics == []
    assert holoscan_test_support._pipeline_observation_last_value() == 5
    assert len(first_instances) == 1
    assert len(second_instances) == 1
    assert first_instances.isdisjoint(second_instances)
    assert input_payload_type == hs.payload.uint32
    assert first_sample.data == 1
    assert first_sample.metadata.sequence > 0
    assert first_sample.metadata.payload_kind == hs.PayloadKind.INLINE_TYPED
    combined_flags = hs.SampleFlags.INVALID | hs.SampleFlags.DEGRADED
    assert isinstance(combined_flags, hs.SampleFlags)
    assert combined_flags & hs.SampleFlags.INVALID
    assert combined_flags & hs.SampleFlags.DEGRADED
    assert not first_sample.is_missing
    assert not first_sample.is_stale


# REQ: PY-OP-001
@pytest.mark.parametrize(
    ("reported_gil", "expected_gil"), [(True, True), (False, False), (None, True)]
)
def test_native_operator_serialization_domain_follows_interpreter_gil(
    monkeypatch, reported_gil, expected_gil
):
    """Verify the compiler-visible placement policy selected for Python callbacks."""

    class Idle(hs.Operator):
        """Provide a callback-only Operator for the serialization-domain plan check."""

        def compute(self, context):
            pass

    #   first[Python]   second[Python]
    #
    # Key setup: Simulate GIL-enabled and free-threaded interpreters before hosting two otherwise
    # disconnected Python callbacks.
    if reported_gil is None:
        monkeypatch.delattr(sys, "_is_gil_enabled", raising=False)
    else:
        monkeypatch.setattr(sys, "_is_gil_enabled", lambda: reported_gil, raising=False)
    graph = hs.Graph("native-serialization-domain")
    graph.op(Idle, "first")
    graph.op(Idle, "second")
    plan = hs.compile(graph)
    document = json.loads(plan.json())

    domains = {host["serialization_domain"] for host in document["operator_hosts"]}
    partition_members = [set(partition["operator_paths"]) for partition in document["partitions"]]

    # Key assertion: GIL-enabled hosts share `python.gil` and one compatible partition;
    # free-threaded hosts omit the domain and retain ordinary independent partitioning.
    assert plan.ok, plan.json()
    assert domains == ({"python.gil"} if expected_gil else {""})
    assert any(members == {"first", "second"} for members in partition_members) is expected_gil


# REQ: PY-OP-001
def test_operator_spec_input_accepts_explicit_optional_permission():
    # Key setup: Declare an unwired Python-native input with explicit optional permission.
    #
    #   [unwired] --> optional.input
    class OptionalDeclaration(hs.Operator):
        side = hs.Input("side", hs.payload.uint32)

        def setup(self, spec):
            spec.input(self.side, queue_depth=2, optional=True)

        def compute(self, context):
            pass

    graph = hs.Graph("native-optional-input-declaration")
    optional = graph.op(OptionalDeclaration, "optional")

    # Key assertion: the Python spelling reaches the foreign-host setup bridge and binds the port.
    assert optional["side"].name == "side"
    assert optional["side"].direction == hs.PortDirection.INPUT


# REQ: PY-OP-002
def test_hosted_parameters_validate_and_project_into_the_plan_catalog():
    # Key setup: Compile one hosted operator with scalar and recompile-required sequence fields.
    #
    #   increment[Python]
    class ParameterProbe(hs.Operator):
        out = hs.Output("out", hs.payload.uint32)
        threshold = hs.Param(4, description="A bounded threshold.", range=(1, 8))
        shape = hs.Param([2, 3], max_elements=3, recompile_required=True)

        def compute(self, context):
            pass

    graph = hs.Graph("hosted-parameter-catalog")
    probe = graph.op(ParameterProbe, "probe", threshold=6, shape=[3, 4, 5])
    plan = hs.compile(graph)
    document = json.loads(plan.json())
    fields = {field["canonical_name"]: field for field in document["parameter_fields"]}

    # Key assertion: Descriptor access and the language-neutral plan schema preserve constraints.
    assert ParameterProbe.threshold.default == 4
    assert ParameterProbe.threshold.range == (1, 8)
    assert ParameterProbe.threshold.description == "A bounded threshold."
    assert "threshold=6" in repr(probe)
    assert fields["threshold"]["logical_type"] == 1
    assert fields["shape"]["mutability"] == 3
    assert fields["shape"]["type_shape"]["element_or_rep_bit_width"] == 64
    assert [(port["local_name"], port["direction"]) for port in document["ports"]] == [("out", 1)]

    with pytest.raises(hs.errors.InvalidArgumentError, match="PARAM_VALIDATION_FAILED"):
        graph.op(ParameterProbe, "range-failure", threshold=9)
    with pytest.raises(hs.errors.InvalidArgumentError, match="PARAM_VALIDATION_FAILED"):
        graph.op(ParameterProbe, "shape-failure", shape=[1, 2, 3, 4])
    with pytest.raises(hs.errors.InvalidArgumentError, match="AUTHORING_PARAMETER_UNKNOWN"):
        graph.op(ParameterProbe, "unknown", typo=1)


# REQ: PY-OP-002
def test_hosted_parameter_values_are_available_during_runtime_initialization():
    # Key setup: Read the per-run copied value from __init__, before callback bindings are attached.
    initialized_values = []

    class InitializerProbe(Increment):
        threshold = hs.Param(4, range=(1, 8))

        def __init__(self):
            initialized_values.append(self.threshold)

    graph = hs.Graph("hosted-parameter-initializer")
    source = graph.op(COUNTER, "source", limit=1)
    probe = graph.op(InitializerProbe, "probe", threshold=7)
    sink = graph.op(SINK, "sink")
    graph.add_flow(source.out, probe["in"], queue_depth=1)
    graph.add_flow(probe.out, sink["in"], queue_depth=1)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    session = run_until_observed(hs.compile(graph), 1)

    # Key assertion: authoring and runtime construction both see their exact parameter copy.
    assert session.diagnostics == []
    assert len(initialized_values) >= 2
    assert set(initialized_values) == {7}


# REQ: PY-OP-003
# REQ: PY-OP-007
@pytest.mark.parametrize("callback", ["compute", "start", "stop"])
@pytest.mark.parametrize(
    "exception_type, expected_code",
    [
        (RuntimeError, hs.errors.ErrorCode.FAILURE),
        (hs.errors.InvalidArgumentError, hs.errors.ErrorCode.INVALID_ARGUMENT),
    ],
    ids=["generic", "typed"],
)
def test_python_callback_exceptions_fail_closed_with_traceback(
    callback, exception_type, expected_code
):
    # Key setup: Raise from one hosted callback in a C++→Python→C++ graph.
    #
    #   source[C++] --uint32--> failing[Python] --uint32--> sink[C++]
    def fail():
        error = exception_type("native callback sentinel")
        if isinstance(error, hs.errors.HoloscanError):
            error.code = expected_code
        raise error

    class Failing(Increment):
        def compute(self, ctx):
            if callback == "compute":
                fail()
            super().compute(ctx)

        def start(self):
            if callback == "start":
                fail()

        def stop(self):
            if callback == "stop":
                fail()

    graph = hs.Graph(f"native-{callback}-failure")
    source = graph.op(COUNTER, "source", limit=1_000_000 if callback == "stop" else 3)
    failing = graph.op(Failing, "failing")
    sink = graph.op(SINK, "sink")
    graph.add_flow(source.out, failing["in"], queue_depth=4)
    graph.add_flow(failing.out, sink["in"], queue_depth=4)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    holoscan_test_support._reset_pipeline_observation()
    session = hs.run_async(hs.compile(graph))
    if callback == "stop":
        assert holoscan_test_support._wait_for_pipeline_observation(1, 5.0)
        session.request_stop()
    assert session.wait(timeout=5.0)

    # Key assertion: The failed run retains both the callback traceback and its error code.
    assert session.termination is hs.RunTermination.FAILED
    errors = [
        item.detail.error
        for item in session.diagnostics
        if item.detail is not None and "native callback sentinel" in item.detail.error.message
    ]
    assert errors
    for error in errors:
        assert error.code is expected_code
        assert "test_native_operator.py" in error.message


# REQ: PY-OP-007
def test_python_callback_exception_with_code_zero_still_fails_closed():
    # Key setup: Raise a non-Holoscan exception whose code resembles ErrorCode.SUCCESS.
    #
    #   source[C++] --uint32--> failing[Python]
    class MisleadingError(RuntimeError):
        code = 0

    class Failing(Increment):
        def compute(self, context):
            raise MisleadingError("misleading success code")

    graph = hs.Graph("native-code-zero-failure")
    source = graph.op(COUNTER, "source", limit=1)
    failing = graph.op(Failing, "failing")
    graph.add_flow(source.out, failing["in"], queue_depth=1)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    session = hs.run(hs.compile(graph))

    # Key assertion: An arbitrary exception code cannot turn a raised exception into success.
    assert session.termination is hs.RunTermination.FAILED
    errors = [
        item.detail.error
        for item in session.diagnostics
        if item.detail is not None and "misleading success code" in item.detail.error.message
    ]
    assert errors
    assert all(error.code is hs.errors.ErrorCode.FAILURE for error in errors)


# REQ: PY-OP-003
# REQ: PY-OP-007
def test_setup_exception_is_translated_at_the_authoring_boundary():
    # Key setup: Raise while the hosted prototype declares its ports.
    class SetupFailure(hs.Operator):
        out = hs.Output("out", hs.payload.uint32)

        def setup(self, spec):
            del spec
            raise RuntimeError("native setup sentinel")

        def compute(self, context):
            pass

    graph = hs.Graph("native-setup-failure")

    # Key assertion: The Python exception remains contained behind Core's stable setup diagnostic.
    with pytest.raises(hs.errors.FailureError, match="GRAPH_OPERATOR_SETUP_RUNTIME_ERROR"):
        graph.op(SetupFailure, "failing")


# REQ: PY-OP-004
def test_conditional_setup_is_rejected_during_runtime_replay():
    # Key setup: Make the runtime instance declare a different port set than its prototype.
    class ConditionalSetup(hs.Operator):
        out = hs.Output("out", hs.payload.uint32)
        setup_count = 0

        def setup(self, spec):
            type(self).setup_count += 1
            if type(self).setup_count == 1:
                spec.output(self.out)

        def compute(self, context):
            pass

    graph = hs.Graph("conditional-native-setup", default_clock=hs.RealtimeClock)
    graph.op(ConditionalSetup, "conditional")
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    session = hs.run(hs.compile(graph))

    # Key assertion: Runtime replay catches nondeterministic setup before normal activation.
    assert session.diagnostics
    assert any("RUNTIME_OPERATOR_REPLAY_MISMATCH" in item.token for item in session.diagnostics)


# REQ: PY-OP-002
def test_float_parameter_shape_follows_the_default_and_normalizes_integer_overrides():
    # Key setup: Override a floating-point parameter with an exactly representable Python integer.
    class FloatParameter(hs.Operator):
        out = hs.Output("out", hs.payload.uint32)
        ratio = hs.Param(0.5, range=(0.0, 2.0))
        observed = []

        def setup(self, spec):
            type(self).observed.append((self.ratio, type(self.ratio)))
            spec.output(self.out)

        def compute(self, context):
            pass

    FloatParameter.observed.clear()
    graph = hs.Graph("float-parameter-default-shape")
    graph.op(FloatParameter, "float-parameter", ratio=1)
    document = json.loads(hs.compile(graph).json())
    field = next(item for item in document["parameter_fields"] if item["canonical_name"] == "ratio")

    # Key assertion: The declared float shape is stable and the callback sees a normalized float.
    assert field["logical_type"] == 3
    assert FloatParameter.observed == [(1.0, float), (1.0, float)]
    with pytest.raises(hs.errors.InvalidArgumentError, match="category=out_of_range"):
        graph.op(FloatParameter, "overflow", ratio=10**1_000)


# REQ: PY-OP-001
def test_operator_spec_rejects_use_after_setup_returns():
    # Key setup: Retain the callback-lifetime spec and a bound port beyond setup().
    class RetainedSpec(hs.Operator):
        first = hs.Output("first", hs.payload.uint32)
        second = hs.Output("second", hs.payload.uint32)
        saved_spec = None
        saved_port = None

        def setup(self, spec):
            type(self).saved_spec = spec
            type(self).saved_port = self.second
            spec.output(self.first)

        def compute(self, context):
            pass

    graph = hs.Graph("retained-operator-spec")
    graph.op(RetainedSpec, "retained")

    # Key assertion: Escaped callback state fails closed instead of dereferencing expired C++ state.
    with pytest.raises(hs.errors.NotReadyError, match="AUTHORING_OPERATOR_SPEC_EXPIRED"):
        RetainedSpec.saved_spec.output(RetainedSpec.saved_port)


# REQ: PY-OP-001
def test_operator_port_access_during_init_has_an_actionable_error():
    # Key setup: Touch a class-body output before the runtime installs callback bindings.
    class EarlyPortAccess(hs.Operator):
        out = hs.Output("out", hs.payload.uint32)

        def __init__(self):
            self.saved = self.out

        def compute(self, context):
            pass

    graph = hs.Graph("early-port-access")

    # Key assertion: The authoring error identifies the invalid lifecycle phase and port.
    with pytest.raises(hs.errors.FailureError, match="Output 'out'.*unavailable.*__init__"):
        graph.op(EarlyPortAccess, "early")


# REQ: PY-OP-002
def test_param_descriptor_cannot_be_reused_under_a_different_name():
    # Key setup: Simulate Python assigning one descriptor to two differently named attributes.
    descriptor = hs.Param(1)
    descriptor.__set_name__(object, "first")

    # Key assertion: The second name is rejected before instances can observe divergent lookup keys.
    with pytest.raises(TypeError, match="already belongs.*first.*reused.*second"):
        descriptor.__set_name__(object, "second")


# REQ: PY-OP-002
def test_parameter_deepcopy_failure_preserves_the_python_exception():
    # Key setup: Declare a callback-only parameter whose value explicitly refuses deep-copying.
    class RefusesDeepcopy:
        def __deepcopy__(self, memo):
            del memo
            raise RuntimeError("deepcopy sentinel")

    class NonCopyableParameter(hs.Operator):
        value = hs.Param(RefusesDeepcopy())

        def compute(self, context):
            pass

    graph = hs.Graph("noncopyable-parameter")

    # Key assertion: Graph capture reports the original Python failure without a generic cast error.
    with pytest.raises(RuntimeError, match="deepcopy sentinel"):
        graph.op(NonCopyableParameter, "noncopyable")


# REQ: PY-OP-002
def test_mutable_parameter_defaults_are_isolated_across_plan_replays():
    # Key setup: Mutate a list parameter in a C++ → Python → C++ graph replayed twice.
    #
    #   source[C++] --uint32--> mutable[Python] --uint32--> sink[C++]
    class MutableParameter(hs.Operator):
        inp = hs.Input("in", hs.payload.uint32)
        out = hs.Output("out", hs.payload.uint32)
        values = hs.Param([1, 2], max_elements=3)
        first_observations = []

        def compute(self, context):
            if not hasattr(self, "observed_default"):
                assignment_rejected = False
                try:
                    self.values = [9]
                except AttributeError:
                    assignment_rejected = True
                type(self).first_observations.append(
                    (list(self.values), assignment_rejected, hasattr(self, "params"))
                )
                self.observed_default = True
            sample = self.inp.receive_data()
            self.values.append(9)
            self.out.emit(sample)

    MutableParameter.first_observations.clear()
    graph = hs.Graph("mutable-parameter-replay")
    source = graph.op(COUNTER, "source", limit=3)
    mutable = graph.op(MutableParameter, "mutable")
    sink = graph.op(SINK, "sink")
    graph.add_flow(source.out, mutable["in"], queue_depth=4)
    graph.add_flow(mutable.out, sink["in"], queue_depth=4)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
    plan = hs.compile(graph)
    run_until_observed(plan)
    run_until_observed(plan)

    # Key assertion: Every run gets a deep copy, direct attributes are read-only, and no
    # unnamespaced alias exists.
    assert MutableParameter.first_observations == [([1, 2], True, False), ([1, 2], True, False)]


# REQ: PY-OP-007
def test_truncated_callback_tracebacks_remain_valid_utf8():
    # Key setup: Shift a long multibyte exception across every possible three-byte cut boundary.
    for offset in range(3):

        class UnicodeFailure(hs.Operator):
            inp = hs.Input("in", hs.payload.uint32)

            def compute(self, context, failure_offset=offset):
                self.inp.receive_data()
                raise RuntimeError("x" * failure_offset + "€" * 5_000)

        graph = hs.Graph(f"unicode-callback-{offset}")
        source = graph.op(COUNTER, "source", limit=1)
        failing = graph.op(UnicodeFailure, "failing")
        graph.add_flow(source.out, failing["in"], queue_depth=1)
        graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
        session = hs.run(hs.compile(graph))

        # Key assertion: Every bounded diagnostic is still representable as valid UTF-8 text.
        assert session.diagnostics
        messages = diagnostic_error_messages(session.diagnostics)
        assert messages
        for message in messages:
            message.encode("utf-8")


# REQ: PY-API-005
@pytest.mark.parametrize("session_count", [1, 3])
def test_hosted_session_reaper_reclaims_last_completed_batch(session_count):
    # Key setup: Drop every RunSession reference from its own hosted callback, then start no more
    # sessions while the process continues executing Python.
    script = textwrap.dedent(
        """
        import gc
        import threading

        import holoscan as hs
        import holoscan_test_support

        holder = {}
        lock = threading.Lock()
        dropped = threading.Event()
        finalized = threading.Event()
        dropped_count = 0
        finalized_count = 0
        session_count = SESSION_COUNT

        class DropSession(hs.Operator):
            inp = hs.Input("in", hs.payload.uint32)
            key = hs.Param("")

            def compute(self, context):
                global dropped_count
                self.inp.receive_data()
                self.was_active = True
                if self.key in holder:
                    del holder[self.key]
                    gc.collect()
                    with lock:
                        dropped_count += 1
                        if dropped_count == session_count:
                            dropped.set()

            def __del__(self):
                global finalized_count
                if getattr(self, "was_active", False):
                    with lock:
                        finalized_count += 1
                        if finalized_count == session_count:
                            finalized.set()

        retained_plans = []
        for index in range(session_count):
            key = f"session-{index}"
            graph = hs.Graph(f"hosted-session-reaper-{index}")
            source = graph.op(
                "holoscan.tests.CounterSource", "source", limit=1_000_000
            )
            dropper = graph.op(DropSession, "dropper", key=key)
            graph.add_flow(source.out, dropper["in"], queue_depth=4)
            graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))
            plan = hs.compile(graph)
            retained_plans.append(plan)
            holder[key] = hs.run_async(plan)
        assert dropped.wait(5.0)
        print("HOSTED_REAPER_DROPPED", flush=True)

        # Block the interpreter main thread in one wait. A pending call cannot service the
        # retirement while this wait releases the GIL, so the notifier fallback must reclaim the
        # final batch from a GIL-owning cleanup thread.
        assert finalized.wait(5.0)
        print("HOSTED_REAPER_RECLAIMED", flush=True)
        """
    ).replace("SESSION_COUNT", str(session_count))
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=15, check=False
    )

    # Key assertion: The interpreter handoff reclaims the final joined batch without another run;
    # successful process exit also retains the bounded shutdown behavior.
    assert result.returncode == 0, result.stderr
    assert "HOSTED_REAPER_DROPPED" in result.stdout
    assert "HOSTED_REAPER_RECLAIMED" in result.stdout


# REQ: PY-OP-003
