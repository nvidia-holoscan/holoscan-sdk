# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable Python Graph authoring tests."""

import gc
import json
import os
import subprocess
import traceback
import weakref

import holoscan_test_support  # noqa: F401
import pytest

import holoscan as hs
from holoscan.graphs import ScalePairGraph, WrappedScaleGraph

COUNTER = "holoscan.test.CounterSource"
DOUBLER = "holoscan.test.DoubleIt"
SINK = "holoscan.test.Sink"


class DoublerGraph(hs.Graph):
    """Reusable typed doubling stage used by the descriptor and re-export test."""

    input = hs.Input("in", hs.payload.uint32)
    output = hs.Output("out", hs.payload.uint32)

    def __init__(self, *, copies=1):
        super().__init__()
        self.copies = copies

    def compose(self):
        self.clock_import = self.import_clock("timeline")
        self.set_default_clock(self.clock_import)
        previous = self.input
        self.members = []
        for index in range(self.copies):
            member = self.op(DOUBLER, f"doubler-{index}")
            self.members.append(member)
            if index == 0:
                self.export_input(self.input, member["in"])
            else:
                self.add_flow(previous, member["in"])
            previous = member.out
        self.export_output(self.output, previous)


class WrappedDoublerGraph(hs.Graph):
    """A nested stage used by test_reusable_graph_descriptors_embedding_and_reexport."""

    input = hs.Input("in", hs.payload.uint32)
    output = hs.Output("out", hs.payload.uint32)

    def __init__(self, *, copies=1):
        super().__init__()
        self.copies = copies

    def compose(self):
        self.clock_import = self.import_clock("timeline")
        self.set_default_clock(self.clock_import)
        self.inner = self.embed(DoublerGraph, "inner", copies=self.copies)
        self.bind_clock(self.inner, self.inner.clock_import, self.clock_import)
        self.export_input(self.input, self.inner.input)
        self.export_output(self.output, self.inner.output)


# REQ: PY-API-008
#   reusable
#   |-- direct[DoublerGraph x2] --> wrapped[WrappedDoublerGraph x3]
#   `-- region[stages] = {direct, wrapped}
def test_reusable_graph_descriptors_embedding_and_reexport():
    # Key setup: Embed direct and wrapped reusable graphs, then connect and group them.
    graph = hs.Graph("reusable")
    direct = graph.embed(DoublerGraph, "direct", copies=2)
    wrapped = graph.embed(WrappedDoublerGraph, "wrapped", copies=3)

    graph.add_flow(direct.output, wrapped.input)
    graph.region("stages").add(direct, wrapped)

    # Key assertion: Class descriptors and instance handles retain typed nested graph metadata.
    assert isinstance(DoublerGraph.input, hs.Input)
    assert DoublerGraph.input.name == "in"
    assert DoublerGraph.output.payload_type == hs.payload.uint32
    assert isinstance(direct, hs.GraphHandle)
    assert direct.path == ("direct",)
    assert wrapped.path == ("wrapped",)
    assert wrapped.inner.path == ("inner",)
    assert len(direct.members) == 2
    assert len(wrapped.inner.members) == 3
    assert direct.input.direction is hs.PortDirection.INPUT
    assert wrapped.output.direction is hs.PortDirection.OUTPUT
    assert direct.input.graph_valid
    assert repr(direct) == "<GraphHandle 'direct' (DoublerGraph) valid=True>"

    # Key assertion: The read-only handle exposes composed data but not Graph mutators.
    assert direct.members[0].name == "doubler-0"
    assert not hasattr(direct, "op")
    with pytest.raises(AttributeError, match="does not expose callable attribute 'op'"):
        direct.op(DOUBLER, "late")


# REQ: PY-API-008
#   temporary
#   `-- child[DoublerGraph]
def test_embedded_graph_handles_do_not_extend_parent_lifetime():
    # Key setup: Retain child and interface-port handles after deleting their owning graph.
    graph = hs.Graph("temporary")
    child = graph.embed(DoublerGraph, "child")
    interface_port = child.output

    del graph
    gc.collect()

    # Key assertion: Both weak handles fail closed and protocol probes remain ordinary misses.
    assert not child.graph_valid
    assert child.path is None
    assert not interface_port.graph_valid
    assert not hasattr(child, "_repr_html_")
    with pytest.raises(AttributeError, match="has no attribute 'output'"):
        _ = child.output


# REQ: PY-API-008
#   top-level[TopLevelGraph]
#   `-- source
def test_top_level_graph_subclass_composes_once_before_compile():
    class TopLevelGraph(hs.Graph):
        """Top-level compose counter for this test."""

        def __init__(self):
            super().__init__("top-level", default_clock=hs.RealtimeClock)
            self.compose_calls = 0

        def compose(self):
            self.compose_calls += 1
            self.source = self.op(COUNTER, "source", limit=1)

    # Key setup: Compile the same top-level Python Graph subclass twice.
    graph = TopLevelGraph()
    first = hs.compile(graph)
    second = hs.compile(graph)

    # Key assertion: The subclass body is present and compose() ran exactly once.
    assert first.ok
    assert second.ok
    assert graph.compose_calls == 1
    assert '"operator_path":"source"' in first.json()


# REQ: PY-API-008
def test_top_level_graph_interface_descriptors_resolve_during_compose():
    # Key setup: Compile a top-level reusable Graph whose compose hook accesses both descriptors.
    graph = DoublerGraph(copies=1)

    plan = hs.compile(graph)

    # Key assertion: Descriptor access authors both interface bindings; only the deliberately
    # unbound reusable clock remains as a compile diagnostic.
    assert [error.token for error in plan.errors] == ["CLOCK_IMPORT_UNBOUND"]
    assert graph.input.graph_valid
    assert graph.output.graph_valid


# REQ: PY-API-008
def test_top_level_compose_failure_is_sticky_and_discards_partial_draft():
    class FailingTopLevelGraph(hs.Graph):
        """Top-level partial-authoring failure fixture for this test."""

        def __init__(self):
            super().__init__("failing-top-level")
            self.compose_calls = 0

        def compose(self):
            self.compose_calls += 1
            self.partial = self.op(COUNTER, "partial", limit=1)
            raise ValueError("top-level compose failed")

    # Key setup: Compile the same partially authored Graph twice after compose() fails.
    graph = FailingTopLevelGraph()
    failures = []
    for _ in range(2):
        with pytest.raises(ValueError, match="top-level compose failed") as caught:
            hs.compile(graph)
        failures.append(caught.value)

    # Key assertion: Failure is terminal, replay preserves it, and the draft handle is retired.
    assert [error.args for error in failures] == [
        ("top-level compose failed",),
        ("top-level compose failed",),
    ]
    assert any(frame.name == "compose" for frame in traceback.extract_tb(failures[0].__traceback__))
    assert graph.compose_calls == 1
    assert not graph.partial.graph_valid
    assert repr(graph) == "<Graph 'failing-top-level' operators=0 flows=0>"
    with pytest.raises(ValueError, match="top-level compose failed"):
        graph.op(COUNTER, "after-failure", limit=1)


# REQ: PY-API-008
def test_top_level_compose_failure_preserves_non_reconstructible_exception_state():
    class InvalidJSONGraph(hs.Graph):
        """JSON failure whose constructor arguments differ from its public args tuple."""

        def compose(self):
            json.loads("{")

    # Key setup: Compile and then author against a terminal JSON composition failure.
    graph = InvalidJSONGraph("invalid-json")
    failures = []
    for action in (lambda: hs.compile(graph), lambda: graph.op(COUNTER, "late", limit=1)):
        with pytest.raises(json.JSONDecodeError) as caught:
            action()
        failures.append(caught.value)

    # Key assertion: Replay raises the exact exception object with its specialized state intact.
    assert failures[1] is failures[0]
    assert (failures[1].doc, failures[1].pos) == ("{", 1)
    assert any(frame.name == "compose" for frame in traceback.extract_tb(failures[1].__traceback__))


# REQ: PY-API-008
def test_top_level_compose_failure_preserves_os_error_state_and_interrupts():
    missing = FileNotFoundError(2, "No such file or directory", "missing-input.json")
    interrupt = KeyboardInterrupt()

    class RaisingGraph(hs.Graph):
        """Raise the supplied stateful exception or interrupt from compose()."""

        def __init__(self, name, failure):
            super().__init__(name)
            self.failure = failure

        def compose(self):
            raise self.failure

    # Key assertion: Stateful errors and interrupts both remain exact terminal failures.
    for name, failure in (("missing-file", missing), ("interrupt", interrupt)):
        graph = RaisingGraph(name, failure)
        caught_failures = []
        for _ in range(2):
            with pytest.raises(type(failure)) as caught:
                hs.compile(graph)
            caught_failures.append(caught.value)
        assert caught_failures == [failure, failure]
        assert caught_failures[1] is caught_failures[0]
    assert missing.filename == "missing-input.json"
    assert missing.errno == 2


# REQ: PY-API-008
def test_top_level_compose_failure_cycle_is_visible_to_python_gc():
    class FailingGraph(hs.Graph):
        """Failure whose traceback frame retains its composing Graph."""

        def compose(self):
            self.self_reference = self
            raise RuntimeError("collectable composition failure")

    def fail_and_release():
        graph = FailingGraph("collectable-failure")
        graph_ref = weakref.ref(graph)
        with pytest.raises(RuntimeError, match="collectable composition failure"):
            hs.compile(graph)
        return graph_ref

    # Key setup: Leave only the graph/error/traceback/frame cycle after terminal failure.
    graph_ref = fail_and_release()
    gc.collect()

    # Key assertion: Native failure retention introduces no opaque edge that hides the cycle.
    assert graph_ref() is None


# REQ: PY-API-008
def test_recursive_top_level_compile_is_rejected_and_failure_is_sticky():
    class RecursiveCompileGraph(hs.Graph):
        """Recursive compile fixture used only by this rejection test."""

        def __init__(self):
            super().__init__("recursive-compile")
            self.compose_calls = 0

        def compose(self):
            self.compose_calls += 1
            hs.compile(self)

    # Key setup: Let compose() attempt to compile the same top-level Graph recursively.
    graph = RecursiveCompileGraph()
    failures = []
    for _ in range(2):
        with pytest.raises(hs.errors.InvalidArgumentError) as caught:
            hs.compile(graph)
        failures.append(caught.value)

    # Key assertion: Real recursive compilation is rejected once and replay keeps Core identity.
    assert graph.compose_calls == 1
    assert [error.token for error in failures] == [
        "GRAPH_COMPOSE_REENTRANT",
        "GRAPH_COMPOSE_REENTRANT",
    ]
    assert all(error.code is hs.errors.ErrorCode.INVALID_ARGUMENT for error in failures)


# REQ: PY-API-008
def test_top_level_and_embedded_compose_can_attach_config_with_parity():
    class ConfiguredGraph(hs.Graph):
        """Configuration-in-compose fixture used only by this parity test."""

        def __init__(self):
            super().__init__()
            self.document = hs.config.parse('{"detector":{"threshold":9}}', hs.config.Format.JSON)
            self.compose_calls = 0

        def compose(self):
            self.compose_calls += 1
            source = self.op(COUNTER, "source", limit=1)
            detector = self.op(
                "holoscan.test.ConfigDetector",
                "detector",
                threshold=6,
                gain=1.0,
                window=8,
            )
            self.add_flow(source.out, detector["in"])
            self.set_default_clock(self.add_clock(hs.RealtimeClock, "clock"))
            self.provenance = hs.params.set_params(
                self,
                self.document,
                [hs.params.bind(detector, "/detector")],
            )

    # Key setup: Run the same configuration transaction from top-level and hosted compose hooks.
    top_level = ConfiguredGraph()
    plan = hs.compile(top_level)
    owner = hs.Graph("configured-owner")
    embedded = owner.embed(ConfiguredGraph, "configured")

    # Key assertion: Side-effect-free Graph unwrapping gives both forms identical provenance.
    assert plan.ok
    assert top_level.compose_calls == 1
    assert embedded.compose_calls == 1
    assert top_level.provenance.config_field_count == 1
    assert embedded.provenance.config_field_count == 1
    assert [field.source_path for field in top_level.provenance.config_fields] == [
        "/detector/threshold"
    ]
    assert [field.source_path for field in embedded.provenance.config_fields] == [
        "/detector/threshold"
    ]


# REQ: PY-API-008
def test_embedded_graph_constructor_can_read_default_clock():
    class ConstructorClockReaderGraph(hs.Graph):
        """Read-only constructor access must not be classified as Graph authoring."""

        def __init__(self):
            super().__init__()
            self.constructor_clock = self.default_clock()

    # Key setup: Read the default selector while constructing an embedded Python Graph.
    graph = hs.Graph("constructor-clock-reader")
    embedded = graph.embed(ConstructorClockReaderGraph, "child")

    # Key assertion: The read survives embedding and retains the omitted/default identity.
    assert embedded.constructor_clock.is_default


# REQ: PY-API-008
def test_embedded_graph_constructor_authoring_is_rejected_eagerly():
    class ConstructorAuthoringGraph(hs.Graph):
        """Constructor-mutation fixture used only by this rejection test."""

        def __init__(self):
            super().__init__()
            self.source = self.op(COUNTER, "source", limit=1)

    # Key setup: Construct an embedded Graph type that authors before compose().
    graph = hs.Graph("constructor-authoring")

    # Key assertion: The binding rejects the unsupported pattern instead of discarding it.
    with pytest.raises(
        hs.errors.InvalidArgumentError,
        match="AUTHORING_GRAPH_CONSTRUCTOR_MUTATION_UNSUPPORTED",
    ):
        graph.embed(ConstructorAuthoringGraph, "child")


# REQ: PY-API-008
def test_embedded_graph_constructor_configured_clock_is_rejected_eagerly():
    class ConstructorClockGraph(hs.Graph):
        """Configured-clock constructor-mutation fixture for this test."""

        def __init__(self):
            super().__init__()
            self.clock = self.add_clock(
                hs.RealtimeClock,
                "lost-clock",
                initial_time_scale=2.0,
            )

    # Key setup: Use the configured-clock branch as the constructor's only authoring operation.
    graph = hs.Graph("constructor-clock-authoring")

    # Key assertion: The declaration is rejected eagerly instead of being silently discarded.
    with pytest.raises(
        hs.errors.InvalidArgumentError,
        match="AUTHORING_GRAPH_CONSTRUCTOR_MUTATION_UNSUPPORTED",
    ):
        graph.embed(ConstructorClockGraph, "child")


# REQ: PY-API-008
@pytest.mark.parametrize("exception", [ValueError("bad reusable graph"), KeyboardInterrupt()])
def test_embedded_graph_compose_preserves_python_exception_and_traceback(exception):
    class RaisingGraph(hs.Graph):
        """Compose-exception fixture used only by this propagation test."""

        def compose(self):
            raise exception

    # Key setup: Raise an ordinary exception or interrupt from an embedded compose() callback.
    graph = hs.Graph("raising-compose")

    # Key assertion: The original Python exception, message, and compose frame survive unchanged.
    with pytest.raises(type(exception)) as caught:
        graph.embed(RaisingGraph, "child")
    assert str(caught.value) == str(exception)
    assert any(frame.name == "compose" for frame in traceback.extract_tb(caught.tb))


# REQ: PY-API-008
def test_compose_exception_with_code_and_token_is_not_duck_typed_as_holoscan_error():
    class ForeignError(Exception):
        """Non-Holoscan exception that happens to carry similarly named attributes."""

        code = hs.errors.ErrorCode.INVALID_ARGUMENT
        token = "FOREIGN_TOKEN"

    class RaisingGraph(hs.Graph):
        """Foreign-error fixture used only by this exact-type translation test."""

        def compose(self):
            raise ForeignError("foreign failure")

    # Key setup: Raise a non-Holoscan exception carrying code and token attributes.
    graph = hs.Graph("foreign-compose-error")

    # Key assertion: Exact type identity preserves the foreign Python exception.
    with pytest.raises(ForeignError, match="foreign failure"):
        graph.embed(RaisingGraph, "child")


# REQ: PY-API-008
def test_embedded_graph_python_reference_cycle_is_collectable():
    class ParentCapturingGraph(hs.Graph):
        """Parent-cycle fixture used only by this garbage-collection test."""

        def __init__(self, *, parent):
            super().__init__()
            self.parent = parent

    # Key setup: Create the parent -> hosted child -> parent cycle described by the ownership model.
    graph = hs.Graph("collectable-cycle")
    graph.embed(ParentCapturingGraph, "child", parent=graph)
    graph_ref = weakref.ref(graph)
    del graph
    gc.collect()

    # Key assertion: All Python ownership edges are visible to cyclic GC.
    assert graph_ref() is None


# REQ: PY-API-008
#   invalid
#   `-- invalid[InvalidGraph]: out<uint64> -/-> DoubleIt.out<uint32>
def test_reusable_graph_interface_validation_is_eager():
    class InvalidGraph(hs.Graph):
        """Expose a mismatched output for test_reusable_graph_interface_validation_is_eager."""

        output = hs.Output("out", hs.payload.uint64)

        def compose(self):
            member = self.op(DOUBLER, "doubler")
            self.export_output(self.output, member.out)

    # Key setup: Embed a graph whose exported descriptor disagrees with its member port.
    graph = hs.Graph("invalid")

    # Key assertion: Composition rejects the interface mismatch during embed().
    with pytest.raises(hs.errors.FailureError, match="GRAPH_COMPOSE_FAILED"):
        graph.embed(InvalidGraph, "invalid")


# REQ: PY-API-008
#   recursive
#   `-- recursive[RecursiveGraph] --> self-embed rejected
def test_reusable_graph_logical_type_preserves_cycle_detection():
    class RecursiveGraph(hs.Graph):
        """Self-embed for test_reusable_graph_logical_type_preserves_cycle_detection."""

        def compose(self):
            self.embed(RecursiveGraph, "recursive")

    # Key setup: Embed a graph whose compose() recursively embeds its own logical type.
    graph = hs.Graph("recursive")

    # Key assertion: The logical graph identity preserves cycle detection.
    with pytest.raises(hs.errors.InvalidArgumentError, match="GRAPH_EMBED_CYCLE"):
        graph.embed(RecursiveGraph, "recursive")


# REQ: PY-API-008
#   root
#   `-- a[IndirectGraphA]
#       `-- b[IndirectGraphB]
#           `-- a[IndirectGraphA] [rejected]
def test_reusable_graph_logical_type_rejects_indirect_cycle():
    class IndirectGraphA(hs.Graph):
        """Mutual-recursion endpoint used by this indirect-cycle test."""

        def compose(self):
            self.embed(IndirectGraphB, "b")

    class IndirectGraphB(hs.Graph):
        """Mutual-recursion endpoint used by this indirect-cycle test."""

        def compose(self):
            self.embed(IndirectGraphA, "a")

    # Key setup: Embed a logical A -> B -> A reusable-Graph chain.
    graph = hs.Graph("indirect-recursion")

    # Key assertion: Hosted logical identity rejects the repeated ancestor type.
    with pytest.raises(hs.errors.InvalidArgumentError, match="GRAPH_EMBED_CYCLE"):
        graph.embed(IndirectGraphA, "a")


# REQ: PY-API-007
# REQ: PY-API-008
def test_embedded_region_duplicate_batch_is_atomic():
    # Key setup: Reject the same embedded Graph twice, then retry the valid membership.
    graph = hs.Graph("duplicate-child-region")
    child = graph.embed(DoublerGraph, "child")
    region = graph.region("stage")

    with pytest.raises(hs.errors.DuplicateNameError, match="GRAPH_REGION_MEMBER_DUPLICATE"):
        region.add(child, child)

    # Key assertion: The rejected batch did not commit its first child.
    region.add(child)


# REQ: PY-API-007
# REQ: PY-API-008
@pytest.mark.parametrize("operator_first", [True, False])
@pytest.mark.parametrize("retire_rejected_child", [True, False])
def test_mixed_region_batch_rejection_is_atomic(operator_first, retire_rejected_child):
    # Key setup: Put a foreign or retired child after valid mixed members in either order.
    graph = hs.Graph("mixed-child-region")
    operator = graph.op(COUNTER, "source", limit=1)
    child = graph.embed(DoublerGraph, "child")
    region = graph.region("stage")
    rejected_owner = hs.Graph("rejected-child-owner")
    rejected_child = rejected_owner.embed(DoublerGraph, "rejected")
    if retire_rejected_child:
        del rejected_owner
        gc.collect()
        assert not rejected_child.graph_valid

    corrected = (operator, child) if operator_first else (child, operator)
    with pytest.raises(hs.errors.InvalidArgumentError, match="GRAPH_REGION_HANDLE_INVALID"):
        region.add(*corrected, rejected_child)

    # Key assertion: Owner validation preceded one commit, so the corrected retry succeeds.
    region.add(*corrected)


# REQ: PY-API-008
def test_tensor_interface_descriptor_uses_registered_payload_token():
    class TensorInterfaceGraph(hs.Graph):
        """Minimal executable form of the reusable-Graph documentation descriptor."""

        input = hs.Input("input", hs.payload.TensorType)

    # Key assertion: The documented token form constructs without requiring a graph provider.
    assert TensorInterfaceGraph.input.payload_type == hs.payload.TensorType


def reusable_graph_plan(*, bind_direct_clock):
    """Build the graph compared by the two independent-C++-authoring tests below."""

    graph = hs.Graph("reusable-graphs-parity")
    source = graph.op(COUNTER, "source", limit=4)
    direct = graph.embed(DoublerGraph, "direct", copies=1)
    wrapped = graph.embed(WrappedDoublerGraph, "wrapped", copies=2)
    direct_sink = graph.op(SINK, "direct-sink")
    wrapped_sink = graph.op(SINK, "wrapped-sink")
    graph.add_flow(source.out, direct.input, queue_depth=4)
    graph.add_flow(source.out, wrapped.input, queue_depth=4)
    graph.add_flow(direct.output, direct_sink["in"], queue_depth=4)
    graph.add_flow(wrapped.output, wrapped_sink["in"], queue_depth=4)

    runtime_clock = graph.add_clock(hs.RealtimeClock, "runtime-clock")
    graph.set_default_clock(runtime_clock)
    if bind_direct_clock:
        graph.bind_clock(direct, direct.clock_import, runtime_clock)
    graph.bind_clock(wrapped, wrapped.clock_import, runtime_clock)
    (
        graph.region("stage")
        .add(direct)
        .add(wrapped)
        .constrain(hs.execution.RegionConstraint.bounded_memory())
    )
    return graph, direct, wrapped, hs.compile(graph)


def cpp_plan(mode):
    helper = os.environ["HOLOSCAN_REUSABLE_GRAPH_PLAN_HELPER"]
    return subprocess.run(
        [helper, mode],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


# REQ: PY-API-008
#   reusable-graphs-parity
#   |-- source --+--> direct[DoublerGraph] ------------> direct-sink
#   |            `--> wrapped[WrappedDoublerGraph] ----> wrapped-sink
#   `-- runtime-clock --> direct, wrapped
def test_reusable_graph_plan_matches_independent_cpp_authoring():
    # Key setup: Build the fully clock-bound Python reusable-graph plan.
    _, _, _, plan = reusable_graph_plan(bind_direct_clock=True)

    # Key assertion: Python and independent typed C++ authoring produce identical plans.
    assert plan.ok
    assert plan.json() == cpp_plan("bound")
    assert '"operator_path":"direct/doubler-0"' in plan.json()
    assert '"operator_path":"wrapped/inner/doubler-1"' in plan.json()


# REQ: PY-API-008
#   reusable-graphs-parity
#   |-- source --+--> direct[DoublerGraph] ------------> direct-sink
#   |            `--> wrapped[WrappedDoublerGraph] ----> wrapped-sink
#   `-- runtime-clock -/-> direct; runtime-clock --> wrapped
def test_unbound_reusable_clock_diagnostic_matches_cpp():
    # Key setup: Leave the direct child clock import unbound in the Python graph.
    _, _, _, plan = reusable_graph_plan(bind_direct_clock=False)

    # Key assertion: Python and C++ preserve the same unbound-clock diagnostic plan.
    assert not plan.ok
    diagnostic = next(
        diagnostic for diagnostic in plan.errors if diagnostic.token == "CLOCK_IMPORT_UNBOUND"
    )
    assert diagnostic.source.graph == 2
    assert diagnostic.json_pointer == "/clock_imports/0"
    assert (
        'Bind clock import "timeline" in graph "/direct" using bind_clock()' in diagnostic.required
    )
    assert 'immediate parent graph "_root"' in diagnostic.required
    assert 'clock import "timeline" in graph "/direct"' in diagnostic.provided
    assert "declared using import_clock()" in diagnostic.provided
    assert plan.json() == cpp_plan("unbound")


# REQ: PY-API-004
def test_partition_lookup_rejects_hosted_member_handle_identity():
    # Key setup: Retain a child-local Operator handle from an accepted root Graph plan.
    owner, direct, _, plan = reusable_graph_plan(bind_direct_clock=True)
    member = direct.members[0]
    assert owner is not None

    # Key assertion: A hosted member cannot path-match an Operator in the root plan authority.
    with pytest.raises(hs.errors.IncompatibleError) as caught:
        plan.partitions.of(member)
    assert caught.value.code is hs.errors.ErrorCode.INCOMPATIBLE


# REQ: PY-API-008
def test_registered_cpp_graph_discovery_and_interface_metadata():
    # Key setup: Enumerate and resolve the C++ reusable-graph proxy rows registered at import.
    graph_types = hs.registry.graph_types()
    names = tuple(graph_type.name for graph_type in graph_types)

    # Key assertion: Discovery preserves ordering, proxy identity, ports, and parameter metadata.
    assert names == tuple(sorted(names))
    assert "holoscan.test.ScalePairGraph" in names
    assert "holoscan.test.WrappedScaleGraph" in names
    assert hs.registry.resolve(ScalePairGraph.name) is ScalePairGraph
    assert ScalePairGraph.input.direction == "input"
    assert ScalePairGraph.output.payload_type == "holoscan.uint32"
    assert ScalePairGraph.factor.python_type == "int"
    assert ScalePairGraph.factor.value_range == (1, 1_000_000)
    assert ScalePairGraph.Handle is hs.GraphHandle
    assert ScalePairGraph.Params is dict


# REQ: PY-API-008
#   parameterized-interface
#   `-- child[ConditionalInterfaceGraph]
#       input --> primary --> output
#                    `----> debug-stage --> debug  (include_debug=True)
def test_registered_graph_ports_follow_real_parameterized_instance():
    # Key setup: Register a graph whose non-default constructor value adds an interface port.
    holoscan_test_support._register_conditional_graph_for_test()
    graph_type = hs.registry.resolve("holoscan.test.ConditionalInterfaceGraph")
    graph = hs.Graph("parameterized-interface")
    child = graph.embed(graph_type, "child", include_debug=True)

    # Key assertion: The handle caches the real child's interface, not a default-instance probe.
    assert "debug" not in graph_type.ports
    assert tuple(child.ports) == ("input", "output", "debug")
    assert child.debug.direction is hs.PortDirection.OUTPUT


# REQ: PY-API-008
def test_registered_graph_protocol_lookups_use_python_exception_types():
    # Key setup: Embed one registered child and probe a name outside its interface.
    graph = hs.Graph("registered-protocol-lookups")
    child = graph.embed(ScalePairGraph, "child")

    # Key assertion: Attribute and item protocols receive their native miss exception types.
    assert not hasattr(child, "missing")
    assert getattr(child, "missing", "fallback") == "fallback"
    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        _ = child.missing
    with pytest.raises(KeyError, match="missing"):
        child["missing"]


# REQ: PY-API-008
def test_registered_graph_discovery_isolates_failing_rows():
    # Key setup: Add a provider row whose default instance deliberately fails composition.
    holoscan_test_support._register_failing_graph_for_test()

    # Key assertion: Healthy graph types remain discoverable despite the unrelated bad row.
    names = {graph_type.name for graph_type in hs.registry.graph_types()}
    assert ScalePairGraph.name in names
    assert "holoscan.test.FailingDiscoveryGraph" not in names


# REQ: PY-API-008
def test_graph_aliases_invalidate_collisions_and_protect_module_names():
    # Key setup: Resolve one unique alias, then register a colliding leaf name and a reserved name.
    holoscan_test_support._register_scale_graph_alias_a_for_test()
    assert hs.graphs.CollisionGraph.name == "holoscan.test.a.CollisionGraph"
    holoscan_test_support._register_scale_graph_alias_b_for_test()
    holoscan_test_support._register_reserved_graph_alias_for_test()
    holoscan_test_support._register_helper_graph_alias_for_test()

    # Key assertion: Dynamic access drops ambiguous aliases and never clobbers module internals.
    assert not hasattr(hs.graphs, "CollisionGraph")
    with pytest.raises(AttributeError, match="CollisionGraph"):
        _ = hs.graphs.CollisionGraph
    assert not hasattr(hs.graphs, "GraphType")
    assert "CollisionGraph" not in hs.graphs.__all__
    assert "_aliases" not in hs.graphs.__all__
    assert callable(hs.graphs._aliases)
    imported = {}
    exec("from holoscan.graphs import *", imported)
    assert "_aliases" not in imported


# REQ: PY-API-008
def test_uninitialized_graph_proxy_rejects_protocol_probes_without_recursion():
    # Key setup: Reproduce the object state used by copy and pickle before __init__ runs.
    proxy = hs.registry.GraphType.__new__(hs.registry.GraphType)

    # Key assertion: Missing protocol hooks remain ordinary attribute misses.
    assert not hasattr(proxy, "__setstate__")
    assert repr(proxy) == "<GraphType '<uninitialized>'>"


def registered_reusable_graph_plan():
    """Build the registered-C++ graph used by the independent-authoring parity test."""

    graph = hs.Graph("registered-reusable-graphs-parity")
    source = graph.op(COUNTER, "source", limit=4)
    direct = ScalePairGraph(graph, "direct", factor=3)
    wrapped = graph.embed(WrappedScaleGraph.name, "wrapped", factor=5)
    direct_sink = graph.op(SINK, "direct-sink")
    wrapped_sink = graph.op(SINK, "wrapped-sink")
    graph.add_flow(source.out, direct.input, queue_depth=4)
    graph.add_flow(source.out, wrapped.input, queue_depth=4)
    graph.add_flow(direct.output, direct_sink["in"], queue_depth=4)
    graph.add_flow(wrapped.output, wrapped_sink["in"], queue_depth=4)

    runtime_clock = graph.add_clock(hs.RealtimeClock, "runtime-clock")
    graph.set_default_clock(runtime_clock)
    graph.bind_clock(direct, direct.imported_clock("timeline"), runtime_clock)
    graph.bind_clock(wrapped, wrapped.imported_clock("timeline"), runtime_clock)
    graph.region("stage").add(direct, wrapped).constrain(
        hs.execution.RegionConstraint.bounded_memory()
    )
    return graph, direct, wrapped, hs.compile(graph)


# REQ: PY-API-008
#   registered-reusable-graphs-parity
#   |-- source --+--> direct[ScalePairGraph] ----------> direct-sink
#   |            `--> wrapped[WrappedScaleGraph] ------> wrapped-sink
#   `-- runtime-clock --> direct, wrapped
def test_registered_cpp_graph_embedding_matches_independent_cpp_authoring():
    # Key setup: Embed registered C++ graphs through both proxy-type and runtime-name forms.
    _, direct, wrapped, plan = registered_reusable_graph_plan()

    # Key assertion: Python matches typed C++ while retaining nested graph and port metadata.
    assert plan.ok
    assert plan.json() == cpp_plan("registered")
    assert direct.type_name == "holoscan.test.ScalePairGraph"
    assert direct.path == ("direct",)
    assert direct["input"] is not None
    assert direct.ports["output"].payload_type == hs.payload.uint32
    assert wrapped.output.direction is hs.PortDirection.OUTPUT
    assert '"operator_path":"direct/scale"' in plan.json()
    assert '"operator_path":"wrapped/inner/scale"' in plan.json()


# REQ: PY-API-008
#   registered-failures
#   |-- unknown[DoesNotExist] [rejected]
#   `-- bad-params[ScalePairGraph factor=0] [rejected]
def test_registered_cpp_graph_failures_preserve_core_tokens():
    # Key setup: Attempt unknown-type and schema-invalid registered graph embeddings.
    graph = hs.Graph("registered-failures")

    # Key assertion: Registry and parameter failures retain their stable Core tokens.
    with pytest.raises(hs.errors.NotFoundError, match="AUTHORING_REGISTRY_UNKNOWN"):
        graph.embed("holoscan.test.DoesNotExist", "unknown")
    with pytest.raises(hs.errors.InvalidArgumentError, match="PARAM_VALIDATION_FAILED"):
        graph.embed(ScalePairGraph, "bad-params", factor=0)


# REQ: PY-API-008
#   registered-lifetime
#   `-- child[ScalePairGraph]
def test_registered_cpp_graph_handles_remain_weak():
    # Key setup: Retain registered child and interface handles after deleting the parent graph.
    graph = hs.Graph("registered-lifetime")
    child = graph.embed(ScalePairGraph, "child")
    interface_port = child.output

    del graph
    gc.collect()

    # Key assertion: Registered C++ handles remain weak and stale attribute probes are safe.
    assert not child.graph_valid
    assert child.path is None
    assert not interface_port.graph_valid
    assert not hasattr(child, "_repr_html_")
    with pytest.raises(AttributeError, match="has no attribute 'output'"):
        _ = child.output
