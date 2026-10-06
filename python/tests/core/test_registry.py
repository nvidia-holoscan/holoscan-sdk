# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Operator registry discovery and proxy tests."""

import pydoc

import holoscan_test_support
import pytest

import holoscan as hs
import holoscan.operators as operators

CounterSource = hs.registry.resolve("holoscan.tests.CounterSource", kind="operator")
AttributeCollisionOperator = hs.registry.resolve(
    "holoscan.tests.AttributeCollisionOperator", kind="operator"
)
DoubleIt = hs.registry.resolve("holoscan.tests.DoubleIt", kind="operator")
ParameterProbe = hs.registry.resolve("holoscan.tests.ParameterProbe", kind="operator")
RequiredRegistrySource = hs.registry.resolve(
    "holoscan.tests.RequiredRegistrySource", kind="operator"
)
Sink = hs.registry.resolve("holoscan.tests.Sink", kind="operator")
TopicWindowFusion = hs.registry.resolve("holoscan.tests.TopicWindowFusion", kind="operator")


# REQ: PY-API-016
def test_registry_enumerates_and_resolves_operator_proxies():
    # Key setup: Enumerate the explicitly loaded private operator registry.
    proxies = hs.registry.operators()
    names = [proxy.name for proxy in proxies]

    # Key assertion: Enumeration is stable and qualified resolution preserves proxy identity.
    assert names == sorted(names)
    assert hs.registry.resolve("holoscan.tests.CounterSource") is CounterSource
    assert repr(CounterSource) == "<OperatorType 'holoscan.tests.CounterSource'>"


# REQ: PY-API-016
def test_operator_module_does_not_export_provider_leaf_names():
    # Key assertion: The Core module exposes its handle without leaking test-provider proxies.
    assert operators.__all__ == ["OperatorHandle"]
    assert not hasattr(operators, "CounterSource")
    assert not hasattr(operators, "DoubleIt")
    assert not hasattr(operators, "Sink")


# REQ: PY-API-016
def test_proxy_help_and_attributes_come_from_registry_records():
    # Key setup: Render help and inspect metadata for one registry proxy.
    source_help = pydoc.render_doc(CounterSource, renderer=pydoc.plaintext)

    # Key assertion: Help, ports, and parameters all derive from the registered row.
    assert "holoscan.tests.CounterSource" in source_help
    assert "Ports:" in source_help
    assert "out: Output[holoscan.uint32]" in source_help
    assert "Parameters:" in source_help
    assert "limit: int, range (1, 1000000)" in source_help
    assert CounterSource.out.payload_type == "holoscan.uint32"
    assert CounterSource.limit.value_range == (1, 1_000_000)
    assert "out" in dir(CounterSource)
    assert "limit" in dir(CounterSource)

    assert CounterSource.Handle is operators.OperatorHandle
    assert CounterSource.Params is dict
    assert Sink.in_.name == "in"
    assert "in_" in dir(Sink)
    assert "in" not in dir(Sink)

    assert ParameterProbe.sensor.nested_schema_id == "holoscan.tests.ParameterProbeSensor"
    assert ParameterProbe.sensor.nested_parameters is not None
    assert ParameterProbe.sensor.nested_parameters["channels"].python_type == "int"


# REQ: PY-API-018
def test_proxy_and_handle_attribute_collisions_follow_runtime_precedence():
    # Key setup: Create a legal row whose ports overlap fixed properties and a parameter.
    graph = hs.Graph("attribute-collisions")
    handle = AttributeCollisionOperator(graph, "collision", shared=3)

    # Key assertion: Fixed properties win, ports remain indexable, and a port wins over a
    # same-name parameter in dynamic attribute lookup.
    assert AttributeCollisionOperator.name == "holoscan.tests.AttributeCollisionOperator"
    assert set(AttributeCollisionOperator.ports) == {
        "class",
        "class_",
        "name",
        "ports",
        "shared",
    }
    assert AttributeCollisionOperator.shared is AttributeCollisionOperator.ports["shared"]
    assert AttributeCollisionOperator.class_ is AttributeCollisionOperator.ports["class_"]
    assert AttributeCollisionOperator.parameters["shared"].python_type == "int"
    assert handle.name == "collision"
    assert set(handle.ports) == {"class", "class_", "name", "ports", "shared"}
    assert isinstance(handle["name"], hs.Port)
    assert handle["name"].name == "name"
    assert isinstance(handle["ports"], hs.Port)
    assert handle["ports"].name == "ports"
    assert isinstance(handle.shared, hs.Port)
    assert handle.shared.name == "shared"
    assert handle["class"].name == "class"
    assert handle.class_.name == "class_"


# REQ: PY-API-016
# REQ: PY-API-018
def test_graph_accepts_operator_proxy_without_losing_string_form():
    # Key setup: Compose one pipeline using both proxy and qualified-string lookup.
    #
    #   source --uint32--> doubler --uint32--> sink
    graph = hs.Graph("proxy-authoring")
    source = CounterSource(graph, "source", limit=2)
    doubler = graph.op("holoscan.tests.DoubleIt", "doubler")
    sink = Sink(graph, "sink")
    graph.add_flow(source.out, doubler["in"])
    graph.add_flow(doubler.out, sink.in_)
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "clock"))

    # Key assertion: Mixed lookup forms retain one valid typed graph.
    assert isinstance(source, CounterSource.Handle)
    assert isinstance(sink, Sink.Handle)
    assert hs.compile(graph).ok


# REQ: PY-API-016
@pytest.mark.parametrize("entrypoint", ["proxy", "graph-proxy", "graph-name"])
def test_every_operator_entrypoint_enforces_required_parameters_before_mutation(entrypoint):
    # Key setup: Select each public spelling for constructing the same registered Operator.
    graph = hs.Graph(f"required-{entrypoint}")
    constructors = {
        "proxy": lambda **params: RequiredRegistrySource(graph, "source", **params),
        "graph-proxy": lambda **params: graph.op(RequiredRegistrySource, "source", **params),
        "graph-name": lambda **params: graph.op(RequiredRegistrySource.name, "source", **params),
    }

    # Key assertion: Omission has one diagnostic and leaves no partial Operator behind.
    with pytest.raises(
        TypeError,
        match=(
            r"holoscan\.tests\.RequiredRegistrySource missing required parameters: "
            r"required_count"
        ),
    ):
        constructors[entrypoint]()
    assert repr(graph) == f"<Graph 'required-{entrypoint}' operators=0 flows=0>"

    # Supplying the required value succeeds on the same unchanged graph.
    source = constructors[entrypoint](required_count=3)
    assert isinstance(source, RequiredRegistrySource.Handle)
    assert repr(graph) == f"<Graph 'required-{entrypoint}' operators=1 flows=0>"


# REQ: PY-API-016
def test_unknown_registry_name_reports_provider_hint_and_matches():
    # Key setup: Resolve a misspelled qualified test-operator name.
    with pytest.raises(KeyError) as caught:
        hs.registry.resolve("holoscan.tests.CounterSorce")

    message = str(caught.value)

    # Key assertion: The error suggests both provider loading and the closest known row.
    assert "did you import its provider module?" in message
    assert "holoscan.tests.CounterSource" in message


# REQ: PY-API-016
def test_registered_operator_discovery_isolates_failing_rows():
    # Key setup: Add an Operator row whose setup deliberately fails during metadata discovery.
    holoscan_test_support._register_failing_operator_for_test()

    # Key assertion: Healthy operators remain discoverable despite the unrelated bad row.
    names = {operator_type.name for operator_type in hs.registry.operators()}
    assert CounterSource.name in names
    assert "holoscan.tests.FailingDiscoveryOperator" not in names


# REQ: PY-API-016
def test_operator_module_remains_static_after_colliding_rows_are_registered():
    # Key setup: Register colliding, static-name, reserved-name, and keyword leaf rows.
    operator_handle = operators.OperatorHandle
    holoscan_test_support._register_operator_aliases_for_test()

    # Key assertion: Registry rows cannot hide or expand the static Core module surface.
    assert operators.OperatorHandle is operator_handle
    assert not hasattr(operators, "OperatorType")
    assert not hasattr(operators, "CollisionOperator")
    assert not hasattr(operators, "class")
    assert "OperatorHandle" in operators.__all__
    assert "OperatorType" not in operators.__all__
    assert "CollisionOperator" not in operators.__all__
    assert "class" not in operators.__all__


# REQ: PY-API-016
def test_uninitialized_operator_proxy_rejects_protocol_probes_without_recursion():
    # Key setup: Reproduce the object state used by copy and pickle before __init__ runs.
    proxy = hs.registry.OperatorType.__new__(hs.registry.OperatorType)

    # Key assertion: Missing protocol hooks remain ordinary attribute misses.
    assert not hasattr(proxy, "__setstate__")
    assert repr(proxy) == "<OperatorType '<uninitialized>'>"


# REQ: PY-API-016
def test_resolve_kind_limits_discovery_and_disambiguates_the_namespace():
    # Key setup: Resolve a known Operator through the explicit registry kind selector.
    resolved = hs.registry.resolve(CounterSource.name, kind="operator")

    # Key assertion: The selector returns only that kind and rejects invalid selector values.
    assert resolved is CounterSource
    with pytest.raises(KeyError, match="unknown registered type"):
        hs.registry.resolve(CounterSource.name, kind="graph")
    with pytest.raises(ValueError, match="kind must be"):
        hs.registry.resolve(CounterSource.name, kind="payload")


# REQ: PY-API-016
#   mismatch-suggestion
#   source[CounterSource].out --X--> fusion[TopicWindowFusion].camera
#                                      `--> fusion.imu is compatible
def test_payload_mismatch_suggests_a_compatible_port_on_the_real_operator():
    # Key setup: Connect a scalar source to the Tensor input of a multi-input Operator instance.
    graph = hs.Graph("mismatch-suggestion")
    source = graph.op(CounterSource, "source", limit=1)
    fusion = graph.op(TopicWindowFusion, "fusion")

    # Key assertion: The mismatch names the compatible input from that authored instance.
    with pytest.raises(hs.errors.IncompatibleError, match=r"Did you mean fusion\.imu\?"):
        graph.add_flow(source.out, fusion.camera)
