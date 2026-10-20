# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone Python Channel catalog and schema tests."""

import gc
import json
import threading
import weakref

import control_payload
import holoscan_test_support
import pytest
from channel_test_utils import unique_channel

import holoscan as hs


class ControlRelay(hs.Operator):
    """Relay one generated schema while changing an observable field."""

    inp = hs.Input("in", control_payload.ControlPayload)
    out = hs.Output("out", control_payload.ControlPayload)

    def setup(self, spec):
        spec.input(self.inp)
        spec.output(self.out)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        value = self.inp.receive_data()
        value.sequence += 1
        self.out.emit(value)


# REQ: PY-CHANNEL-001
# REQ: PY-CHANNEL-009
def test_name_based_generated_schema_round_trip():
    # Key setup: Use the generated Python dataclass as the required name-based payload token.
    name = unique_channel("python-channel-schema")
    options = hs.channel.ChannelOptions(enable_event_wake=True)
    value = control_payload.ControlPayload(sequence=9, gain=1.25, labels=["left", "right"])

    with hs.channel.Runtime() as runtime:
        subscriber = runtime.subscriber(name, control_payload.ControlPayload, options=options)
        publisher = runtime.publisher(name, control_payload.ControlPayload, options=options)
        publisher.update_connections()
        subscriber.refresh_connections()

        # Key assertion: Channel uses the same generated schema materializer as hosted ports.
        assert publisher.emit(value) == 1
        sample = subscriber.receive(hs.time.ms(500))
        assert sample.data == value
        publisher.close()


# REQ: PY-CHANNEL-009
def test_process_local_and_raw_pod_clients_remain_explicitly_unsupported():
    # Key setup: Resolve one process-local Python object row and one catalog-authored raw POD row.
    pod_endpoint = unique_channel("python-channel-pod-unsupported")
    pod_catalog = hs.topics.EndpointCatalog.parse(
        holoscan_test_support._pod_client_catalog(pod_endpoint)
    )

    with hs.channel.Runtime() as runtime:
        # Key assertion: Completed Tensor support does not accidentally admit payload families for
        # which the Python standalone client has no lossless public value representation.
        with pytest.raises(
            hs.errors.NotSupportedError, match="AUTHORING_CHANNEL_NAME_PAYLOAD_TYPE_UNSUPPORTED"
        ):
            runtime.publisher(unique_channel("python-channel-pyobject-unsupported"), hs.pyobject)
        with pytest.raises(
            hs.errors.NotSupportedError, match="PYTHON_CHANNEL_POD_PAYLOAD_UNSUPPORTED"
        ):
            runtime.publisher(pod_catalog, pod_endpoint)


# REQ: PY-CHANNEL-004
@pytest.mark.parametrize("retirement", ["close", "abandon"])
def test_schema_subscriber_concurrent_retirement_releases_python_type(retirement):
    # Key setup: Race wrapper release with both deterministic retirement paths.  The subscriber's
    # generated-schema class is intentionally ephemeral so its last reference is observable.
    for iteration in range(16):
        runtime = hs.channel.Runtime()

        class EphemeralControlPayload(control_payload.ControlPayload):
            pass

        schema_type = weakref.ref(EphemeralControlPayload)
        subscriber = runtime.subscriber(
            unique_channel(f"python-channel-schema-retirement-{retirement}-{iteration}"),
            EphemeralControlPayload,
        )
        start = threading.Barrier(2)

        def retire_runtime(runtime=runtime, start=start):
            start.wait()
            if retirement == "close":
                runtime.close()
            else:
                runtime.__exit__(RuntimeError, RuntimeError("abandon"), None)

        retiring = threading.Thread(target=retire_runtime)
        retiring.start()
        start.wait()
        del subscriber
        del EphemeralControlPayload
        retiring.join(timeout=2)
        assert not retiring.is_alive()
        gc.collect()

        # Key assertion: Endpoint retirement drops its schema-class reference without leaking it
        # or performing unsafe Python refcount work from the GIL-released retirement scope.
        assert schema_type() is None


# REQ: PY-CHANNEL-001
@pytest.mark.parametrize(
    ("payload_type", "value"),
    [
        (hs.payload.schema.String, {"data": "built-in"}),
        (hs.payload.schema.Time, {"time_ns": 17, "acqtime_ns": 19}),
    ],
)
def test_name_based_builtin_inline_schema_round_trip(payload_type, value):
    # Key setup: Built-in schema tokens use their SDK-owned native object APIs.
    name = unique_channel("python-channel-builtin-schema")
    options = hs.channel.ChannelOptions(enable_event_wake=True)
    with hs.channel.Runtime() as runtime:
        subscriber = runtime.subscriber(name, payload_type, options=options)
        publisher = runtime.publisher(name, payload_type, options=options)
        publisher.update_connections()
        subscriber.refresh_connections()

        # Key assertion: Built-in String and Time values share the ordinary schema carrier.
        assert publisher.emit(value) == 1
        assert subscriber.receive(1.0).data == value
        publisher.close()


# REQ: PY-CHANNEL-002
def test_catalog_inference_load_parse_and_compatibility(tmp_path):
    # Key setup: Compile a graph that transforms one catalog import into one catalog export.
    publisher_service = unique_channel("python-channel-catalog-in")
    subscriber_service = unique_channel("python-channel-catalog-out")
    publisher_endpoint = f"{publisher_service}-publisher"
    subscriber_endpoint = f"{subscriber_service}-subscriber"
    graph = hs.Graph("channel-client-catalog", default_clock=hs.RealtimeClock)
    imported = graph.import_topic(
        f"holomq://{publisher_service}", hs.payload.uint32, endpoint_name=publisher_endpoint
    )
    exported = graph.export_topic(
        f"holomq://{subscriber_service}", hs.payload.uint32, endpoint_name=subscriber_endpoint
    )
    doubler = graph.op("holoscan.test.DoubleIt", "doubler")
    graph.add_flow(imported.output, doubler["in"], queue_depth=2)
    graph.add_flow(doubler.out, exported.input, queue_depth=2)
    plan = hs.compile(graph)
    assert plan.ok, plan.json()
    plan_json = plan.json()

    catalog_path = tmp_path / "channel-catalog.json"
    plan.endpoint_catalog.save_json(catalog_path)
    publisher_catalog = hs.topics.EndpointCatalog.load(catalog_path)
    subscriber_catalog = hs.topics.EndpointCatalog.parse(plan.endpoint_catalog.json())
    publisher_view = publisher_catalog.endpoint(publisher_endpoint)
    subscriber_view = subscriber_catalog.endpoint(subscriber_endpoint)

    # Key assertion: Inspection retains immutable roles, identities, bounds, and catalog bytes.
    assert publisher_view.direction is hs.topics.TopicDirection.IMPORT
    assert subscriber_view.direction is hs.topics.TopicDirection.EXPORT
    assert publisher_view.scalar_identity == "uint32"
    assert subscriber_view.scalar_identity == "uint32"
    assert publisher_view.uri == f"holomq://{publisher_service}"
    assert json.loads(publisher_catalog.json())["scope"] == publisher_catalog.scope
    assert publisher_catalog.compatibility_with(publisher_catalog).compatible

    with hs.run_async(plan) as session, hs.channel.Runtime() as runtime:
        subscriber = runtime.subscriber(subscriber_catalog, subscriber_endpoint)
        publisher = runtime.publisher(publisher_catalog, publisher_endpoint)
        publisher.wait_for_subscribers(1.0)
        subscriber.refresh_connections()
        assert publisher.emit(17, timeout=1.0) == 1
        assert subscriber.receive(1.0).data == 34
        with pytest.raises(hs.errors.IncompatibleError, match="EXPLICIT_PAYLOAD_MISMATCH"):
            runtime.publisher(
                publisher_catalog, publisher_endpoint, payload_type=hs.payload.float32
            )
        session.request_stop()

    # Key assertion: Client attachment and traffic do not mutate the authored Plan artifact.
    assert plan.json() == plan_json


# REQ: PY-CHANNEL-001
# REQ: PY-CHANNEL-002
def test_catalog_schema_inference_materializes_the_generated_dataclass():
    # Key setup: Omit payload_type on both catalog forms for an imported generated schema.
    input_service = unique_channel("python-channel-schema-in")
    output_service = unique_channel("python-channel-schema-out")
    input_endpoint = f"{input_service}-publisher"
    output_endpoint = f"{output_service}-subscriber"
    graph = hs.Graph("channel-client-schema-catalog", default_clock=hs.RealtimeClock)
    payload = hs.payload.resolve(control_payload.ControlPayload._registry_name)
    imported = graph.import_topic(
        f"holomq://{input_service}",
        payload,
        endpoint_name=input_endpoint,
    )
    exported = graph.export_topic(
        f"holomq://{output_service}",
        payload,
        endpoint_name=output_endpoint,
    )
    relay = graph.op(ControlRelay, "relay")
    graph.add_flow(imported.output, relay["in"])
    graph.add_flow(relay.out, exported.input)
    plan = hs.compile(graph)
    assert plan.ok, plan.json()

    with hs.run_async(plan) as session, hs.channel.Runtime() as runtime:
        subscriber = runtime.subscriber(plan.endpoint_catalog, output_endpoint)
        publisher = runtime.publisher(plan.endpoint_catalog, input_endpoint)
        publisher.wait_for_subscribers(1.0)
        subscriber.refresh_connections()
        publisher.emit(control_payload.ControlPayload(sequence=13), timeout=1.0)
        sample = subscriber.receive(1.0)

        # Key assertion: The catalog identity recovers the imported facade without a type hint.
        assert sample.data == control_payload.ControlPayload(sequence=14)
        session.request_stop()
