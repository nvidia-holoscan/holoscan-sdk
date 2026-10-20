# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External topic authoring tests."""

import ctypes
import gc
import json
import os
import subprocess
import uuid

import holoscan_test_support  # noqa: F401
import pytest

import holoscan as hs

CAMERA_SOURCE = "holoscan.test.TopicCameraSource"
WINDOW_FUSION = "holoscan.test.TopicWindowFusion"
IMU_ENDPOINT = "baseline.camera_imu.imu"
CONTROL_ENDPOINT = "baseline.camera_imu.control"
DESCRIPTOR_RELAY = "holoscan.test.DescriptorTensorRelay"
DESCRIPTOR_PAYLOAD = "tensor_topic.ExternalFrame"
SDK_DESCRIPTOR_SINK = "holoscan.test.SdkTensorPacketSink"

_descriptor_fixture = None
_sdk_descriptor_fixture = None


class FusionGraph(hs.Graph):
    """Fuse camera and imported IMU samples for the topic-plan parity test.

    Representative: test_camera_imu_topic_plan_matches_independent_cpp_authoring.
    """

    control = hs.Output("control", hs.payload.uint32)

    def compose(self):
        clock = self.add_clock(hs.RealtimeClock, "capture-clock")
        self.set_default_clock(clock)
        imu = self.import_topic(
            f"holomq://{IMU_ENDPOINT}",
            hs.payload.uint32,
            endpoint_name=IMU_ENDPOINT,
            capture_time=clock,
        )
        camera = self.op(CAMERA_SOURCE, "camera")
        fusion = self.op(WINDOW_FUSION, "window-fusion")
        self.add_flow(camera.frame, fusion.camera, queue_depth=4)
        self.add_flow(imu.output, fusion.imu, queue_depth=32)
        self.export_output(self.control, fusion.control)


def camera_imu_plan():
    """Build the Python side of the camera/IMU topic-plan parity test."""
    graph = hs.Graph("holoscan-camera-imu-control")
    graph.set_default_clock(graph.add_clock(hs.RealtimeClock, "runtime-clock"))
    control = graph.export_topic(
        f"holomq://{CONTROL_ENDPOINT}",
        hs.payload.uint32,
        endpoint_name=CONTROL_ENDPOINT,
    )
    fusion = graph.embed(FusionGraph, "fusion")
    graph.add_flow(fusion.control, control.input, queue_depth=8)
    return hs.compile(graph)


def cpp_plan(mode):
    """Return the independently authored C++ plan selected by ``mode``."""
    return subprocess.run(
        [os.environ["HOLOSCAN_TOPIC_PLAN_HELPER"], mode],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def host_u8_boundary(*, shape=(2, 4), copy_if_needed=False):
    """Build the uint8 Tensor contract used by descriptor-topic tests."""
    arguments = dict(
        representation=hs.tensor.TensorRepresentation(
            memory_kind=hs.tensor.MemoryKind.HOST,
            dtype=hs.tensor.DLDataType(hs.tensor.DLDataTypeCode.DLUINT, 8, 1),
            rank=len(shape),
            strides=hs.tensor.TensorStridePolicy.CONTIGUOUS_ONLY,
        ),
        bounds=hs.tensor.TensorBounds(max_shape=shape),
    )
    if copy_if_needed:
        arguments["transfer"] = hs.tensor.TensorTransferPolicy(
            foreign_emit_policy=hs.tensor.InterfaceTensorEmitPolicy.COPY_IF_NEEDED
        )
    return hs.tensor.TensorBoundaryContract(**arguments)


def host_f32_boundary(*, shape=(3,)):
    """Build the float32 calibration contract used by the peer test."""
    return hs.tensor.TensorBoundaryContract(
        representation=hs.tensor.TensorRepresentation(
            memory_kind=hs.tensor.MemoryKind.HOST,
            dtype=hs.tensor.DLDataType(hs.tensor.DLDataTypeCode.DLFLOAT, 32, 1),
            rank=len(shape),
            strides=hs.tensor.TensorStridePolicy.CONTIGUOUS_ONLY,
        ),
        bounds=hs.tensor.TensorBounds(max_shape=shape),
        transfer=hs.tensor.TensorTransferPolicy(
            foreign_emit_policy=hs.tensor.InterfaceTensorEmitPolicy.COPY_IF_NEEDED
        ),
    )


def declare_test_topic(graph, method, kind, label):
    """Declare one import/export variant used by constructor and compose tests."""
    payload = hs.payload.uint32
    options = {}
    if kind == "tensor":
        payload = hs.payload.TensorType
        options["tensor"] = host_u8_boundary()
    elif kind == "descriptor":
        payload = hs.payload.schema.TensorPacket
        options["tensor_fields"] = {"data": host_u8_boundary()}
    return getattr(graph, method)(f"holomq://{label}", payload, **options)


def install_descriptor_fixture():
    """Register and resolve the fixture payload used by the peer test."""
    global _descriptor_fixture
    if _descriptor_fixture is None:
        library_path = os.environ.get("HOLOSCAN_DESCRIPTOR_TOPIC_FIXTURE")
        if library_path is None:
            pytest.skip("descriptor Tensor topic fixture is unavailable")
        _descriptor_fixture = ctypes.CDLL(library_path)
        register = _descriptor_fixture.holoscan_python_register_descriptor_topic_fixture
        register.restype = ctypes.c_char_p
        error = register()
        assert error is None, error.decode()
    return hs.payload.resolve(DESCRIPTOR_PAYLOAD)


def install_sdk_descriptor_fixture():
    """Register the SDK TensorPacket sink used by the examples-independent test."""
    global _sdk_descriptor_fixture
    if _sdk_descriptor_fixture is None:
        library_path = os.environ.get("HOLOSCAN_SDK_DESCRIPTOR_TOPIC_FIXTURE")
        if library_path is None:
            pytest.skip("SDK descriptor Tensor topic fixture is unavailable")
        _sdk_descriptor_fixture = ctypes.CDLL(library_path)
        register = _sdk_descriptor_fixture.holoscan_python_register_sdk_descriptor_topic_fixture
        register.restype = ctypes.c_char_p
        error = register()
        assert error is None, error.decode()


# REQ: PY-API-009
def test_scalar_topics_accept_kwargs_and_options_objects():
    # Key setup: Connect imported and exported scalar topics through a registered transform.
    #
    #   holomq://input --> doubler --> holomq://output
    graph = hs.Graph("scalar-topics", default_clock=hs.RealtimeClock)
    clock = graph.default_clock()
    imported = graph.import_topic(
        "holomq://input",
        hs.payload.uint32,
        endpoint_name="input",
        capture_time=clock,
        attach=hs.topics.TopicAttachPolicy(max_wait=hs.time.ms(250)),
        bounds=hs.topics.EndpointBounds(
            publisher=hs.topics.PublisherEndpointBounds(max_loaned_samples=2)
        ),
    )
    exported = graph.export_topic(
        "holomq://output",
        hs.payload.uint32,
        hs.topics.TopicOptions(
            endpoint_name="output",
            bounds=hs.topics.EndpointBounds(
                subscriber=hs.topics.SubscriberEndpointBounds(queue_depth=2, max_borrowed_samples=1)
            ),
        ),
    )
    doubler = graph.op("holoscan.test.DoubleIt", "doubler")
    graph.add_flow(imported.output, doubler["in"], queue_depth=2)
    graph.add_flow(doubler.out, exported.input, queue_depth=2)

    plan = hs.compile(graph)

    # Key assertion: Both option forms preserve typed handles and compile without warnings.
    assert plan.ok
    assert imported.output.name == "output"
    assert exported.input.name == "input"
    assert imported.output.payload_type == hs.payload.uint32
    assert exported.input.payload_type == hs.payload.uint32
    assert not plan.warnings


# REQ: PY-API-008
@pytest.mark.parametrize("method", ["import_topic", "export_topic"])
@pytest.mark.parametrize("kind", ["scalar", "tensor", "descriptor"])
def test_embedded_topic_constructor_authoring_is_rejected(method, kind):
    class ConstructorTopicGraph(hs.Graph):
        """Topic-only constructor mutation fixture for this test."""

        def __init__(self):
            super().__init__()
            self.topic = declare_test_topic(self, method, kind, f"constructor-{method}-{kind}")

    # Key setup: Make one successful topic declaration the constructor's only authoring operation.
    parent = hs.Graph(f"constructor-{method}-{kind}-parent")

    # Key assertion: Every dispatch path is rejected eagerly instead of being silently discarded.
    with pytest.raises(
        hs.errors.InvalidArgumentError,
        match="AUTHORING_GRAPH_CONSTRUCTOR_MUTATION_UNSUPPORTED",
    ):
        parent.embed(ConstructorTopicGraph, "child")


# REQ: PY-API-008
@pytest.mark.parametrize("method", ["import_topic", "export_topic"])
@pytest.mark.parametrize("kind", ["scalar", "tensor", "descriptor"])
def test_embedded_topic_compose_authoring_succeeds(method, kind):
    class ComposedTopicGraph(hs.Graph):
        """Topic-in-compose fixture for this test."""

        def compose(self):
            self.topic = declare_test_topic(self, method, kind, f"compose-{method}-{kind}")

    # Key setup: Author each topic variant from the supported reusable-Graph compose hook.
    parent = hs.Graph(f"compose-{method}-{kind}-parent")
    child = parent.embed(ComposedTopicGraph, "child")

    # Key assertion: Successful compose-time declarations remain live on the hosted Graph.
    assert child.graph_valid
    assert child.topic.graph_valid


# REQ: PY-API-008
def test_failed_topic_constructor_declaration_does_not_mark_local_authoring():
    class FailedConstructorTopicGraph(hs.Graph):
        """Rejected constructor declaration followed by valid compose authoring."""

        def __init__(self):
            super().__init__()
            try:
                self.import_topic(
                    "holomq://missing-descriptor-contract",
                    hs.payload.schema.TensorPacket,
                )
            except hs.errors.InvalidArgumentError as error:
                self.failure_token = error.token

        def compose(self):
            self.topic = self.import_topic("holomq://compose-after-failure", hs.payload.uint32)

    # Key setup: Catch a pre-mutation declaration failure in the child constructor.
    parent = hs.Graph("failed-constructor-topic-parent")
    child = parent.embed(FailedConstructorTopicGraph, "child")

    # Key assertion: Only successful mutation trips the constructor guard.
    assert child.failure_token == "AUTHORING_DESCRIPTOR_TENSOR_CONTRACT_REQUIRED"
    assert child.topic.graph_valid


# REQ: PY-API-009
def test_topic_option_values_are_mutable_and_none_resets_to_defaults():
    # Key setup: Mutate nested option values through their owning Python objects.
    graph = hs.Graph("topic-option-values")
    clock = graph.add_clock(hs.RealtimeClock, "capture-clock")
    publisher = hs.topics.PublisherEndpointBounds(max_loaned_samples=2)
    bounds = hs.topics.EndpointBounds(publisher=publisher)
    options = hs.topics.TopicOptions(
        endpoint_name="input",
        memory_conversion=hs.tensor.MemoryConversionPolicy.ALLOW_IF_SAFE,
        schema=hs.payload.schema.String,
        capture_time=clock,
        clock_health=hs.topics.ClockHealthOptions(max_report_age=hs.time.ms(100)),
        attach=hs.topics.TopicAttachPolicy(max_wait=hs.time.ms(250)),
        bounds=bounds,
    )
    bounds.publisher.max_loaned_samples = 8
    options.clock_health.max_report_age = hs.time.ms(200)
    options.attach.max_wait = hs.time.ms(500)
    options.bounds.publisher.max_loaned_samples = 9

    # Key assertion: Nested writes persist, values are inspectable, and None resets fields.
    assert bounds.publisher.max_loaned_samples == 8
    assert options.endpoint_name == "input"
    assert options.memory_conversion is hs.tensor.MemoryConversionPolicy.ALLOW_IF_SAFE
    assert options.schema == hs.payload.schema.String
    assert options.capture_time.name == "capture-clock"
    assert options.clock_health.max_report_age.nanoseconds == 200_000_000
    assert options.attach.max_wait.nanoseconds == 500_000_000
    assert options.bounds.publisher.max_loaned_samples == 9

    options.clock_health.max_report_age = None
    options.attach.max_wait = None
    options.bounds.publisher.max_loaned_samples = None
    assert options.clock_health.max_report_age is None
    assert options.attach.max_wait is None
    assert options.bounds.publisher.max_loaned_samples is None

    options.endpoint_name = None
    options.memory_conversion = None
    options.schema = None
    options.capture_time = None
    options.clock_health = None
    options.attach = None
    options.bounds = None
    assert options.endpoint_name is None
    assert options.memory_conversion is None
    assert options.schema is None
    assert options.capture_time is None
    assert options.clock_health is None
    assert options.attach.peer is hs.topics.PeerRequirement.OPTIONAL
    assert options.attach.max_wait is None
    assert options.bounds.publisher is None
    assert options.bounds.subscriber is None

    graph.import_topic("holomq://none-options", hs.payload.uint32, attach=None, bounds=None)
    assert hs.compile(graph).ok


# REQ: PY-API-009
def test_topic_options_report_topic_specific_type_errors():
    # Key setup: Supply a connection enum and a non-payload registry token to topic APIs.
    graph = hs.Graph("topic-option-types")

    # Key assertion: Topic option and payload mismatches fail at their Python authoring boundaries.
    with pytest.raises(
        hs.errors.InvalidArgumentError,
        match="AUTHORING_TOPIC_OPTION_TYPE_MISMATCH field_name=memory_conversion",
    ):
        graph.import_topic("holomq://wrong-memory", hs.payload.uint32, memory_conversion="REQUIRE")
    with pytest.raises(TypeError):
        graph.import_topic("holomq://wrong-payload", hs.RealtimeClock)
    with pytest.raises(
        hs.errors.InvalidArgumentError, match="AUTHORING_PAYLOAD_TYPE_TOKEN_REQUIRED"
    ):
        hs.Input("input", hs.RealtimeClock)
    with pytest.raises(
        hs.errors.InvalidArgumentError, match="AUTHORING_PAYLOAD_TYPE_TOKEN_REQUIRED"
    ):
        hs.Output("output", hs.RealtimeClock)


# REQ: PY-API-009
def test_inline_schema_topics_use_registered_payload_identity(tmp_path):
    # Key setup: Import a generated-schema payload and persist its compiled endpoint catalog.
    #
    #   holomq://strings --> schema-topic
    payload = hs.payload.schema.String
    graph = hs.Graph("schema-topic")
    topic = graph.import_topic(
        "holomq://strings",
        payload,
        endpoint_name="strings",
        schema=payload,
    )

    plan = hs.compile(graph)
    catalog = json.loads(plan.endpoint_catalog.json())

    # Key assertion: The catalog records the complete schema identity and saves deterministically.
    assert plan.ok
    assert topic.output.payload_type == payload
    assert catalog["services"][0]["payload"]["family"] == "schema"
    assert catalog["services"][0]["payload"]["schema_identity"]["canonical_name"] == (
        "holoscan.schema.String"
    )
    payload_catalog = catalog["services"][0]["payload"]
    assert {"family", "writer_max_serialized_bytes", "schema_identity"} <= set(payload_catalog)
    assert set(payload_catalog["schema_identity"]) == {
        "canonical_name",
        "compatibility_epoch",
        "family_id",
        "file_identifier",
    }

    catalog_path = tmp_path / "catalog.json"
    plan.endpoint_catalog.save_json(catalog_path)
    assert catalog_path.read_text() == plan.endpoint_catalog.json()
    plan.endpoint_catalog.save_json(catalog_path)
    assert catalog_path.read_text() == plan.endpoint_catalog.json()


# REQ: PY-API-014
def test_plain_tensor_topic_matches_independent_cpp_authoring():
    # Key setup: Import a bounded host Tensor topic without a descriptor schema.
    #
    #   holomq://tensor-input --> tensor-topic
    graph = hs.Graph("tensor-topic")
    topic = graph.import_topic(
        "holomq://tensor-input",
        hs.payload.TensorType,
        endpoint_name="tensor-input",
        tensor=host_u8_boundary(),
    )

    plan = hs.compile(graph)

    # Key assertion: Python and independently authored C++ preserve the same Tensor plan.
    assert plan.ok
    assert hs.payload.TensorType == hs.payload.resolve(hs.payload.Tensor._registry_name)
    assert topic.output.payload_type == hs.payload.TensorType
    assert plan.json() == cpp_plan("tensor")


# REQ: PY-API-014
def test_tensor_boundary_validation_uses_core_contract_rules():
    # Key setup: Give an imported Tensor topic a zero-byte logical bound.
    #
    #   holomq://bad-tensor -X-> bad-tensor-topic
    graph = hs.Graph("bad-tensor-topic")
    contract = hs.tensor.TensorBoundaryContract(
        representation=hs.tensor.TensorRepresentation(
            memory_kind=hs.tensor.MemoryKind.HOST,
            rank=1,
        ),
        bounds=hs.tensor.TensorBounds(max_rank=1, max_byte_span=0),
    )

    # Key assertion: Topic authoring reuses Core's stable Tensor-bound rejection.
    with pytest.raises(hs.errors.InvalidArgumentError, match="GRAPH_TENSOR_LOGICAL_BOUNDS_INVALID"):
        graph.import_topic("holomq://bad-tensor", hs.payload.TensorType, tensor=contract)


# REQ: PY-API-014
def test_tensor_boundary_values_have_consistent_owned_mutation_semantics():
    # Key setup: Mutate nested scalar values and replace immutable Tensor-bound sequences.
    source_representation = hs.tensor.TensorRepresentation(
        memory_kind=hs.tensor.MemoryKind.HOST,
        rank=2,
    )
    contract = hs.tensor.TensorBoundaryContract(
        representation=source_representation,
        bounds=hs.tensor.TensorBounds(max_shape=[2, 4], max_abs_byte_strides=[4, 1]),
        transfer=hs.tensor.TensorTransferPolicy(
            foreign_emit_policy=hs.tensor.InterfaceTensorEmitPolicy.COPY_IF_NEEDED
        ),
    )
    source_representation.rank = 7
    contract.representation.rank = 3
    contract.transfer.max_inline_bytes = 128

    # Key assertion: Nested state aliases its owner; sequence snapshots reject in-place edits.
    assert contract.representation.rank == 3
    assert contract.bounds.max_shape == (2, 4)
    assert contract.bounds.max_abs_byte_strides == (4, 1)
    assert contract.transfer.max_inline_bytes == 128
    with pytest.raises(TypeError):
        contract.bounds.max_shape[0] = 8
    with pytest.raises(AttributeError):
        contract.bounds.max_shape.append(8)
    contract.bounds.max_shape = [3, 5]
    contract.bounds.max_abs_byte_strides = [5, 1]
    assert contract.bounds.max_shape == (3, 5)
    assert contract.bounds.max_abs_byte_strides == (5, 1)

    contract.transfer = None
    contract.memory_conversion = hs.tensor.MemoryConversionPolicy.REQUIRE
    assert contract.transfer is None
    assert contract.memory_conversion is hs.tensor.MemoryConversionPolicy.REQUIRE


# REQ: PY-API-014
def test_topic_options_tensor_property_owns_mutable_admitted_contract():
    # Key setup: Store an invalid source contract, repair the owned nested value, and retain it.
    source = host_u8_boundary(shape=(0, 4))
    options = hs.topics.TopicOptions(tensor=source)
    source.bounds.max_shape = [9, 9]
    retained = options.tensor

    # Key assertion: The source is copied while nested access addresses the admitted value.
    assert retained is not source
    assert retained.bounds.max_shape == (0, 4)
    retained.bounds.max_shape = [3, 5]
    graph = hs.Graph("mutable-tensor-topic-options")
    graph.import_topic("holomq://mutable-tensor", hs.payload.TensorType, options)
    assert hs.compile(graph).ok

    replacement = host_u8_boundary(shape=(4, 6))
    options.tensor = replacement
    replacement.bounds.max_shape = [7, 7]
    assert options.tensor.bounds.max_shape == (4, 6)
    options.tensor = None
    assert options.tensor is None


# REQ: PY-API-014
def test_topic_options_tensor_fields_property_owns_mutable_admitted_contracts():
    # Key setup: Retain one stored descriptor-field contract after isolating its source value.
    install_sdk_descriptor_fixture()
    source = host_u8_boundary(shape=(0, 4))
    options = hs.topics.TopicOptions(tensor_fields={"data": source})
    source.bounds.max_shape = [9, 9]
    retained = options.tensor_fields["data"]

    # Key assertion: Mapping values mutate the stored contract used by descriptor admission.
    assert retained is not source
    assert retained.bounds.max_shape == (0, 4)
    retained.bounds.max_shape = [3, 5]
    graph = hs.Graph("mutable-descriptor-topic-options", default_clock=hs.RealtimeClock)
    imported = graph.import_topic(
        "holomq://mutable-descriptor", hs.payload.schema.TensorPacket, options
    )
    sink = graph.op(SDK_DESCRIPTOR_SINK, "mutable-descriptor-sink")
    graph.add_flow(imported.output, sink.input, queue_depth=1)
    plan = hs.compile(graph)
    assert plan.ok, [diagnostic.token for diagnostic in plan.diagnostics]

    replacement = host_u8_boundary(shape=(4, 6))
    options.tensor_fields = {"data": replacement}
    replacement.bounds.max_shape = [7, 7]
    assert options.tensor_fields["data"].bounds.max_shape == (4, 6)
    options.tensor_fields = None
    assert options.tensor_fields is None


# REQ: PY-API-014
@pytest.mark.parametrize("descriptor", [False, True])
def test_retained_tensor_children_survive_replacement_reset_and_owner_collection(descriptor):
    # Key setup: Retain nested values from an owned contract before replacing and clearing it.
    options = hs.topics.TopicOptions()
    source = host_u8_boundary(shape=(2, 4))
    if descriptor:
        options.tensor_fields = {"data": source}
        retained = options.tensor_fields["data"]
    else:
        options.tensor = source
        retained = options.tensor
    bounds = retained.bounds
    representation = retained.representation
    del retained, source

    replacement = host_u8_boundary(shape=(5, 6))
    if descriptor:
        options.tensor_fields = {"data": replacement}
        assert options.tensor_fields["data"].bounds.max_shape == (5, 6)
        options.tensor_fields = None
    else:
        options.tensor = replacement
        assert options.tensor.bounds.max_shape == (5, 6)
        options.tensor = None
    del options, replacement
    gc.collect()

    # Key assertion: Retained nested values own their state independently of the original option.
    assert bounds.max_shape == (2, 4)
    assert representation.rank == 2
    bounds.max_shape = (7, 8)
    representation.rank = 3
    assert bounds.max_shape == (7, 8)
    assert representation.rank == 3


# REQ: PY-API-014
@pytest.mark.parametrize("descriptor", [False, True])
def test_self_assignment_keeps_owned_tensor_contracts(descriptor):
    # Key setup: Assign an owned Tensor option value back to the same option.
    source = host_u8_boundary(shape=(2, 4))
    options = hs.topics.TopicOptions()
    if descriptor:
        options.tensor_fields = {"data": source}
        options.tensor_fields = options.tensor_fields
        retained = options.tensor_fields["data"]
    else:
        options.tensor = source
        options.tensor = options.tensor
        retained = options.tensor

    # Key assertion: Self-assignment preserves the owned value and remains isolated from the source.
    source.bounds.max_shape = (9, 9)
    assert retained.bounds.max_shape == (2, 4)


# REQ: PY-API-014
@pytest.mark.parametrize("descriptor", [False, True])
def test_invalid_tensor_property_assignment_preserves_previous_value(descriptor):
    # Key setup: Start with a valid option and attempt a replacement that fails partway through.
    options = hs.topics.TopicOptions()
    original = host_u8_boundary(shape=(2, 4))
    if descriptor:
        install_sdk_descriptor_fixture()
        options.tensor_fields = {"data": original}
        with pytest.raises(hs.errors.InvalidArgumentError):
            options.tensor_fields = {"data": host_u8_boundary(shape=(7, 8)), "bad": object()}

        # Key assertion: Descriptor admission still observes the complete original mapping.
        graph = hs.Graph("rejected-options-update", default_clock=hs.RealtimeClock)
        imported = graph.import_topic(
            "holomq://rejected-update", hs.payload.schema.TensorPacket, options
        )
        sink = graph.op(SDK_DESCRIPTOR_SINK, "sink")
        graph.add_flow(imported.output, sink.input, queue_depth=1)
        plan = hs.compile(graph)
        assert plan.ok, [diagnostic.token for diagnostic in plan.diagnostics]
        catalog = json.loads(plan.endpoint_catalog.json())
        tensor = catalog["services"][0]["payload"]["descriptor_tensor_fields"][0]["contract"]
        assert tuple(axis["max_extent"] for axis in tensor["bounds"]["axes"]) == (2, 4)
    else:
        options.tensor = original
        with pytest.raises(hs.errors.InvalidArgumentError):
            options.tensor = object()

        # Key assertion: Scalar Tensor options also preserve the prior owned contract.
        assert options.tensor is not None
        assert options.tensor.bounds.max_shape == (2, 4)


# REQ: PY-API-014
def test_tensor_boundary_value_errors_name_the_exact_field():
    # Key setup: Pass invalid values to each sequence and policy conversion boundary.

    # Key assertion: Every failure uses the topic-option token and exact keyword name.
    for keyword in ("max_shape", "max_abs_byte_strides"):
        with pytest.raises(
            hs.errors.InvalidArgumentError,
            match=rf"AUTHORING_TOPIC_OPTION_TYPE_MISMATCH field_name={keyword}",
        ):
            hs.tensor.TensorBounds(**{keyword: "invalid"})
    with pytest.raises(
        hs.errors.InvalidArgumentError,
        match="AUTHORING_TOPIC_OPTION_TYPE_MISMATCH field_name=foreign_emit_policy",
    ):
        hs.tensor.TensorTransferPolicy(foreign_emit_policy="COPY_IF_NEEDED")


# REQ: PY-API-014
def test_sdk_descriptor_tensor_topic_records_named_field_contracts():
    # Key setup: Connect an SDK descriptor-Tensor topic to a test-only typed sink.
    #
    #   holomq://tensor-packet --> sdk-sink
    install_sdk_descriptor_fixture()
    graph = hs.Graph("descriptor-tensor-topic", default_clock=hs.RealtimeClock)
    payload = hs.payload.schema.TensorPacket
    imported = graph.import_topic(
        "holomq://tensor-packet",
        payload,
        endpoint_name="tensor-packet",
        tensor_fields={"data": host_u8_boundary()},
    )
    sink = graph.op(SDK_DESCRIPTOR_SINK, "sdk-sink")
    graph.add_flow(imported.output, sink.input, queue_depth=1)

    plan = hs.compile(graph)
    catalog = json.loads(plan.endpoint_catalog.json())

    # Key assertion: The SDK descriptor path is admitted and retains its generated field ID.
    assert plan.ok, [diagnostic.token for diagnostic in plan.diagnostics]
    assert imported.output.payload_type == payload
    assert catalog["services"][0]["payload"]["transport_profile_tag"] != 0
    assert catalog["services"][0]["payload"]["descriptor_tensor_fields"][0]["field_id"] == 3


# REQ: PY-API-014
def test_descriptor_tensor_topic_dispatch_rejects_incompatible_contract_forms():
    # Key setup: Exercise both import and export dispatch with missing or misplaced field contracts.
    graph = hs.Graph("descriptor-tensor-dispatch-errors")
    descriptor = hs.payload.schema.TensorPacket
    fields = {"data": host_u8_boundary()}

    # Key assertion: Each rejected contract form reports its descriptor-specific authoring token.
    for declare, uri in (
        (graph.import_topic, "holomq://missing-import"),
        (graph.export_topic, "holomq://missing-export"),
    ):
        with pytest.raises(
            hs.errors.InvalidArgumentError,
            match="AUTHORING_DESCRIPTOR_TENSOR_CONTRACT_REQUIRED",
        ):
            declare(uri, descriptor)

    for declare, uri in (
        (graph.import_topic, "holomq://scalar-import"),
        (graph.export_topic, "holomq://scalar-export"),
    ):
        with pytest.raises(
            hs.errors.InvalidArgumentError,
            match="AUTHORING_DESCRIPTOR_TENSOR_PAYLOAD_REQUIRED",
        ):
            declare(uri, hs.payload.uint32, tensor_fields=fields)

    graph.import_topic("holomq://none-fields", hs.payload.uint32, tensor_fields=None)
    assert hs.compile(graph).ok


# REQ: PY-API-014
def test_descriptor_tensor_topic_reports_known_fields():
    # Key setup: Request an unknown Tensor field from an SDK descriptor payload.
    #
    #   holomq://tensor-packet -X-> descriptor-tensor-field-error
    graph = hs.Graph("descriptor-tensor-field-error")

    # Key assertion: The rejection reports the payload's known descriptor field names.
    with pytest.raises(
        hs.errors.NotFoundError,
        match=r"AUTHORING_DESCRIPTOR_TENSOR_FIELD_UNKNOWN.*known_fields=data",
    ):
        graph.import_topic(
            "holomq://tensor-packet",
            hs.payload.schema.TensorPacket,
            tensor_fields={"pixels": host_u8_boundary()},
        )
    with pytest.raises(
        hs.errors.NotFoundError,
        match=r"AUTHORING_DESCRIPTOR_TENSOR_FIELD_UNKNOWN.*known_fields=data",
    ):
        graph.export_topic(
            "holomq://tensor-packet-output",
            hs.payload.schema.TensorPacket,
            tensor_fields={"pixels": host_u8_boundary()},
        )


# REQ: PY-API-014
def test_descriptor_tensor_topic_rejects_partial_generated_inventory():
    # Key setup: Provide only one field contract for a registered two-Tensor descriptor payload.
    payload = install_descriptor_fixture()
    graph = hs.Graph("partial-descriptor-inventory")

    # Key assertion: Core rejects the incomplete generated field inventory before compilation.
    with pytest.raises(
        hs.errors.InvalidArgumentError,
        match="GRAPH_DESCRIPTOR_TENSOR_TOPIC_INVENTORY_INCOMPLETE",
    ):
        graph.import_topic(
            "holomq://partial-descriptor",
            payload,
            tensor_fields={"pixels": host_u8_boundary()},
        )


# REQ: PY-API-014
# REQ: SCHEMA-PY-003
def test_descriptor_tensor_topic_interoperates_with_cpp_peer(tmp_path):
    # Key setup: Relay two named Tensor fields between a Python graph and C++ peer.
    #
    #   C++ publisher --> relay --> C++ subscriber
    payload = install_descriptor_fixture()
    instance = uuid.uuid4().hex
    input_name = f"descriptor-tensor-import-{instance}"
    output_name = f"descriptor-tensor-export-{instance}"
    attach = hs.topics.TopicAttachPolicy(
        peer=hs.topics.PeerRequirement.REQUIRED,
        max_wait=hs.time.ms(10_000),
    )
    tensor_fields = {
        "pixels": host_u8_boundary(copy_if_needed=True),
        "calibration": host_f32_boundary(),
    }

    graph = hs.Graph("descriptor-tensor-external-topic", default_clock=hs.RealtimeClock)
    imported = graph.import_topic(
        f"holomq://{input_name}",
        payload,
        endpoint_name=input_name,
        attach=attach,
        bounds=hs.topics.EndpointBounds(
            publisher=hs.topics.PublisherEndpointBounds(max_loaned_samples=2)
        ),
        tensor_fields=tensor_fields,
    )
    exported = graph.export_topic(
        f"holomq://{output_name}",
        payload,
        endpoint_name=output_name,
        attach=attach,
        bounds=hs.topics.EndpointBounds(
            subscriber=hs.topics.SubscriberEndpointBounds(queue_depth=1, max_borrowed_samples=1)
        ),
        tensor_fields=tensor_fields,
    )
    relay = graph.op(DESCRIPTOR_RELAY, "relay")
    graph.add_flow(imported.output, relay.input, queue_depth=2)
    graph.add_flow(relay.output, exported.input, queue_depth=1)

    plan = hs.compile(graph)

    # Key assertion: Profiles reject the mismatch peer and admit the valid round trip unchanged.
    assert plan.ok, plan.json()
    catalog = json.loads(plan.endpoint_catalog.json())
    for service in catalog["services"]:
        payload_record = service["payload"]
        assert payload_record["transport_profile_tag"] != 0
        assert [field["field_id"] for field in payload_record["descriptor_tensor_fields"]] == [
            2,
            3,
        ]

    catalog_path = tmp_path / "descriptor-catalog.json"
    plan.endpoint_catalog.save_json(catalog_path)

    mismatch = subprocess.run(
        [
            os.environ["HOLOSCAN_DESCRIPTOR_TOPIC_MISMATCH_PEER"],
            catalog_path,
            input_name,
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert mismatch.returncode == 0, mismatch.stderr
    assert "profile mismatch rejected" in mismatch.stdout

    with hs.run_async(plan) as session:
        client = subprocess.run(
            [
                os.environ["HOLOSCAN_DESCRIPTOR_TOPIC_CLIENT"],
                "--catalog",
                catalog_path,
                "--instance",
                instance,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )

    assert client.returncode == 0, client.stderr
    assert "descriptor Tensor external-topic client complete" in client.stdout
    assert session.termination is hs.RunTermination.STOPPED
    assert session.diagnostics == []


# REQ: PY-API-009
def test_endpoint_catalog_save_accepts_native_filesystem_paths(tmp_path):
    # Key setup: Construct equivalent bytes and surrogate-escaped paths on the host filesystem.
    graph = hs.Graph("filesystem-catalog")
    graph.import_topic("holomq://filesystem", hs.payload.uint32, endpoint_name="filesystem")
    plan = hs.compile(graph)
    path_bytes = os.fsencode(tmp_path) + b"/catalog-\xff.json"
    path_text = os.fsdecode(path_bytes)

    # Key assertion: Both legal os.PathLike representations preserve the canonical catalog bytes.
    plan.endpoint_catalog.save_json(path_bytes)
    with open(path_bytes, "rb") as stream:
        assert stream.read().decode() == plan.endpoint_catalog.json()
    plan.endpoint_catalog.save_json(path_text)
    with open(path_bytes, "rb") as stream:
        assert stream.read().decode() == plan.endpoint_catalog.json()


# REQ: PY-API-009
def test_topic_handles_do_not_extend_graph_lifetime():
    # Key setup: Retain imported/exported topic handles after releasing their owning graph.
    #
    #   holomq://input --> retired-topics
    #   retired-topics --> holomq://output
    graph = hs.Graph("retired-topics")
    imported = graph.import_topic("holomq://input", hs.payload.uint32, endpoint_name="input")
    exported = graph.export_topic("holomq://output", hs.payload.uint32, endpoint_name="output")
    output = imported.output
    input_ = exported.input

    del graph
    gc.collect()

    # Key assertion: Every weak topic handle fails closed after graph retirement.
    assert not imported.graph_valid
    assert not exported.graph_valid
    assert not output.graph_valid
    assert not input_.graph_valid
    replacement = hs.Graph("replacement")
    with pytest.raises(hs.errors.InvalidArgumentError, match="GRAPH_EXTERNAL_TOPIC_HANDLE_INVALID"):
        replacement.add_flow(output, input_)


# REQ: PY-API-009
def test_camera_imu_topic_plan_matches_independent_cpp_authoring():
    # Key setup: Compile a nested camera/IMU fusion graph with imported and exported topics.
    #
    #   camera ---------\
    #                    fusion --> holomq://control
    #   holomq://imu ---/
    plan = camera_imu_plan()

    # Key assertion: Python and independently authored C++ produce the same accepted plan.
    assert plan.ok
    assert plan.json() == cpp_plan("camera")


# REQ: PY-API-009
def test_generated_endpoint_warning_matches_independent_cpp_authoring():
    # Key setup: Import a topic without pinning its generated endpoint name.
    #
    #   holomq://generated --> generated-endpoint
    graph = hs.Graph("generated-endpoint")
    graph.import_topic("holomq://generated", hs.payload.uint32)

    plan = hs.compile(graph)

    # Key assertion: Python and C++ preserve the same nonfatal generated-name warning.
    assert plan.ok
    assert not plan.errors
    assert [warning.token for warning in plan.warnings] == [
        "ENDPOINT_NAME_GENERATED_PIN_RECOMMENDED"
    ]
    assert plan.warnings[0].severity is hs.diagnostics.DiagnosticSeverity.WARNING
    assert plan.json() == cpp_plan("warning")


# REQ: PY-API-009
def test_saved_catalog_interoperates_with_cpp_peer(tmp_path):
    # Key setup: Run a Python topic graph against a C++ peer using the saved endpoint catalog.
    #
    #   C++ publisher --> doubler --> C++ subscriber
    process_id = os.getpid()
    input_name = f"python-topic-input-{process_id}"
    output_name = f"python-topic-output-{process_id}"
    graph = hs.Graph("python-cpp-topic-peer", default_clock=hs.RealtimeClock)
    imported = graph.import_topic(
        f"holomq://{input_name}", hs.payload.uint32, endpoint_name=input_name
    )
    exported = graph.export_topic(
        f"holomq://{output_name}", hs.payload.uint32, endpoint_name=output_name
    )
    doubler = graph.op("holoscan.test.DoubleIt", "doubler")
    graph.add_flow(imported.output, doubler["in"], queue_depth=2)
    graph.add_flow(doubler.out, exported.input, queue_depth=2)
    plan = hs.compile(graph)
    catalog_path = tmp_path / "catalog.json"
    plan.endpoint_catalog.save_json(catalog_path)

    # Key assertion: The peer round trip doubles the value and the Python session stops cleanly.
    assert plan.ok
    with hs.run_async(plan) as session:
        peer = subprocess.run(
            [
                os.environ["HOLOSCAN_TOPIC_PEER_HELPER"],
                catalog_path,
                input_name,
                output_name,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )

    assert peer.returncode == 0, peer.stderr
    assert "value=42" in peer.stdout
    assert session.termination is hs.RunTermination.STOPPED
