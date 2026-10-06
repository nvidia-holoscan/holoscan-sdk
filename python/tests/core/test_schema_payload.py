# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated schema values carried by Python-hosted operators."""

import ctypes
import os
import selectors
import subprocess
import sys
import threading
import time
import uuid

import control_payload
import pytest
import stereo_payload

import holoscan as hs


def diagnostic_rows(diagnostics):
    """Return stable tokens and optional owning error details for assertion output."""
    return [
        (
            diagnostic.token,
            diagnostic.detail.error.message if diagnostic.detail is not None else "",
        )
        for diagnostic in diagnostics
    ]


def read_process_line(stream, timeout):
    """Read one subprocess line without allowing a missing peer message to hang the test."""
    deadline = time.monotonic() + timeout
    received = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(stream.fileno(), selectors.EVENT_READ)
        while b"\n" not in received:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise subprocess.TimeoutExpired("schema payload peer stdout", timeout)
            chunk = os.read(stream.fileno(), 4096)
            if not chunk:
                break
            received.extend(chunk)
    return received.decode()


def stereo_tensor_fields():
    """Build the exact host uint8 contracts shared with the Python peer test."""
    contract = hs.tensor.TensorBoundaryContract(
        representation=hs.tensor.TensorRepresentation(
            memory_kind=hs.tensor.MemoryKind.HOST,
            dtype=hs.tensor.DLDataType(1, 8, 1),
            rank=2,
            strides=hs.tensor.TensorStridePolicy.CONTIGUOUS_ONLY,
        ),
        bounds=hs.tensor.TensorBounds(max_shape=(2, 2)),
        transfer=hs.tensor.TensorTransferPolicy(
            foreign_emit_policy=hs.tensor.InterfaceTensorEmitPolicy.COPY_IF_NEEDED
        ),
    )
    return {"left": contract, "right": contract}


class ControlSource(hs.Operator):
    """Emit one generated control value; used by test_generated_schema_round_trip."""

    out = hs.Output("out", control_payload.ControlPayload)

    def setup(self, spec):
        spec.output(self.out)

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        self.out.emit(
            control_payload.ControlPayload(sequence=7, gain=2.5, labels=["left", "right"])
        )


class ControlSink(hs.Operator):
    """Capture one generated control value; used by test_generated_schema_round_trip."""

    inp = hs.Input("in", control_payload.ControlPayload)
    received = None

    def setup(self, spec):
        spec.input(self.inp)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        type(self).received = self.inp.receive_data()


class ControlDictSource(hs.Operator):
    """Emit a schema-compatible dict; used by test_generated_schema_dict_bridge."""

    out = hs.Output("out", control_payload.ControlPayload)

    def setup(self, spec):
        spec.output(self.out)

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        self.out.emit({"sequence": 11, "gain": 0.5, "labels": ["dict"]})


class ValidatingControlDictSource(hs.Operator):
    """Exercise rejected dict values; used by test_generated_schema_dict_bridge_rejects_invalid."""

    out = hs.Output("out", control_payload.ControlPayload)
    errors = {}

    def setup(self, spec):
        spec.output(self.out)

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        errors = {}
        invalid = {
            "unknown": {"unexpected": 1},
            "range": {"sequence": 2**64},
            "float_range": {"gain": 10**10000},
            "bool": {"sequence": True},
        }
        for name, value in invalid.items():
            try:
                self.out.emit(value)
            except hs.errors.InvalidArgumentError as error:
                errors[name] = error.token

        recursive = []
        recursive.append(recursive)
        try:
            self.out.emit({"sequence": 1, "gain": 1.0, "labels": recursive})
        except RecursionError:
            errors["recursive"] = "RecursionError"

        type(self).errors = errors
        self.out.emit({"sequence": 12, "gain": 0.75, "labels": ["valid"]})


class ControlRelay(hs.Operator):
    """Relay a generated schema value; used by test_generated_schema_cross_process_cpp_peer."""

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


class StereoSource(hs.Operator):
    """Publish a two-field host Tensor loan; used by test_descriptor_schema_tensor_loan."""

    out = hs.Output("out", stereo_payload.StereoPayload)
    abandoned_tensor = None
    discarded_mapping = None
    discarded_tensor = None
    published_tensor_addresses = {}
    retained_tensors = ()
    double_enter_token = None
    consumed_token = None
    rejection_tokens = {}

    def setup(self, spec):
        output = spec.output(self.out)
        tensor_spec = hs.tensor.OutputSpec(
            storage="runtime_pool",
            memory_kind=hs.tensor.MemoryKind.HOST,
            dtype=hs.tensor.DLDataType(1, 8, 1),
            rank=2,
            transfer=hs.tensor.TensorTransferPolicy(
                foreign_emit_policy=hs.tensor.InterfaceTensorEmitPolicy.COPY_IF_NEEDED
            ),
        )
        output.produces_tensor(tensor_spec, hs.tensor.byte_span(4), field="left")
        output.produces_tensor(tensor_spec, hs.tensor.byte_span(4), field="right")

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        request = hs.tensor.LoanRequest(shape=(2, 2), dtype=hs.tensor.DLDataType(1, 8, 1))
        descriptor = stereo_payload.StereoPayload(
            header=stereo_payload.FrameHeader(sequence=9, source="python"), exposure_us=2500
        )
        rejection_tokens = {}
        try:
            self.out.allocate_tensors(descriptor, {"left": request})
        except hs.errors.InvalidArgumentError as error:
            rejection_tokens["count"] = error.token
        try:
            self.out.allocate_tensors(descriptor, {"left": request, "missing": request})
        except hs.errors.NotFoundError as error:
            rejection_tokens["unknown"] = error.token
        try:
            self.out.allocate_tensors(descriptor, {"left": request, "right": object()})
        except TypeError:
            rejection_tokens["type"] = "TypeError"
        try:
            self.out.allocate_tensors({"unexpected": 1}, {"left": request, "right": request})
        except hs.errors.InvalidArgumentError as error:
            rejection_tokens["descriptor"] = error.token
        type(self).rejection_tokens = rejection_tokens

        discarded = self.out.allocate_tensors(descriptor, {"left": request, "right": request})
        type(self).discarded_mapping = discarded.tensors
        type(self).discarded_tensor = type(self).discarded_mapping["left"]
        del discarded

        try:
            with self.out.allocate_tensors(
                descriptor, {"left": request, "right": request}
            ) as abandoned:
                type(self).abandoned_tensor = abandoned.tensors["left"]
                storage = (ctypes.c_uint8 * 4).from_address(abandoned.tensors["left"].data)
                storage[:] = (99, 99, 99, 99)
                raise RuntimeError("abandon schema Tensor loan")
        except RuntimeError as error:
            if str(error) != "abandon schema Tensor loan":
                raise

        with self.out.allocate_tensors(descriptor, {"left": request, "right": request}) as loan:
            tensors = loan.tensors
            type(self).retained_tensors = (tensors["left"], tensors["right"])
            type(self).published_tensor_addresses = {
                "left": tensors["left"].data,
                "right": tensors["right"].data,
            }
            try:
                loan.__enter__()
            except hs.errors.InvalidArgumentError as error:
                type(self).double_enter_token = error.token
            left = (ctypes.c_uint8 * 4).from_address(tensors["left"].data)
            right = (ctypes.c_uint8 * 4).from_address(tensors["right"].data)
            left[:] = (1, 2, 3, 4)
            right[:] = (5, 6, 7, 8)
        try:
            _ = loan.tensors
        except hs.errors.ConsumedLoanError as error:
            type(self).consumed_token = error.token


class StereoSink(hs.Operator):
    """Capture descriptor metadata and Tensors; used by test_descriptor_schema_tensor_loan."""

    inp = hs.Input("in", stereo_payload.StereoPayload)
    received = None
    received_event = threading.Event()

    def setup(self, spec):
        spec.input(self.inp)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        type(self).received = self.inp.receive_data()
        type(self).received_event.set()


class ProjectedStereoTensorSink(hs.Operator):
    """Capture one projected Tensor field from StereoSource."""

    inp = hs.Input("in", hs.Tensor)
    received = None
    received_event = threading.Event()

    def setup(self, spec):
        spec.input(self.inp).accepts_projected_tensor()

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        type(self).received = self.inp.receive_data()
        type(self).received_event.set()


class StereoTopicSource(StereoSource):
    """Publish one valid stereo frame for test_descriptor_schema_cross_process_cpp_peer."""

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1000)))

    def compute(self, context):
        request = hs.tensor.LoanRequest(shape=(2, 2), dtype=hs.tensor.DLDataType(1, 8, 1))
        descriptor = stereo_payload.StereoPayload(
            header=stereo_payload.FrameHeader(sequence=9, source="python"), exposure_us=2500
        )
        try:
            with self.out.allocate_tensors(descriptor, {"left": request, "right": request}) as loan:
                tensors = loan.tensors
                left = (ctypes.c_uint8 * 4).from_address(tensors["left"].data)
                right = (ctypes.c_uint8 * 4).from_address(tensors["right"].data)
                left[:] = (1, 2, 3, 4)
                right[:] = (5, 6, 7, 8)
        except hs.errors.NotReadyError:
            # The required peer may still be attaching. A fresh activation receives a fresh loan.
            return


# REQ: SCHEMA-PY-001; SCHEMA-PY-002
def test_generated_schema_round_trip():
    # Key setup: Compose a generated ordinary-schema pipeline.
    #
    #   ControlSource ──ControlPayload──> ControlSink
    #
    graph = hs.Graph("python-schema-round-trip", default_clock=hs.RealtimeClock)
    ControlSink.received = None
    source = graph.op(ControlSource, "source")
    sink = graph.op(ControlSink, "sink")
    graph.add_flow(source.out, sink["in"])

    plan = hs.compile(graph)
    assert plan.ok, [error.token for error in plan.errors]
    hs.run(plan)

    # Key assertion: The received value retains dataclass identity and every supported field kind.
    assert isinstance(ControlSink.received, control_payload.ControlPayload)
    assert ControlSink.received.sequence == 7
    assert ControlSink.received.gain == pytest.approx(2.5)
    assert ControlSink.received.labels == ["left", "right"]


# REQ: SCHEMA-PY-002
def test_generated_schema_dict_bridge():
    # Key setup: Emit a small control schema from its interim dict representation.
    #
    #   ControlDictSource ──dict/ControlPayload──> ControlSink
    #
    graph = hs.Graph("python-schema-dict-bridge", default_clock=hs.RealtimeClock)
    ControlSink.received = None
    source = graph.op(ControlDictSource, "source")
    sink = graph.op(ControlSink, "sink")
    graph.add_flow(source.out, sink["in"])

    plan = hs.compile(graph)
    assert plan.ok, [error.token for error in plan.errors]
    hs.run(plan)

    # Key assertion: The native codec materializes the dict as the generated dataclass.
    assert ControlSink.received == control_payload.ControlPayload(
        sequence=11, gain=0.5, labels=["dict"]
    )


# REQ: SCHEMA-PY-002
def test_generated_schema_dict_bridge_rejects_invalid():
    # Key setup: Try unknown, out-of-range, mistyped, and recursive dict values before a valid emit.
    #
    #   ValidatingControlDictSource ──ControlPayload──> ControlSink
    #
    graph = hs.Graph("python-schema-invalid-dict", default_clock=hs.RealtimeClock)
    ValidatingControlDictSource.errors = {}
    ControlSink.received = None
    source = graph.op(ValidatingControlDictSource, "source")
    sink = graph.op(ControlSink, "sink")
    graph.add_flow(source.out, sink["in"])

    plan = hs.compile(graph)
    assert plan.ok, [error.token for error in plan.errors]
    hs.run(plan)

    # Key assertion: Invalid values fail deterministically and do not consume publication authority.
    assert ValidatingControlDictSource.errors == {
        "unknown": "PYTHON_SCHEMA_FIELD_UNKNOWN",
        "range": "PYTHON_SCHEMA_FIELD_VALUE_OUT_OF_RANGE",
        "float_range": "PYTHON_SCHEMA_FIELD_VALUE_OUT_OF_RANGE",
        "bool": "PYTHON_SCHEMA_FIELD_TYPE_MISMATCH",
        "recursive": "RecursionError",
    }
    assert ControlSink.received == control_payload.ControlPayload(
        sequence=12, gain=0.75, labels=["valid"]
    )


# REQ: SCHEMA-PY-003
# REQ: SCHEMA-PY-005
def test_descriptor_schema_tensor_loan():
    # Key setup: Compose a generated descriptor-schema pipeline with one atomic two-Tensor loan.
    #
    #   StereoSource ──StereoPayload(left, right)──> StereoSink
    #
    graph = hs.Graph("python-schema-tensor-loan", default_clock=hs.RealtimeClock)
    StereoSink.received = None
    StereoSink.received_event.clear()
    StereoSource.abandoned_tensor = None
    StereoSource.discarded_mapping = None
    StereoSource.discarded_tensor = None
    StereoSource.retained_tensors = ()
    StereoSource.double_enter_token = None
    StereoSource.consumed_token = None
    StereoSource.rejection_tokens = {}
    source = graph.op(StereoSource, "source")
    sink = graph.op(StereoSink, "sink")
    graph.add_flow(source.out, sink["in"])

    plan = hs.compile(graph)
    assert plan.ok, [error.token for error in plan.errors]
    with hs.run_async(plan) as session:
        assert StereoSink.received_event.wait(timeout=5), diagnostic_rows(session.diagnostics)
        # Key assertion: Metadata and the read-only named Tensor view survive native carriage.
        frame = StereoSink.received
        assert isinstance(frame, stereo_payload.StereoPayload)
        assert frame.header == stereo_payload.FrameHeader(sequence=9, source="python")
        assert frame.exposure_us == 2500
        assert set(frame.tensors) == {"left", "right"}
        assert bytes((ctypes.c_uint8 * 4).from_address(frame.left.data)) == bytes((1, 2, 3, 4))
        assert bytes((ctypes.c_uint8 * 4).from_address(frame.right.data)) == bytes((5, 6, 7, 8))
        migrated = stereo_payload.StereoPayload(left=frame.left, right=frame.right)
        assert migrated.left is frame.left
        assert migrated.right is frame.right
        with pytest.raises(TypeError):
            frame.tensors["extra"] = frame.left
        del migrated
        del frame
        StereoSink.received = None
        session.request_stop()

    assert StereoSource.rejection_tokens == {
        "count": "TENSOR_SCHEMA_BUNDLE_FIELD_COUNT_MISMATCH",
        "unknown": "AUTHORING_DESCRIPTOR_TENSOR_FIELD_UNKNOWN",
        "type": "TypeError",
        "descriptor": "PYTHON_SCHEMA_FIELD_UNKNOWN",
    }
    assert StereoSource.double_enter_token == "PYTHON_TENSOR_SCHEMA_LOAN_ALREADY_ENTERED"
    assert StereoSource.consumed_token == "PYTHON_TENSOR_LOAN_CONSUMED"
    assert StereoSource.abandoned_tensor.data == 0
    assert StereoSource.discarded_tensor.data == 0
    assert all(tensor.data == 0 for tensor in StereoSource.discarded_mapping.values())
    assert [tensor.data for tensor in StereoSource.retained_tensors] == [0, 0]


# REQ: SCHEMA-PY-003
# REQ: GRAPH-043
def test_descriptor_schema_projects_nonzero_tensor_field_without_copy():
    # Key setup: Select the second field of a two-Tensor schema and record both loan addresses
    # before publication transfers ownership to Runtime.
    #
    #   StereoSource.right ──Tensor projection──> ProjectedStereoTensorSink
    #
    graph = hs.Graph("python-schema-right-tensor-projection", default_clock=hs.RealtimeClock)
    StereoSource.published_tensor_addresses = {}
    StereoSource.retained_tensors = ()
    ProjectedStereoTensorSink.received = None
    ProjectedStereoTensorSink.received_event.clear()
    source = graph.op(StereoSource, "source")
    sink = graph.op(ProjectedStereoTensorSink, "sink")
    graph.add_flow(source.out.project("right"), sink["in"])

    plan = hs.compile(graph)
    assert plan.ok, [error.token for error in plan.errors]
    with hs.run_async(plan) as session:
        assert ProjectedStereoTensorSink.received_event.wait(timeout=5), diagnostic_rows(
            session.diagnostics
        )

        # Key assertion: The named nonzero field, not the first field, arrives with its original
        # runtime-loan backing and distinct contents.
        tensor = ProjectedStereoTensorSink.received
        assert tensor.data == StereoSource.published_tensor_addresses["right"]
        assert tensor.data != StereoSource.published_tensor_addresses["left"]
        assert bytes((ctypes.c_uint8 * 4).from_address(tensor.data)) == bytes((5, 6, 7, 8))
        del tensor
        ProjectedStereoTensorSink.received = None
        session.request_stop()

    assert session.diagnostics == []


# REQ: SCHEMA-PY-003
def test_schema_payload_type_preserves_transport_profile():
    # Key setup: Resolve ordinary and descriptor payload rows installed by generated codec modules.
    control = hs.payload.resolve(control_payload.ControlPayload._registry_name)
    stereo = hs.payload.resolve(stereo_payload.StereoPayload._registry_name)

    # Key assertion: Only the descriptor schema carries its exact nonzero profile tag.
    assert control.transport_profile_tag == 0
    assert stereo.transport_profile_tag != 0


def test_schema_payload_registration_is_independent_of_optional_module_entry_points():
    # Key setup: Clear the optional-module provider index in a fresh process before importing one
    # generated payload module.
    script = "\n".join(
        (
            "import holoscan as hs",
            "from holoscan import _module_discovery",
            "_module_discovery.refresh_for_tests(())",
            "import control_payload",
            "payload = hs.payload.resolve(control_payload.ControlPayload._registry_name)",
            "assert payload.name == control_payload.ControlPayload._registry_name",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], check=False, capture_output=True, text=True
    )

    # Key assertion: Generated schema registration resolves without a holoscan.modules provider.
    assert completed.returncode == 0, completed.stderr


# REQ: SCHEMA-PY-001
def test_generated_schema_cross_process_cpp_peer(tmp_path):
    # Key setup: Relay one generated dataclass between external C++ publisher and subscriber.
    #
    #   C++ publisher ──> ControlRelay ──> C++ subscriber
    #
    instance = uuid.uuid4().hex
    input_name = f"python-schema-input-{instance}"
    output_name = f"python-schema-output-{instance}"
    payload = hs.payload.resolve(control_payload.ControlPayload._registry_name)
    attach = hs.topics.TopicAttachPolicy(
        peer=hs.topics.PeerRequirement.REQUIRED,
        max_wait=hs.time.ms(10_000),
    )
    graph = hs.Graph("python-schema-cross-process", default_clock=hs.RealtimeClock)
    imported = graph.import_topic(
        f"holomq://{input_name}", payload, endpoint_name=input_name, attach=attach
    )
    exported = graph.export_topic(
        f"holomq://{output_name}", payload, endpoint_name=output_name, attach=attach
    )
    relay = graph.op(ControlRelay, "relay")
    graph.add_flow(imported.output, relay["in"])
    graph.add_flow(relay.out, exported.input)
    plan = hs.compile(graph)
    assert plan.ok, plan.json()
    catalog_path = tmp_path / "schema-catalog.json"
    plan.endpoint_catalog.save_json(catalog_path)

    with hs.run_async(plan) as session:
        peer = subprocess.Popen(
            [
                os.environ["HOLOSCAN_SCHEMA_PAYLOAD_PEER"],
                catalog_path,
                input_name,
                output_name,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert peer.stdout is not None
            first_line = read_process_line(peer.stdout, timeout=15)
            session.request_stop()
            remaining_stdout, peer_stderr = peer.communicate(input="\n", timeout=15)
            peer_stdout = first_line + remaining_stdout
        finally:
            session.request_stop()
            if peer.poll() is None:
                peer.kill()
                peer.communicate(timeout=5)

    # Key assertion: The exact generated native type and values cross both process boundaries.
    assert peer.returncode == 0, peer_stderr
    assert "sequence=42" in peer_stdout
    assert session.termination is hs.RunTermination.STOPPED, diagnostic_rows(session.diagnostics)


# REQ: SCHEMA-PY-003
def test_descriptor_schema_cross_process_cpp_peer(tmp_path):
    # Key setup: Export a generated two-Tensor dataclass to an independent C++ subscriber.
    #
    #   StereoSource ──StereoPayload──> holomq://stereo ──> C++ ChannelRuntime subscriber
    #
    instance = uuid.uuid4().hex
    endpoint_name = f"python-stereo-{instance}"
    attach = hs.topics.TopicAttachPolicy(
        peer=hs.topics.PeerRequirement.REQUIRED,
        max_wait=hs.time.ms(10_000),
    )
    graph = hs.Graph("python-schema-stereo-publisher", default_clock=hs.RealtimeClock)
    exported = graph.export_topic(
        f"holomq://{endpoint_name}",
        hs.payload.resolve(stereo_payload.StereoPayload._registry_name),
        endpoint_name=endpoint_name,
        attach=attach,
        tensor_fields=stereo_tensor_fields(),
    )
    source = graph.op(StereoTopicSource, "source")
    graph.add_flow(source.out, exported.input)
    plan = hs.compile(graph)
    assert plan.ok, plan.json()
    catalog_path = tmp_path / "stereo-catalog.json"
    plan.endpoint_catalog.save_json(catalog_path)

    with hs.run_async(plan) as session:
        peer = subprocess.run(
            [
                os.environ["HOLOSCAN_SCHEMA_PAYLOAD_PEER"],
                "--stereo",
                catalog_path,
                endpoint_name,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )

    # Key assertion: Generated metadata and both named Tensor fields cross the process boundary.
    assert peer.returncode == 0, peer.stderr
    assert "generated stereo C++ peer complete" in peer.stdout
    assert session.termination is hs.RunTermination.STOPPED, diagnostic_rows(session.diagnostics)
