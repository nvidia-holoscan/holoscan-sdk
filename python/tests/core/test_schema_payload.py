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
import vector_payload

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


class ControlBorrowSink(hs.Operator):
    """Inspect a scalar selection and optionally materialize its generated value."""

    inp = hs.Input("in", control_payload.ControlPayload)
    request_data = True
    projection_calls = 0
    received = None
    retained = None

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        before = type(self).projection_calls
        selected = self.inp.selected_sample()
        assert type(self).projection_calls == before
        assert isinstance(selected.metadata, hs.SampleMetadata)
        assert type(self).projection_calls == before
        assert selected.is_stale is False
        assert type(self).projection_calls == before
        if type(self).request_data:
            type(self).received = selected.data
            assert type(self).projection_calls == before + 1
        type(self).retained = selected


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


# REQ: PY-OP-012
# REQ: SCHEMA-PY-001
@pytest.mark.parametrize("request_data", [False, True], ids=["metadata-only", "materialize"])
def test_generated_schema_selected_sample_materializes_lazily_and_expires(
    monkeypatch, request_data
):
    # Key setup: Count generated-schema projections in metadata-only and materializing callbacks.
    original = control_payload.ControlPayload._from_holoscan_schema

    def counted_projection(fields):
        ControlBorrowSink.projection_calls += 1
        return original(fields)

    monkeypatch.setattr(
        control_payload.ControlPayload, "_from_holoscan_schema", staticmethod(counted_projection)
    )
    ControlBorrowSink.request_data = request_data
    ControlBorrowSink.projection_calls = 0
    ControlBorrowSink.received = None
    ControlBorrowSink.retained = None
    graph = hs.Graph("python-schema-selected-sample", default_clock=hs.RealtimeClock)
    source = graph.op(ControlSource, "source")
    sink = graph.op(ControlBorrowSink, "sink")
    graph.add_flow(source.out, sink["in"])

    session = hs.run(hs.compile(graph))

    # Key assertion: Only a data access projects the schema; a metadata-only callback completes
    # without conversion, and all facade properties reject access after callback retirement.
    assert session.diagnostics == []
    assert ControlBorrowSink.projection_calls == int(request_data)
    assert ControlBorrowSink.retained is not None
    assert ControlBorrowSink.received == (
        control_payload.ControlPayload(sequence=7, gain=2.5, labels=["left", "right"])
        if request_data
        else None
    )
    for property_name in ("data", "metadata", "is_stale"):
        with pytest.raises(hs.errors.NotReadyError, match="PYTHON_SELECTED_SAMPLE_EXPIRED"):
            getattr(ControlBorrowSink.retained, property_name)
    assert ControlBorrowSink.projection_calls == int(request_data)


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


# REQ: SCHEMA-PY-002
@pytest.mark.skipif(sys.platform == "win32", reason="the stack limit is set with a POSIX shell")
def test_generated_schema_recursive_value_is_rejected_on_an_8_mib_stack():
    # Key setup: Rerun the invalid-dict test in a child whose soft stack limit is at most a common
    # 8 MiB default. Runtime threads take their stack size from it, unlike the larger limit a
    # development container may set for this process.
    #
    #   sh (ulimit -S -s 8192) -> pytest ...::test_generated_schema_dict_bridge_rejects_invalid
    #
    selected = f"{__file__}::test_generated_schema_dict_bridge_rejects_invalid"
    lower_stack = (
        'limit=$(ulimit -S -s); if [ "$limit" = unlimited ] || [ "$limit" -gt 8192 ]; then '
        "ulimit -S -s 8192 || exit 97; fi; "
    )
    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            lower_stack + 'exec "$0" -m pytest -q -p no:cacheprovider "$1"',
            sys.executable,
            selected,
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    # Key assertion: The self-referential value raises RecursionError in the callback rather than
    # exhausting the Runtime thread's stack.
    assert result.returncode == 0, result.stdout + result.stderr


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
# REQ: GRAPH-045
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


# REQ: PY-OP-015
# REQ: PY-API-021
@pytest.mark.parametrize("override", [False, True])
def test_selected_schema_forward_skips_conversion_and_preserves_metadata(monkeypatch, override):
    # Graph: source --ControlPayload--> relay --ControlPayload--> sink.
    # Fail any Python schema conversion during relay publication; source and sink remain owning.
    received = []
    forwarded = []
    retained = []
    errors = {}
    source_ports = []

    class Source(hs.Operator):
        """Publish two schema values with semantic metadata for forwarding qualification."""

        out = hs.Output("out", control_payload.ControlPayload)

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(2)

        def compute(self, ctx):
            source_ports[:] = [self.out]
            self.out.emit(
                control_payload.ControlPayload(sequence=7, gain=2.5, labels=["left", "right"]),
                options=hs.EmitOptions(
                    capture_time=ctx.activation_time,
                    frame_id=41,
                    trace_id=73,
                    flags=hs.SampleFlags.DEGRADED,
                ),
            )

    class Relay(hs.Operator):
        """Publish a live native borrow and reject attempts outside its authority."""

        inp = hs.Input("in", control_payload.ControlPayload)
        out = hs.Output("out", control_payload.ControlPayload)
        wrong = hs.Output("wrong", hs.schema.Time)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            selected = self.inp.selected_sample()
            received.append(selected.metadata)
            if retained:
                with pytest.raises(hs.errors.NotReadyError) as expired:
                    self.out.forward(retained[0])
                errors["previous_callback"] = expired.value.token
            else:
                retained.extend([selected, self.out])

            with pytest.raises(hs.errors.IncompatibleError) as wrong_type:
                self.wrong.forward(selected)
            errors["wrong_type"] = wrong_type.value.token
            with pytest.raises(hs.errors.InvalidArgumentError) as wrong_operator:
                source_ports[0].forward(selected)
            errors["wrong_operator"] = wrong_operator.value.token

            def forward_from_foreign_thread():
                try:
                    self.out.forward(selected)
                except hs.errors.NotReadyError as error:
                    errors["foreign_thread"] = error.token

            worker = threading.Thread(target=forward_from_foreign_thread)
            worker.start()
            worker.join(timeout=1.0)
            assert not worker.is_alive()

            def reject_conversion(*_args, **_kwargs):
                raise AssertionError("borrowed forwarding must not convert a Python schema value")

            with monkeypatch.context() as patch:
                patch.setattr(
                    control_payload.ControlPayload, "_from_holoscan_schema", reject_conversion
                )
                patch.setattr(
                    control_payload.ControlPayload, "_holoscan_schema_values", reject_conversion
                )
                options = hs.EmitOptions(frame_id=42) if override else None
                self.out.forward(selected, options=options)

    class Sink(hs.Operator):
        """Retain forwarded values and metadata independently of the relay's borrow."""

        inp = hs.Input("in", control_payload.ControlPayload)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            forwarded.append(self.inp.receive())
            if len(forwarded) == 2:
                ctx.request_completion()

    graph = hs.Graph("python-borrowed-schema-forward", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    relay = graph.op(Relay, "relay")
    sink = graph.op(Sink, "sink")
    graph.add_flow(source.out, relay["in"], queue_depth=2)
    graph.add_flow(relay.out, sink["in"], queue_depth=2)
    session = hs.run(hs.compile(graph))

    # Borrowed forwarding preserves payload and semantic metadata and restamps Runtime facts.
    assert session.diagnostics == []
    assert len(forwarded) == 2
    for original, sample in zip(received, forwarded, strict=True):
        assert sample.data == control_payload.ControlPayload(
            sequence=7, gain=2.5, labels=["left", "right"]
        )
        assert sample.metadata.capture_time == original.capture_time
        assert sample.metadata.frame_id == (42 if override else 41)
        assert sample.metadata.trace_id == 73
        assert sample.metadata.flags == hs.SampleFlags.DEGRADED
        assert sample.metadata.sequence > 0
        assert sample.metadata.publish_time.valid
    assert errors == {
        "previous_callback": "PYTHON_SELECTED_SAMPLE_EXPIRED",
        "foreign_thread": "PYTHON_SELECTED_SAMPLE_EXPIRED",
        "wrong_type": "GRAPH_PORT_TYPE_MISMATCH",
        "wrong_operator": "PYTHON_FORWARD_SELECTED_OPERATOR_MISMATCH",
    }
    with pytest.raises(hs.errors.NotReadyError, match="PYTHON_SELECTED_SAMPLE_EXPIRED"):
        retained[1].forward(retained[0])


# REQ: PY-OP-015
def test_selected_schema_forward_rejects_tensor_descriptors():
    # Graph: two-Tensor source --> descriptor probe. Descriptor publication requires output loans.
    rejected = []

    class Source(StereoSource):
        """Publish one descriptor transaction using the existing two-field loan contract."""

        def compute(self, context):
            request = hs.tensor.LoanRequest(shape=(2, 2), dtype=hs.tensor.DLDataType(1, 8, 1))
            with self.out.allocate_tensors(
                stereo_payload.StereoPayload(), {"left": request, "right": request}
            ) as loan:
                for tensor in loan.tensors.values():
                    storage = (ctypes.c_uint8 * tensor.nbytes).from_address(tensor.data)
                    storage[:] = [1] * tensor.nbytes

    class Probe(StereoSource):
        """Reject borrowed descriptor publication while retaining ordinary input access."""

        inp = hs.Input("in", stereo_payload.StereoPayload)

        def setup(self, spec):
            super().setup(spec)
            spec.input(self.inp)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            with pytest.raises(hs.errors.NotSupportedError) as error:
                self.out.forward(self.inp.selected_sample())
            rejected.append(error.value.code)
            ctx.request_completion()

    graph = hs.Graph("python-borrowed-descriptor-rejection", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    probe = graph.op(Probe, "probe")
    graph.add_flow(source.out, probe["in"], queue_depth=1)
    session = hs.run(hs.compile(graph))
    assert session.diagnostics == []
    assert rejected == [hs.errors.ErrorCode.NOT_SUPPORTED]


# REQ: SCHEMA-PY-007
@pytest.mark.parametrize("read", ["receive", "receive_data", "selected_sample", "selected_batch"])
@pytest.mark.parametrize("byte_input", [bytes, bytearray, list, tuple])
def test_generated_schema_vectors_round_trip(read, byte_input):
    # Graph: Python source --> Python relay --> sink. Retain values after session teardown.
    expected = vector_payload.VectorPayload(
        blob=bytes(range(256)) * 1628,
        signed_bytes=[-128, 0, 127],
        shorts=[-32768, 32767],
        ushorts=[0, 65535],
        ints=[-(2**31), 2**31 - 1],
        uints=[0, 2**32 - 1],
        longs=[-(2**63), 2**63 - 1],
        ulongs=[0, 2**64 - 1],
        floats=[-0.5, 0.0, 1.25],
        doubles=[-1e300, 1e300],
        bools=[True, False],
        labels=["left", "right"],
    )
    retained = []

    class Source(hs.Operator):
        """Exercise both generated value and dictionary authoring of all scalar vector types."""

        out = hs.Output("out", vector_payload.VectorPayload)

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

        def compute(self, context):
            fields = expected._holoscan_schema_values()
            fields["blob"] = byte_input(expected.blob)
            self.out.emit(fields)

    class Relay(hs.Operator):
        """Read through each owning projection and emit the materialized value again."""

        inp = hs.Input("in", vector_payload.VectorPayload)
        out = hs.Output("out", vector_payload.VectorPayload)

        def contract(self):
            if read == "selected_batch":
                return (
                    hs.Contract()
                    .trigger(hs.OnClock(period=hs.time.ms(5)))
                    .input(self.inp, hs.DrainAll())
                )
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            if read == "selected_batch":
                sample = next(iter(self.inp.selected_batch()), None)
                if sample is None:
                    return
                value = sample.data
            elif read == "receive_data":
                value = self.inp.receive_data()
            else:
                value = getattr(self.inp, read)().data
            assert isinstance(value.blob, bytes)
            assert isinstance(value.signed_bytes, list)
            retained.append(value)
            self.out.emit(value)

    class Sink(hs.Operator):
        """Complete after validating the relay's reconstructed native payload."""

        inp = hs.Input("in", vector_payload.VectorPayload)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            retained.append(self.inp.receive_data())
            context.request_completion()

    graph = hs.Graph("python-schema-vectors", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    relay = graph.op(Relay, "relay")
    sink = graph.op(Sink, "sink")
    graph.add_flow(source.out, relay["in"])
    graph.add_flow(relay.out, sink["in"])
    session = hs.run(hs.compile(graph))
    assert session.diagnostics == []
    assert retained == [expected, expected]
    assert vector_payload.VectorPayload().blob == b""


@pytest.mark.parametrize(
    ("field", "value", "token"),
    [
        ("blob", [-1], "PYTHON_SCHEMA_FIELD_VALUE_OUT_OF_RANGE"),
        ("blob", [256], "PYTHON_SCHEMA_FIELD_VALUE_OUT_OF_RANGE"),
        ("blob", [True], "PYTHON_SCHEMA_FIELD_TYPE_MISMATCH"),
        ("floats", [float("inf")], "PYTHON_SCHEMA_FIELD_VALUE_OUT_OF_RANGE"),
        ("floats", [1e300], "PYTHON_SCHEMA_FIELD_VALUE_OUT_OF_RANGE"),
        ("bools", b"\x01", "PYTHON_SCHEMA_FIELD_TYPE_MISMATCH"),
        ("shorts", b"\x01", "PYTHON_SCHEMA_FIELD_TYPE_MISMATCH"),
    ],
)
# REQ: SCHEMA-PY-007
def test_generated_schema_vectors_reject_invalid_elements(field, value, token):
    failures = []

    class Source(hs.Operator):
        """Reject an invalid vector without publishing it, then publish an empty byte vector."""

        out = hs.Output("out", vector_payload.VectorPayload)

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

        def compute(self, context):
            with pytest.raises(hs.errors.InvalidArgumentError) as error:
                self.out.emit({field: value})
            failures.append(error.value.token)
            self.out.emit({"blob": b""})

    graph = hs.Graph("python-schema-invalid-vectors", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")

    class Sink(hs.Operator):
        """Keep the valid follow-up publication connected to an input."""

        inp = hs.Input("in", vector_payload.VectorPayload)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            assert self.inp.receive_data().blob == b""
            context.request_completion()

    sink = graph.op(Sink, "sink")
    graph.add_flow(source.out, sink["in"])
    session = hs.run(hs.compile(graph))
    assert session.diagnostics == []
    assert failures == [token]
