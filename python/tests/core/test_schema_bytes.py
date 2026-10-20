# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owning byte-vector values across native, generated Python and Channel boundaries."""

import ctypes
import gc
import uuid

import byte_channel_payload
import byte_payload as schema
import byte_schema_support
import byte_tensor_payload
import pytest

import holoscan as hs


def run_graph(*, size=256, mode="receive_data", authored=None, invalid=None):
    """Run a bounded native source or one Python emission and retain the received value."""
    received = []
    selected = []
    rejected = []
    byte_schema_support.configure(size)

    class Source(hs.Operator):
        out = hs.Output("out", schema.BytePayload)

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

        def compute(self, context):
            if invalid is not None:
                try:
                    self.out.emit(invalid)
                except (hs.HoloscanError, TypeError) as error:
                    rejected.append(error)
            self.out.emit(authored)

    class Sink(hs.Operator):
        inp = hs.Input("in", schema.BytePayload)

        def contract(self):
            if mode == "batch":
                return (
                    hs.Contract()
                    .trigger(hs.OnClock(period=hs.time.ms(10)))
                    .input(self.inp, hs.DrainAll())
                )
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            if mode == "batch":
                batch = self.inp.selected_batch()
                if not batch:
                    return
                sample = batch[0]
                selected.append((batch, sample))
                received.append(sample.data)
            elif mode == "selected":
                sample = self.inp.selected_sample()
                selected.append(sample)
                received.append(sample.data)
                assert sample.data == received[-1]
            elif mode == "receive":
                received.append(self.inp.receive().data)
            else:
                received.append(self.inp.receive_data())
            context.request_completion()

    graph = hs.Graph("schema-bytes", default_clock=hs.RealtimeClock)
    source = graph.op("holoscan.test.ByteSource" if authored is None else Source, "source")
    sink = graph.op(Sink, "sink")
    graph.add_flow(source["out"], sink["in"], queue_depth=1)
    if mode != "batch":
        graph.partition("serial").add(source).add(sink)
    plan = hs.compile(graph)
    assert plan.ok, [error.token for error in plan.errors]
    session = hs.run_async(plan)
    assert session.wait(timeout=10)
    assert session.diagnostics == []
    assert session.termination is hs.RunTermination.REQUESTED_COMPLETED
    return received[0], selected, rejected


# REQ: SCHEMA-PY-007
@pytest.mark.parametrize("size", [0, 1, 256, 407000, 416768])
@pytest.mark.parametrize("mode", ["receive", "receive_data", "selected", "batch"])
def test_native_byte_vector_is_owning_bytes(size, mode):
    value, selected, _ = run_graph(size=size, mode=mode)
    expected = bytes(range(256)) * (size // 256) + bytes(range(size % 256))
    assert type(value.blob) is bytes
    gc.collect()
    assert value.blob == expected
    assert value.numbers == [1, 128, 65536]
    for sample in selected:
        if mode == "batch":
            batch, item = sample
            assert item.data.blob == expected
            with pytest.raises(hs.errors.NotReadyError, match="PYTHON_SELECTED_BATCH_EXPIRED"):
                len(batch)
            continue
        with pytest.raises(hs.errors.NotReadyError, match="PYTHON_SELECTED_SAMPLE_EXPIRED"):
            _ = sample.data
    assert value.blob == expected


# REQ: SCHEMA-PY-007
@pytest.mark.parametrize("mapping", [False, True])
def test_byte_vector_authoring_round_trip_and_controls(mapping):
    value = schema.BytePayload(
        blob=b"\x00\x80\xff",
        nested=schema.ByteNested(blob=b"nested\x00"),
        signed_values=[-128, 0, 127],
        modes=[schema.ByteMode.Zero, schema.ByteMode.High],
        flags=[schema.ByteFlags.First | schema.ByteFlags.Last],
        fixed=schema.ByteArray(values=(0, 1, 128, 255)),
        numbers=[0, 256],
        labels=["hello"],
    )
    authored = value._holoscan_schema_values() if mapping else value
    received, _, _ = run_graph(authored=authored)
    assert received == value
    assert type(received.blob) is bytes
    assert type(received.nested.blob) is bytes
    assert type(received.signed_values) is list
    assert type(received.fixed.values) is tuple
    relayed, _, _ = run_graph(authored=received)
    assert relayed == value


# REQ: SCHEMA-PY-007
@pytest.mark.parametrize("authored", [{}, {"blob": b""}])
def test_byte_vector_absent_and_empty_defaults(authored):
    assert schema.BytePayload().blob == b""
    assert schema.ByteNested().blob == b""
    received, _, _ = run_graph(authored=authored)
    assert received.blob == b""
    assert received.nested is None


# REQ: SCHEMA-PY-007
@pytest.mark.parametrize(
    "invalid",
    [
        (-1, 256),
        [True],
        [1.5],
        ["x"],
        None,
        "abc",
        4,
        memoryview(b"abc"),
    ],
)
def test_byte_vector_rejects_invalid_input_without_consuming_publication(invalid):
    value, _, errors = run_graph(authored={"blob": b"valid"}, invalid={"blob": invalid})
    assert len(errors) == 1
    if isinstance(invalid, memoryview):
        assert isinstance(errors[0], TypeError)
    else:
        token = (
            "PYTHON_SCHEMA_FIELD_VALUE_OUT_OF_RANGE"
            if isinstance(invalid, tuple)
            else "PYTHON_SCHEMA_FIELD_TYPE_MISMATCH"
        )
        assert token in str(errors[0])
        assert "field=holoscan.test.BytePayload.blob" in str(errors[0])
    assert value.blob == b"valid"


# REQ: SCHEMA-PY-007
@pytest.mark.parametrize("field", ["signed_values", "numbers", "labels", "modes", "flags"])
def test_bytes_cannot_replace_other_vector_kinds(field):
    value, _, errors = run_graph(authored={"blob": b"valid"}, invalid={field: b"\x00\x01"})
    assert len(errors) == 1
    assert "PYTHON_SCHEMA_FIELD_TYPE_MISMATCH" in str(errors[0])
    assert f"field=holoscan.test.BytePayload.{field}" in str(errors[0])
    assert value.blob == b"valid"


# REQ: SCHEMA-PY-007
def test_byte_vector_standalone_channel_round_trip():
    name = f"byte-schema-{uuid.uuid4().hex}"
    with hs.channel.Runtime() as runtime:
        subscriber = runtime.subscriber(name, byte_channel_payload.ByteChannelPayload)
        publisher = runtime.publisher(name, byte_channel_payload.ByteChannelPayload)
        publisher.wait_for_subscribers(5.0)
        subscriber.refresh_connections()
        publisher.emit(byte_channel_payload.ByteChannelPayload(blob=bytes(range(256))), timeout=5.0)
        received = subscriber.receive(5.0).data
        assert type(received.blob) is bytes
        assert received.blob == bytes(range(256))
        publisher.emit(byte_channel_payload.ByteChannelPayload(blob=b"replacement"), timeout=5.0)
        assert subscriber.receive(5.0).data.blob == b"replacement"
        assert received.blob == bytes(range(256))
    gc.collect()
    assert received.blob == bytes(range(256))


# REQ: SCHEMA-PY-007
def test_byte_vector_descriptor_metadata_preserves_tensor_storage():
    received = []
    pointers = []

    class Source(hs.Operator):
        out = hs.Output("out", byte_tensor_payload.ByteTensorPayload)

        def setup(self, spec):
            spec.output(self.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage="runtime_pool",
                    memory_kind=hs.tensor.MemoryKind.HOST,
                    dtype=hs.tensor.DLDataType(1, 8, 1),
                    rank=1,
                    transfer=hs.tensor.TensorTransferPolicy(
                        foreign_emit_policy=hs.tensor.InterfaceTensorEmitPolicy.COPY_IF_NEEDED
                    ),
                ),
                hs.tensor.byte_span(4),
                field="data",
            )

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

        def compute(self, context):
            descriptor = byte_tensor_payload.ByteTensorPayload(blob=b"\x00\x80\xff")
            request = hs.tensor.LoanRequest(shape=(4,), dtype=hs.tensor.DLDataType(1, 8, 1))
            with self.out.allocate_tensors(descriptor, {"data": request}) as loan:
                tensor = loan.tensors["data"]
                pointers.append(tensor.data)
                (ctypes.c_uint8 * 4).from_address(tensor.data)[:] = (1, 2, 3, 4)

    class Sink(hs.Operator):
        inp = hs.Input("in", byte_tensor_payload.ByteTensorPayload)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            value = self.inp.receive_data()
            assert value.blob == b"\x00\x80\xff"
            assert type(value.blob) is bytes
            assert value.data.data == pointers[0]
            assert list((ctypes.c_uint8 * 4).from_address(value.data.data)) == [1, 2, 3, 4]
            received.append(value.blob)

    graph = hs.Graph("byte-descriptor", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    sink = graph.op(Sink, "sink")
    graph.add_flow(source.out, sink["in"])
    session = hs.run(hs.compile(graph))
    assert session.diagnostics == []
    assert received == [b"\x00\x80\xff"]


# REQ: SCHEMA-PY-007
def test_byte_vector_metadata_and_native_forward_do_not_materialize(monkeypatch):
    received = []
    byte_schema_support.configure(256)
    original = schema.BytePayload._from_holoscan_schema
    calls = []

    def counted(fields):
        calls.append(1)
        return original(fields)

    monkeypatch.setattr(schema.BytePayload, "_from_holoscan_schema", staticmethod(counted))

    class Relay(hs.Operator):
        inp = hs.Input("in", schema.BytePayload)
        out = hs.Output("out", schema.BytePayload)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            before = len(calls)
            sample = self.inp.selected_sample()
            assert isinstance(sample.metadata, hs.SampleMetadata)
            self.out.forward(sample)
            assert len(calls) == before

    class Sink(hs.Operator):
        inp = hs.Input("in", schema.BytePayload)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            received.append(self.inp.receive_data())
            context.request_completion()

    graph = hs.Graph("byte-forward", default_clock=hs.RealtimeClock)
    source = graph.op("holoscan.test.ByteSource", "source")
    relay = graph.op(Relay, "relay")
    sink = graph.op(Sink, "sink")
    graph.add_flow(source["out"], relay["in"], queue_depth=1)
    graph.add_flow(relay.out, sink["in"], queue_depth=1)
    graph.partition("serial").add(source).add(relay).add(sink)
    session = hs.run(hs.compile(graph))
    assert session.diagnostics == []
    assert len(calls) == 1
    assert received[0].blob == bytes(range(256))
