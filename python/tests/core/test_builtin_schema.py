# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Core-owned generated schema values and their canonical import paths."""

import ctypes

import numpy as np
import pytest

import holoscan as hs
from holoscan.schema import Header, Image, Imu, String, TensorPacket, Time
from holoscan.schema.geometry import Matrix3f, Quaternionf, Vector3f
from holoscan.schema.image import (
    UNKNOWN_IMAGE_SIGNIFICANT_BITS,
    ColorSpace,
    ImageEncoding,
    ImagePlaneLayout,
)
from holoscan.schema.imu import ImuFlags


class HeaderSource(hs.Operator):
    """Emit one built-in Header value."""

    out = hs.Output("out", Header)

    def setup(self, spec):
        spec.output(self.out)

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        self.out.emit(
            Header(
                stamp=Time(time_ns=13, acqtime_ns=11),
                frame_id="camera",
                capture_timestamp_ns=17,
                device_sequence=19,
            )
        )


class HeaderSink(hs.Operator):
    """Capture one built-in Header value."""

    inp = hs.Input("in", Header)
    received = None

    def setup(self, spec):
        spec.input(self.inp)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        type(self).received = self.inp.receive_data()


class CoreSchemaSource(hs.Operator):
    """Emit the remaining Core-owned schema roots."""

    string_out = hs.Output("string_out", String)
    time_out = hs.Output("time_out", Time)
    packet_out = hs.Output("packet_out", TensorPacket)
    token_error = None

    def setup(self, spec):
        spec.output(self.string_out)
        spec.output(self.time_out)
        spec.output(self.packet_out).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=1,
                transfer=hs.tensor.TensorTransferPolicy(
                    foreign_emit_policy=hs.tensor.InterfaceTensorEmitPolicy.COPY_IF_NEEDED
                ),
            ),
            hs.tensor.byte_span(6),
            field="data",
        )

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        self.string_out.emit(String(data="hello"))
        try:
            self.time_out.emit(hs.payload.schema.Time)
        except TypeError as error:
            type(self).token_error = str(error)
        self.time_out.emit(Time(time_ns=41, acqtime_ns=37))
        self.packet_out.emit(
            TensorPacket(
                sequence=43,
                source="camera",
                scale=0.25,
                data=hs.Tensor.as_tensor(np.arange(6, dtype=np.uint8)),
            )
        )


class StringSink(hs.Operator):
    """Capture one built-in String value."""

    inp = hs.Input("in", String)
    received = None

    def setup(self, spec):
        spec.input(self.inp)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        type(self).received = self.inp.receive_data()


class TimeSink(hs.Operator):
    """Capture one built-in Time value."""

    inp = hs.Input("in", Time)
    received = None

    def setup(self, spec):
        spec.input(self.inp)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        type(self).received = self.inp.receive_data()


class TensorPacketSink(hs.Operator):
    """Capture one built-in TensorPacket value."""

    inp = hs.Input("in", TensorPacket)
    received = None

    def setup(self, spec):
        spec.input(self.inp)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        type(self).received = self.inp.receive_data()


class ImageSource(hs.Operator):
    """Emit one sensor Image through its component-owned descriptor codec."""

    out = hs.Output("out", Image)
    rejected_code = None
    rejected_token = None

    def setup(self, spec):
        spec.output(self.out).produces_tensor(
            hs.tensor.OutputSpec(
                storage="runtime_pool",
                memory_kind=hs.tensor.MemoryKind.HOST,
                dtype=hs.tensor.DLDataType(1, 8, 1),
                rank=2,
                transfer=hs.tensor.TensorTransferPolicy(
                    foreign_emit_policy=hs.tensor.InterfaceTensorEmitPolicy.COPY_IF_NEEDED
                ),
            ),
            hs.tensor.byte_span(6),
            field="data",
        )

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        invalid = Image(
            height=-1,
            width=3,
            encoding=ImageEncoding.MONO8,
            data=hs.Tensor.as_tensor(np.arange(6, dtype=np.uint8).reshape(2, 3)),
        )
        try:
            self.out.emit(invalid)
        except hs.errors.InvalidArgumentError as error:
            type(self).rejected_code = error.code
            type(self).rejected_token = error.token

        descriptor = Image(
            header=Header(frame_id="camera", capture_timestamp_ns=23, device_sequence=29),
            height=2,
            width=3,
            encoding=ImageEncoding.MONO8,
            color_space=ColorSpace.UNSPECIFIED,
            significant_bits=UNKNOWN_IMAGE_SIGNIFICANT_BITS,
            exposure_time_ns=31,
            gain=1.5,
            data=hs.Tensor.as_tensor(np.arange(6, dtype=np.uint8).reshape(2, 3)),
        )
        self.out.emit(descriptor)


class ImageSink(hs.Operator):
    """Capture one generated Image value."""

    inp = hs.Input("in", Image)
    received = None

    def setup(self, spec):
        spec.input(self.inp)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        type(self).received = self.inp.receive_data()


class ImuSource(hs.Operator):
    """Emit one generated Imu value."""

    out = hs.Output("out", Imu)

    def setup(self, spec):
        spec.output(self.out)

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        self.out.emit(
            Imu(
                header=Header(frame_id="imu", capture_timestamp_ns=37),
                orientation=Quaternionf(w=1.0),
                angular_velocity=Vector3f(x=1.0, y=2.0, z=3.0),
                linear_acceleration=Vector3f(z=9.8),
                flags=ImuFlags.ORIENTATION_VALID | ImuFlags.UNCALIBRATED,
                orientation_covariance=Matrix3f(m=range(9)),
            )
        )


class ImuSink(hs.Operator):
    """Capture one generated Imu value."""

    inp = hs.Input("in", Imu)
    received = None

    def setup(self, spec):
        spec.input(self.inp)

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        type(self).received = self.inp.receive_data()


# REQ: SCHEMA-PY-004
def test_core_schema_values_have_canonical_imports_and_registry_rows():
    # Key setup: Import each rooted Core schema through holoscan.schema.
    values = (
        Header(),
        String(data="value"),
        TensorPacket(sequence=3, source="source", scale=0.5),
        Time(time_ns=5, acqtime_ns=4),
    )

    # Key assertion: The value classes are the canonical registered payload facades.
    assert hs.schema.Header is Header
    for value in values:
        token = hs.payload.resolve(type(value)._registry_name)
        assert token.name == type(value)._registry_name
        assert token == getattr(hs.payload.schema, type(value).__name__)


# REQ: SCHEMA-PY-004
def test_schema_payload_tokens_explain_value_construction():
    # Key setup: Resolve the token and value namespaces for the same schema identity.
    token = hs.payload.schema.Time

    # Key assertion: Calling a token points users to the corresponding value constructor.
    assert hs.payload.schema.Header == hs.payload.resolve(Header._registry_name)
    assert token == hs.payload.resolve(Time._registry_name)
    assert token is not Time
    with pytest.raises(
        TypeError,
        match=r"holoscan\.payload\.schema\.Time is a payload identity token; "
        r"construct values with holoscan\.schema\.Time\(\.\.\.\) instead",
    ):
        token()


# REQ: SCHEMA-PY-004
@pytest.mark.parametrize("kind", ["Time", "String", "Header"])
@pytest.mark.parametrize("use_cached_token", [True, False])
def test_builtin_token_declared_emit(kind, use_cached_token):
    # Key setup: Declare an output with either the import-time token or a current registry lookup.
    payload = getattr(hs.payload.schema, kind)
    if not use_cached_token:
        payload = hs.payload.resolve(payload.name)
    value = {
        "Time": {"time_ns": 41},
        "String": {"data": "hello"},
        "Header": {"frame_id": "camera"},
    }[kind]

    class Source(hs.Operator):
        out = hs.Output("out", payload)

        def setup(self, spec):
            spec.output(self.out)

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

        def compute(self, context):
            self.out.emit(value)

    class Sink(hs.Operator):
        inp = hs.Input("in", getattr(hs.schema, kind))
        received = None

        def setup(self, spec):
            spec.input(self.inp)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            type(self).received = self.inp.receive_data()

    graph = hs.Graph("builtin-token-output", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    sink = graph.op(Sink, "sink")
    graph.add_flow(source.out, sink["in"])

    plan = hs.compile(graph)
    assert plan.ok, [error.token for error in plan.errors]
    session = hs.run(plan)

    # Key assertion: Historical tokens dispatch through the currently installed schema facade.
    assert session.diagnostics == []
    assert Sink.received is not None
    for field, expected in value.items():
        assert getattr(Sink.received, field) == expected


# REQ: SCHEMA-PY-004
@pytest.mark.parametrize("read_mode", ["receive", "receive_data", "selected_batch"])
@pytest.mark.parametrize("use_cached_token", [True, False])
def test_builtin_token_declared_receive(read_mode, use_cached_token):
    # Key setup: Declare an input with either the import-time token or a current registry lookup.
    payload = hs.payload.schema.Time
    if not use_cached_token:
        payload = hs.payload.resolve(payload.name)

    class Source(hs.Operator):
        out = hs.Output("out", Time)

        def setup(self, spec):
            spec.output(self.out)

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

        def compute(self, context):
            self.out.emit(Time(time_ns=41))

    class Sink(hs.Operator):
        inp = hs.Input("in", payload)
        received = None

        def setup(self, spec):
            spec.input(self.inp)

        def contract(self):
            if read_mode == "selected_batch":
                return (
                    hs.Contract()
                    .trigger(hs.OnClock(period=hs.time.ms(10)))
                    .input(self.inp, hs.DrainAll())
                )
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, ctx):
            if read_mode == "receive":
                type(self).received = self.inp.receive().data
            elif read_mode == "receive_data":
                type(self).received = self.inp.receive_data()
            else:
                batch = self.inp.selected_batch()
                if batch:
                    type(self).received = batch[0].data
                    ctx.request_completion()

    graph = hs.Graph("builtin-token-input", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    sink = graph.op(Sink, "sink")
    graph.add_flow(source.out, sink["in"])

    plan = hs.compile(graph)
    assert plan.ok, [error.token for error in plan.errors]
    session = hs.run(plan)

    # Key assertion: Every token-declared input route uses the active schema facade callbacks.
    assert session.diagnostics == []
    assert Sink.received["time_ns"] == 41


# REQ: SCHEMA-PY-004
def test_geometry_values_are_frozen_and_fixed_arrays_are_tuples():
    # Key setup: Construct the rootless geometry helper values from ordinary Python inputs.
    vector = Vector3f(x=1.0, y=2.0, z=3.0)
    quaternion = Quaternionf(x=0.0, y=0.0, z=0.0, w=1.0)
    matrix = Matrix3f(m=range(9))

    # Key assertion: Struct values are immutable and fixed arrays retain exact tuple length.
    assert (vector.x, vector.y, vector.z) == (1.0, 2.0, 3.0)
    assert quaternion.w == 1.0
    assert matrix.m == tuple(range(9))
    with pytest.raises(AttributeError):
        vector.x = 4.0
    with pytest.raises(ValueError, match="exactly 9 elements"):
        Matrix3f(m=(1.0, 2.0))


# REQ: SCHEMA-PY-004
def test_header_round_trip_reuses_the_canonical_time_class():
    # Key setup: Carry Header through the native schema codec with its included Time value.
    graph = hs.Graph("builtin-header-round-trip", default_clock=hs.RealtimeClock)
    HeaderSink.received = None
    source = graph.op(HeaderSource, "source")
    sink = graph.op(HeaderSink, "sink")
    graph.add_flow(source.out, sink["in"])

    plan = hs.compile(graph)
    assert plan.ok, [error.token for error in plan.errors]
    hs.run(plan)

    # Key assertion: Header does not generate a competing nested Time class.
    assert isinstance(HeaderSink.received, Header)
    assert HeaderSink.received.stamp.__class__ is Time
    assert HeaderSink.received.stamp.time_ns == 13
    assert HeaderSink.received.stamp.acqtime_ns == 11
    assert HeaderSink.received.frame_id == "camera"
    assert HeaderSink.received.capture_timestamp_ns == 17
    assert HeaderSink.received.device_sequence == 19


# REQ: SCHEMA-PY-004
def test_core_schema_roots_round_trip_with_tensor_mapping():
    # Key setup: Carry String, Time, and TensorPacket values through their native codecs.
    graph = hs.Graph("builtin-core-schema-round-trip", default_clock=hs.RealtimeClock)
    CoreSchemaSource.token_error = None
    StringSink.received = None
    TimeSink.received = None
    TensorPacketSink.received = None
    source = graph.op(CoreSchemaSource, "source")
    string_sink = graph.op(StringSink, "string_sink")
    time_sink = graph.op(TimeSink, "time_sink")
    packet_sink = graph.op(TensorPacketSink, "packet_sink")
    graph.add_flow(source.string_out, string_sink["in"])
    graph.add_flow(source.time_out, time_sink["in"])
    graph.add_flow(source.packet_out, packet_sink["in"])

    plan = hs.compile(graph)
    assert plan.ok, [error.token for error in plan.errors]
    hs.run(plan)

    # Key assertion: Every root rematerializes its canonical class and packet Tensor mapping.
    assert StringSink.received.__class__ is String
    assert StringSink.received.data == "hello"
    assert TimeSink.received.__class__ is Time
    assert (TimeSink.received.time_ns, TimeSink.received.acqtime_ns) == (41, 37)
    packet = TensorPacketSink.received
    assert packet.__class__ is TensorPacket
    assert (packet.sequence, packet.source, packet.scale) == (43, "camera", 0.25)
    assert set(packet.tensors) == {"data"}
    assert packet.data.__class__ is hs.Tensor
    assert packet.tensors["data"] is packet.data
    assert bytes((ctypes.c_uint8 * 6).from_address(packet.data.data)) == bytes(range(6))
    assert CoreSchemaSource.token_error == (
        "holoscan.payload.schema.Time is a payload identity token; "
        "emit an instance of holoscan.schema.Time instead"
    )
    TensorPacketSink.received = None


# REQ: SCHEMA-PY-004
def test_sensor_schema_values_have_canonical_imports_and_shapes():
    # Key setup: Construct both Sensor I/O values through their public component paths.
    plane = ImagePlaneLayout(offset_bytes=8, row_stride_bytes=16, size_bytes=32)
    imu = Imu(
        header=Header(frame_id="imu"),
        flags=ImuFlags.ORIENTATION_VALID | ImuFlags.GYROSCOPE_SATURATED,
        orientation_covariance=Matrix3f(m=range(9)),
    )

    # Key assertion: Root exports reuse the module classes and retain exact production shapes.
    assert hs.schema.Image is Image
    assert hs.schema.Imu is Imu
    assert plane == ImagePlaneLayout(offset_bytes=8, row_stride_bytes=16, size_bytes=32)
    assert imu.header.__class__ is Header
    assert imu.orientation.__class__ is Quaternionf
    assert imu.orientation_covariance.__class__ is Matrix3f
    assert imu.orientation_covariance.m == tuple(range(9))
    assert imu.flags & ImuFlags.GYROSCOPE_SATURATED
    assert hs.payload.resolve(Image._registry_name).name == Image._registry_name
    assert hs.payload.resolve(Imu._registry_name).name == Imu._registry_name


# REQ: SCHEMA-PY-004
def test_image_round_trip_uses_custom_validation_and_preserves_tensor_lifetime():
    # Key setup: Carry one generated Image and its Tensor through native runtime storage.
    graph = hs.Graph("builtin-image-round-trip", default_clock=hs.RealtimeClock)
    ImageSource.rejected_code = None
    ImageSource.rejected_token = None
    ImageSink.received = None
    source = graph.op(ImageSource, "source")
    sink = graph.op(ImageSink, "sink")
    graph.add_flow(source.out, sink["in"])

    plan = hs.compile(graph)
    assert plan.ok, [error.token for error in plan.errors]
    hs.run(plan)

    # Key assertion: Image uses the sensor codec's semantic verdict and one named Tensor field.
    image = ImageSink.received
    assert isinstance(image, Image)
    assert image.header.__class__ is Header
    assert image.header.frame_id == "camera"
    assert image.encoding is ImageEncoding.MONO8
    assert UNKNOWN_IMAGE_SIGNIFICANT_BITS == 255
    assert image.significant_bits == UNKNOWN_IMAGE_SIGNIFICANT_BITS
    assert image.plane_layouts == []
    assert set(image.tensors) == {"data"}
    assert image.data.__class__ is hs.Tensor
    assert image.tensors["data"] is image.data
    assert bytes((ctypes.c_uint8 * 6).from_address(image.data.data)) == bytes(range(6))
    assert ImageSource.rejected_code is hs.errors.ErrorCode.INVALID_ARGUMENT
    assert ImageSource.rejected_token
    ImageSink.received = None


# REQ: SCHEMA-PY-005
# REQ: SCHEMA-PY-004
# REQ: GRAPH-043
@pytest.mark.parametrize(
    ("encoding", "channels"),
    [(ImageEncoding.RGB16, 3), (ImageEncoding.RGBA16, 4)],
)
def test_16_bit_image_tensor_field_projects_into_python_tensor_input(encoding, channels):
    # Key setup: Publish a generated Image from one provider-owned Tensor loan and explicitly
    # narrow its data field to a Tensor edge. The allocation path has no fallback copy.
    expected = np.arange(2 * 3 * channels, dtype=np.uint16).reshape(2, 3, channels)

    class Source(hs.Operator):
        out = hs.Output("out", Image)
        published_tensor_address = None

        def setup(self, spec):
            spec.output(self.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage="runtime_pool",
                    memory_kind=hs.tensor.MemoryKind.HOST,
                    dtype=hs.tensor.DLDataType(1, 16, 1),
                    rank=3,
                ),
                hs.tensor.byte_span(expected.nbytes),
                field="data",
            )

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

        def compute(self, context):
            descriptor = Image(
                height=2,
                width=3,
                encoding=encoding,
                significant_bits=16,
            )
            request = hs.tensor.LoanRequest(
                shape=expected.shape, dtype=hs.tensor.DLDataType(1, 16, 1)
            )
            with self.out.allocate_tensors(descriptor, {"data": request}) as loan:
                tensor = loan.tensors["data"]
                type(self).published_tensor_address = tensor.data
                destination = (ctypes.c_uint16 * expected.size).from_address(tensor.data)
                destination[:] = tuple(int(value) for value in expected.flat)

    class TensorSink(hs.Operator):
        inp = hs.Input("in", hs.Tensor)
        received_address = None
        received_values = None

        def setup(self, spec):
            spec.input(self.inp).accepts_projected_tensor()

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            tensor = self.inp.receive_data()
            type(self).received_address = tensor.data
            type(self).received_values = np.from_dlpack(tensor).copy()

    graph = hs.Graph("python-image-tensor-projection", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    sink = graph.op(TensorSink, "sink")
    projection = source.out.project("data")
    assert isinstance(projection, hs.tensor.TensorProjection)
    assert projection.field_name == "data"
    assert projection.source.name == "out"
    with pytest.raises(
        hs.errors.InvalidArgumentError, match="GRAPH_PROJECTED_TENSOR_FIELD_UNKNOWN"
    ):
        source.out.project("missing")
    flow = graph.add_flow(projection, sink["in"])
    assert flow.source.name == "out"
    assert flow.tensor_projection is not None
    assert flow.tensor_projection.field_name == "data"
    assert flow.tensor_projection.source.name == flow.source.name
    assert repr(flow) == "<Flow source.out -> sink.in tensor_projection='data'>"

    plan = hs.compile(graph)
    assert plan.ok, [error.token for error in plan.errors]
    session = hs.run(plan)

    # Key assertion: Both production 16-bit encodings retain their generated Image validation and
    # expose the producer's runtime-owned backing without a projection copy or conversion.
    assert session.diagnostics == []
    assert TensorSink.received_address == Source.published_tensor_address
    np.testing.assert_array_equal(TensorSink.received_values, expected)


# REQ: GRAPH-043
def test_tensor_field_projection_rejects_invalid_python_authoring():
    # Key setup: Reuse one generated Image output declaration across projection boundary failures.
    class Source(hs.Operator):
        out = hs.Output("out", Image)

        def setup(self, spec):
            spec.output(self.out).produces_tensor(
                hs.tensor.OutputSpec(
                    storage="runtime_pool",
                    memory_kind=hs.tensor.MemoryKind.HOST,
                    dtype=hs.tensor.DLDataType(1, 16, 1),
                    rank=3,
                ),
                hs.tensor.byte_span(2 * 3 * 3 * 2),
                field="data",
            )

        def compute(self, context):
            pass

    class PlainTensorSink(hs.Operator):
        inp = hs.Input("in", hs.Tensor)

        def setup(self, spec):
            spec.input(self.inp)

        def compute(self, context):
            pass

    class ProjectedTensorSink(hs.Operator):
        inp = hs.Input("in", hs.Tensor)

        def setup(self, spec):
            spec.input(self.inp).accepts_projected_tensor()

        def compute(self, context):
            pass

    class ScalarSink(hs.Operator):
        inp = hs.Input("in", hs.payload.uint32)

        def setup(self, spec):
            spec.input(self.inp)

        def compute(self, context):
            pass

    class ScalarSource(hs.Operator):
        out = hs.Output("out", hs.payload.uint32)

        def setup(self, spec):
            spec.output(self.out)

        def compute(self, context):
            pass

    missing_opt_in_graph = hs.Graph(
        "python-projection-missing-opt-in", default_clock=hs.RealtimeClock
    )
    source = missing_opt_in_graph.op(Source, "source")
    sink = missing_opt_in_graph.op(PlainTensorSink, "sink")
    with pytest.raises(hs.errors.InvalidArgumentError) as unknown_field:
        source.out.project("missing")
    with pytest.raises(hs.errors.IncompatibleError) as missing_opt_in:
        missing_opt_in_graph.add_flow(source.out.project("data"), sink["in"])

    unsupported_source_graph = hs.Graph("python-projection-unsupported-source")
    scalar_source = unsupported_source_graph.op(ScalarSource, "scalar_source")
    with pytest.raises(hs.errors.NotSupportedError) as unsupported_source:
        scalar_source.out.project("data")

    non_tensor_graph = hs.Graph("python-projection-non-tensor-target")
    source = non_tensor_graph.op(Source, "source")
    scalar_sink = non_tensor_graph.op(ScalarSink, "sink")
    with pytest.raises(hs.errors.IncompatibleError) as non_tensor:
        non_tensor_graph.add_flow(source.out.project("data"), scalar_sink["in"])

    source_graph = hs.Graph("python-projection-source-graph")
    source = source_graph.op(Source, "source")
    target_graph = hs.Graph("python-projection-target-graph")
    target = target_graph.op(ProjectedTensorSink, "target")
    with pytest.raises(hs.errors.InvalidArgumentError) as cross_graph:
        source_graph.add_flow(source.out.project("data"), target["in"])

    # Key assertion: Python retains Core's explicit admission, exact target type, and Graph
    # ownership boundaries with their canonical diagnostics and actionable projection context.
    assert unknown_field.value.token == "GRAPH_PROJECTED_TENSOR_FIELD_UNKNOWN"
    assert "port=source.out" in str(unknown_field.value)
    assert "field_name=missing" in str(unknown_field.value)
    assert "known_fields=data" in str(unknown_field.value)
    assert unsupported_source.value.token == "GRAPH_TENSOR_PROJECTION_SOURCE_UNSUPPORTED"
    assert "port=scalar_source.out" in str(unsupported_source.value)
    assert "field_name=data" in str(unsupported_source.value)
    assert missing_opt_in.value.token == "GRAPH_PROJECTED_TENSOR_NOT_ACCEPTED"
    assert non_tensor.value.token == "GRAPH_TENSOR_PROJECTION_TARGET_UNSUPPORTED"
    assert "source=source.out" in str(non_tensor.value)
    assert "source_payload=holoscan.schema.Image" in str(non_tensor.value)
    assert "field_name=data" in str(non_tensor.value)
    assert "target=sink.in" in str(non_tensor.value)
    assert "actual_target_payload=holoscan.uint32" in str(non_tensor.value)
    assert "required_target_payload=holoscan.Tensor" in str(non_tensor.value)
    assert cross_graph.value.token == "GRAPH_CROSS_GRAPH_PORT"


# REQ: SCHEMA-PY-004
def test_imu_round_trip_reuses_header_and_geometry_classes():
    # Key setup: Carry an Imu with flags, inline geometry, and a fixed matrix through native C++.
    graph = hs.Graph("builtin-imu-round-trip", default_clock=hs.RealtimeClock)
    ImuSink.received = None
    source = graph.op(ImuSource, "source")
    sink = graph.op(ImuSink, "sink")
    graph.add_flow(source.out, sink["in"])

    plan = hs.compile(graph)
    assert plan.ok, [error.token for error in plan.errors]
    hs.run(plan)

    # Key assertion: Decoding reuses the canonical component-independent helper classes exactly.
    imu = ImuSink.received
    assert isinstance(imu, Imu)
    assert imu.header.__class__ is Header
    assert imu.orientation.__class__ is Quaternionf
    assert imu.angular_velocity.__class__ is Vector3f
    assert imu.orientation_covariance.__class__ is Matrix3f
    assert imu.orientation_covariance.m == tuple(range(9))
    assert imu.flags == ImuFlags.ORIENTATION_VALID | ImuFlags.UNCALIBRATED
