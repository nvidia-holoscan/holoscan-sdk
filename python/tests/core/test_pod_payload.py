# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python access to raw-POD payload types registered from C++."""

import struct
from collections import defaultdict
from collections.abc import Mapping

import holoscan_test_support
import pytest

import holoscan as hs

# holoscan_test_support registers the C++ raw-POD type and its Operators on import.
POSE = hs.payload.resolve("holoscan.test.Pose")
POSE_SOURCE = "holoscan.test.PoseSource"
POSE_SINK = "holoscan.test.PoseSink"
# The value the C++ PoseSource emits, as a received Python value.
EXPECTED = {
    "sequence": 7,
    "position": (1.5, -2.25, 3.0),
    "confidence": 0.5,
    "quadrant": -3,
    "flags": (1, 254),
}
# Native layout of holoscan.test.Pose: uint64, 3 x float64, float32, int16, 2 x uint8; 40 bytes.
POSE_LAYOUT = struct.Struct("=Q3dfh2B")
# holoscan.test.Singleton has a uint32 scalar and two one-element arrays, uint32[1] and
# std::array<float, 1>, which Python exchanges as one-element tuples.
SINGLETON = hs.payload.resolve("holoscan.test.Singleton")
SINGLETON_VALUE = {"count": 3, "ids": (42,), "gains": (0.5,)}
SINGLETON_LAYOUT = struct.Struct("=2If")


def native_pose(value):
    return POSE_LAYOUT.pack(
        value["sequence"],
        *value["position"],
        value["confidence"],
        value["quadrant"],
        *value["flags"],
    )


class PoseEmitter(hs.Operator):
    """Emit one configured value; used by the raw-POD tests."""

    out = hs.Output("out", POSE)
    value = None
    error = None

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        try:
            self.out.emit(type(self).value)
        except Exception as error:  # noqa: BLE001 - tests inspect every emit failure
            type(self).error = error


class SingletonEmitter(hs.Operator):
    """Emit one configured one-element-array value; used by the raw-POD tests."""

    out = hs.Output("out", SINGLETON)
    value = None
    error = None

    def contract(self):
        return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

    def compute(self, context):
        try:
            self.out.emit(type(self).value)
        except hs.HoloscanError as error:
            type(self).error = error


class SingletonRecorder(hs.Operator):
    """Record each received one-element-array value; used by the raw-POD tests."""

    inp = hs.Input("in", SINGLETON)
    received = []

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        type(self).received.append(self.inp.receive_data())


class DerivedInt(int):
    """An int subclass, which takes the generic scalar conversion; used by the raw-POD tests."""


class ListClearingInt(int):
    """An int whose float conversion empties the list that holds it; used by the raw-POD tests."""

    target = None

    def __float__(self):
        type(self).target.clear()
        return float(int(self))


class FailingLengthTuple(tuple):
    """A sequence whose length lookup raises; used by the raw-POD tests."""

    error = None

    def __len__(self):
        raise type(self).error


class CountingMapping(Mapping):
    """Record every value lookup of a field mapping; used by the raw-POD tests."""

    def __init__(self, values, *, getter_error=None):
        self.values = dict(values)
        self.lookups = []
        self.getter_error = getter_error

    def __getitem__(self, key):
        self.lookups.append(key)
        if self.getter_error is not None:
            raise self.getter_error
        return self.values[key]

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)


class PoseRecorder(hs.Operator):
    """Record each received value through the configured read path; used by the raw-POD tests."""

    inp = hs.Input("in", POSE)
    mode = "receive"
    received = []

    def contract(self):
        return hs.Contract().trigger(hs.OnEach(self.inp))

    def compute(self, context):
        if type(self).mode == "receive":
            value = self.inp.receive().data
        elif type(self).mode == "receive_data":
            value = self.inp.receive_data()
        else:
            value = self.inp.selected_sample().data
        type(self).received.append(value)


class PoseBatchRecorder(hs.Operator):
    """Drain raw-POD values through the selected-batch path; used by the raw-POD tests."""

    inp = hs.Input("in", POSE)
    received = []

    def setup(self, spec):
        spec.input(self.inp, queue_depth=4)

    def contract(self):
        return (
            hs.Contract().trigger(hs.OnClock(period=hs.time.ms(5))).input(self.inp, hs.DrainAll())
        )

    def compute(self, context):
        type(self).received.extend(sample.data for sample in self.inp.selected_batch())
        if type(self).received:
            context.request_completion()


def reset():
    PoseEmitter.value = None
    PoseEmitter.error = None
    PoseRecorder.mode = "receive"
    PoseRecorder.received = []
    PoseBatchRecorder.received = []
    SingletonEmitter.value = None
    SingletonEmitter.error = None
    SingletonRecorder.received = []
    holoscan_test_support._reset_test_pose()


def run_pose_to_cpp(value):
    PoseEmitter.value = value
    graph = hs.Graph("pod-python-to-cpp", default_clock=hs.RealtimeClock)
    emitter = graph.op(PoseEmitter, "emitter")
    sink = graph.op(POSE_SINK, "sink")
    graph.add_flow(emitter.out, sink["in"])
    return hs.run(hs.compile(graph))


# REQ: PY-OP-014
def test_cpp_pod_payload_reaches_a_python_operator_as_a_field_dict():
    # Key setup: Send the C++ raw-POD value to a Python Operator.
    #
    #   source[C++] --holoscan.test.Pose--> recorder[Python]
    reset()
    graph = hs.Graph("pod-cpp-to-python", default_clock=hs.RealtimeClock)
    source = graph.op(POSE_SOURCE, "source")
    recorder = graph.op(PoseRecorder, "recorder")
    graph.add_flow(source.out, recorder["in"])

    session = hs.run(hs.compile(graph))

    # Key assertion: Numeric POD arrays, including uint8, retain tuple output.
    assert session.diagnostics == []
    assert PoseRecorder.received == [EXPECTED]


# REQ: PY-OP-014
@pytest.mark.parametrize(
    "value",
    [
        EXPECTED,
        {**EXPECTED, "position": list(EXPECTED["position"]), "flags": list(EXPECTED["flags"])},
        {**EXPECTED, "sequence": DerivedInt(7), "position": (1.5, -2.25, DerivedInt(3))},
        {**EXPECTED, "flags": bytes(EXPECTED["flags"])},
        {**EXPECTED, "flags": bytearray(EXPECTED["flags"])},
        native_pose(EXPECTED),
        bytearray(native_pose(EXPECTED)),
    ],
    ids=[
        "mapping",
        "list-arrays",
        "ints-and-int-subclasses",
        "bytes-field",
        "bytearray-field",
        "bytes",
        "bytearray",
    ],
)
def test_python_operator_emits_a_pod_payload_to_a_cpp_operator(value):
    # Key setup: Emit the raw-POD value from Python, as a field mapping or as native bytes.
    #
    #   emitter[Python] --holoscan.test.Pose--> sink[C++]
    reset()

    session = run_pose_to_cpp(value)

    # Key assertion: The C++ Operator receives exactly the native object the value describes.
    assert session.diagnostics == []
    assert PoseEmitter.error is None
    assert holoscan_test_support._last_test_pose() == EXPECTED


# REQ: PY-OP-014
@pytest.mark.parametrize("mode", ["receive", "receive_data", "selected_sample", "selected_batch"])
def test_pod_payload_round_trips_between_python_operators(mode):
    # Key setup: Send a raw-POD value between Python Operators and read it through each path.
    #
    #   emitter[Python] --holoscan.test.Pose--> recorder[Python]
    reset()
    PoseEmitter.value = EXPECTED
    PoseRecorder.mode = mode
    recorder_type = PoseBatchRecorder if mode == "selected_batch" else PoseRecorder
    graph = hs.Graph(
        f"pod-python-round-trip-{mode.replace('_', '-')}", default_clock=hs.RealtimeClock
    )
    emitter = graph.op(PoseEmitter, "emitter")
    recorder = graph.op(recorder_type, "recorder")
    graph.add_flow(emitter.out, recorder["in"], queue_depth=4)

    session = hs.run(hs.compile(graph))

    # Key assertion: Every read path returns the same field dict that was emitted.
    assert session.diagnostics == []
    assert recorder_type.received == [EXPECTED]


# REQ: PY-OP-014
@pytest.mark.parametrize(
    ("value", "token", "detail"),
    [
        ({**EXPECTED, "flags": (1, 256)}, "AUTHORING_PAYLOAD_VALUE_OUT_OF_RANGE", "'flags'"),
        ({**EXPECTED, "flags": b"\x01"}, "AUTHORING_POD_FIELD_LENGTH_MISMATCH", "'flags'"),
        ({**EXPECTED, "flags": bytearray(3)}, "AUTHORING_POD_FIELD_LENGTH_MISMATCH", "'flags'"),
        ({**EXPECTED, "sequence": 7.5}, "AUTHORING_PAYLOAD_VALUE_TYPE_MISMATCH", "'sequence'"),
        ({**EXPECTED, "position": (1.0, 2.0)}, "AUTHORING_POD_FIELD_LENGTH_MISMATCH", "'position'"),
        (
            {key: item for key, item in EXPECTED.items() if key != "quadrant"},
            "AUTHORING_POD_FIELD_MISSING",
            "'quadrant'",
        ),
        ({**EXPECTED, "extra": 1}, "AUTHORING_POD_FIELD_UNKNOWN", "5 fields"),
        (native_pose(EXPECTED)[:-1], "AUTHORING_POD_PAYLOAD_SIZE_MISMATCH", "40 bytes"),
        (list(EXPECTED.values()), "AUTHORING_PAYLOAD_VALUE_TYPE_MISMATCH", "holoscan.test.Pose"),
    ],
    ids=[
        "out-of-range",
        "short-byte-field",
        "long-byte-field",
        "wrong-type",
        "wrong-length",
        "missing-field",
        "unknown-key",
        "short-buffer",
        "not-a-mapping",
    ],
)
def test_pod_payload_emit_rejects_a_malformed_value(value, token, detail):
    # Key setup: Emit a malformed raw-POD value and catch the error inside the callback.
    #
    #   emitter[Python] --holoscan.test.Pose--> recorder[Python]
    reset()
    PoseEmitter.value = value
    graph = hs.Graph("pod-python-malformed", default_clock=hs.RealtimeClock)
    emitter = graph.op(PoseEmitter, "emitter")
    recorder = graph.op(PoseRecorder, "recorder")
    graph.add_flow(emitter.out, recorder["in"])

    session = hs.run(hs.compile(graph))

    # Key assertion: The error names its cause, and nothing reaches the consumer.
    assert session.diagnostics == []
    assert PoseEmitter.error is not None
    assert PoseEmitter.error.token == token
    assert detail in str(PoseEmitter.error)
    assert PoseRecorder.received == []


# REQ: PY-OP-014
def test_pod_payload_emit_checks_mapping_keys_before_reading_values():
    # Key setup: Emit a defaultdict that lacks a field its default factory could supply.
    #
    #   emitter[Python] --holoscan.test.Pose--> sink[C++]
    reset()
    value = defaultdict(int, {key: item for key, item in EXPECTED.items() if key != "quadrant"})

    session = run_pose_to_cpp(value)

    # Key assertion: The missing field is rejected, nothing is sent, and the mapping is unchanged.
    assert session.diagnostics == []
    assert PoseEmitter.error is not None
    assert PoseEmitter.error.token == "AUTHORING_POD_FIELD_MISSING"
    assert "'quadrant'" in str(PoseEmitter.error)
    assert "quadrant" not in value
    assert holoscan_test_support._last_test_pose() is None


# REQ: PY-OP-014
def test_pod_payload_emit_reads_each_mapping_value_once():
    # Key setup: Emit a custom mapping that records each value lookup.
    #
    #   emitter[Python] --holoscan.test.Pose--> sink[C++]
    reset()
    value = CountingMapping(EXPECTED)

    session = run_pose_to_cpp(value)

    # Key assertion: The C++ Operator receives the value after exactly one lookup per field.
    assert session.diagnostics == []
    assert PoseEmitter.error is None
    assert sorted(value.lookups) == sorted(EXPECTED)
    assert holoscan_test_support._last_test_pose() == EXPECTED


# REQ: PY-OP-014
@pytest.mark.parametrize(
    ("fields", "token", "detail"),
    [
        ({**EXPECTED, "extra": 1}, "AUTHORING_POD_FIELD_UNKNOWN", "5 fields"),
        (
            {key: item for key, item in EXPECTED.items() if key != "quadrant"},
            "AUTHORING_POD_FIELD_MISSING",
            "'quadrant'",
        ),
    ],
    ids=["unknown-key", "missing-field"],
)
def test_pod_payload_emit_rejects_mapping_keys_before_calling_getters(fields, token, detail):
    # Key setup: Emit a custom mapping with wrong keys whose getter raises if it is ever called.
    #
    #   emitter[Python] --holoscan.test.Pose--> sink[C++]
    reset()
    value = CountingMapping(fields, getter_error=RuntimeError("getter called"))

    session = run_pose_to_cpp(value)

    # Key assertion: The key error wins without any value lookup, and nothing is sent.
    assert session.diagnostics == []
    assert PoseEmitter.error is not None
    assert PoseEmitter.error.token == token
    assert detail in str(PoseEmitter.error)
    assert value.lookups == []
    assert holoscan_test_support._last_test_pose() is None


# REQ: PY-OP-014
@pytest.mark.parametrize(
    "value",
    [SINGLETON_VALUE, SINGLETON_LAYOUT.pack(3, 42, 0.5)],
    ids=["mapping", "bytes"],
)
def test_one_element_pod_arrays_are_one_element_tuples(value):
    # Key setup: Send a raw-POD value whose arrays have one element between Python Operators.
    #
    #   emitter[Python] --holoscan.test.Singleton--> recorder[Python]
    reset()
    SingletonEmitter.value = value
    graph = hs.Graph("pod-one-element-arrays", default_clock=hs.RealtimeClock)
    emitter = graph.op(SingletonEmitter, "emitter")
    recorder = graph.op(SingletonRecorder, "recorder")
    graph.add_flow(emitter.out, recorder["in"])

    session = hs.run(hs.compile(graph))

    # Key assertion: One-element arrays arrive as one-element tuples, unlike the scalar field.
    assert session.diagnostics == []
    assert SingletonEmitter.error is None
    assert SingletonRecorder.received == [SINGLETON_VALUE]


# REQ: PY-OP-014
@pytest.mark.parametrize(
    ("value", "token", "detail"),
    [
        ({**SINGLETON_VALUE, "ids": 42}, "AUTHORING_POD_FIELD_LENGTH_MISMATCH", "'ids'"),
        ({**SINGLETON_VALUE, "gains": 0.5}, "AUTHORING_POD_FIELD_LENGTH_MISMATCH", "'gains'"),
        ({**SINGLETON_VALUE, "count": (3,)}, "AUTHORING_PAYLOAD_VALUE_TYPE_MISMATCH", "'count'"),
    ],
    ids=["scalar-for-c-array", "scalar-for-std-array", "tuple-for-scalar"],
)
def test_one_element_pod_arrays_keep_their_shape_on_emit(value, token, detail):
    # Key setup: Emit a scalar for a one-element array field, or a tuple for the scalar field.
    #
    #   emitter[Python] --holoscan.test.Singleton--> recorder[Python]
    reset()
    SingletonEmitter.value = value
    graph = hs.Graph("pod-one-element-array-shape", default_clock=hs.RealtimeClock)
    emitter = graph.op(SingletonEmitter, "emitter")
    recorder = graph.op(SingletonRecorder, "recorder")
    graph.add_flow(emitter.out, recorder["in"])

    session = hs.run(hs.compile(graph))

    # Key assertion: The field's declared shape decides, and nothing reaches the consumer.
    assert session.diagnostics == []
    assert SingletonEmitter.error is not None
    assert SingletonEmitter.error.token == token
    assert detail in str(SingletonEmitter.error)
    assert SingletonRecorder.received == []


# REQ: PY-OP-014
def test_pod_payload_emit_releases_a_rejected_buffer():
    # Key setup: Emit a bytearray one byte short of the native object.
    #
    #   emitter[Python] --holoscan.test.Pose--> sink[C++]
    reset()
    value = bytearray(native_pose(EXPECTED)[:-1])

    session = run_pose_to_cpp(value)

    # Key assertion: The emit is rejected and the bytearray's buffer export has been released.
    assert session.diagnostics == []
    assert PoseEmitter.error is not None
    assert PoseEmitter.error.token == "AUTHORING_POD_PAYLOAD_SIZE_MISMATCH"
    value.append(0)
    assert holoscan_test_support._last_test_pose() is None


# REQ: PY-OP-014
def test_pod_payload_emit_survives_an_element_that_empties_its_list():
    # Key setup: Emit a mapping whose array field is a list whose first element empties the list
    # while it is converted.
    #
    #   emitter[Python] --holoscan.test.Pose--> sink[C++]
    reset()
    position = [ListClearingInt(1), -2.25, 3.0]
    ListClearingInt.target = position

    session = run_pose_to_cpp({**EXPECTED, "position": position})

    # Key assertion: The next element read fails with IndexError instead of reading freed memory,
    # and nothing is sent.
    assert session.diagnostics == []
    assert isinstance(PoseEmitter.error, IndexError)
    assert holoscan_test_support._last_test_pose() is None


# REQ: PY-OP-014
@pytest.mark.parametrize(
    "error",
    [RuntimeError("length lookup failed"), hs.errors.NotReadyError("length lookup failed")],
    ids=["python-error", "holoscan-error"],
)
def test_pod_payload_emit_keeps_an_array_length_lookup_error(error):
    # Key setup: Emit a mapping whose array field is a sequence with a raising __len__.
    #
    #   emitter[Python] --holoscan.test.Pose--> sink[C++]
    reset()
    FailingLengthTuple.error = error
    value = {**EXPECTED, "position": FailingLengthTuple(EXPECTED["position"])}

    session = run_pose_to_cpp(value)

    # Key assertion: emit() raises the original exception, typed code included, rather than a
    # length mismatch, and nothing is sent.
    assert session.diagnostics == []
    assert PoseEmitter.error is error
    assert holoscan_test_support._last_test_pose() is None


@pytest.mark.parametrize("byte_input", [bytes, bytearray, list, tuple])
@pytest.mark.parametrize("read", ["receive", "receive_data", "selected_sample"])
# REQ: PY-OP-017
def test_pod_opaque_bytes_preserve_numeric_array_semantics(byte_input, read):
    payload_type = hs.payload.resolve("holoscan.test.ByteRecord")
    expected = {
        "tag": 255,
        "singleton": b"\x80",
        "cbytes": b"\x00\x7f\xff",
        "bytes": b"\xff\x80\x00",
        "numeric": (0, 255),
        "value": 128,
    }
    retained = []

    class Source(hs.Operator):
        """Publish explicit std::byte fields alongside ordinary numeric uint8 fields."""

        out = hs.Output("out", payload_type)

        def contract(self):
            return hs.Contract().trigger(hs.OnClock(period=hs.time.ms(1))).complete_after(1)

        def compute(self, context):
            fields = dict(expected)
            for name in ("singleton", "cbytes", "bytes"):
                fields[name] = byte_input(fields[name])
            invalid = (
                ("tag", 256, "AUTHORING_PAYLOAD_VALUE_OUT_OF_RANGE"),
                ("bytes", [-1, 0, 0], "AUTHORING_PAYLOAD_VALUE_OUT_OF_RANGE"),
                ("bytes", [True, 0, 0], "AUTHORING_PAYLOAD_VALUE_TYPE_MISMATCH"),
                ("bytes", b"xx", "AUTHORING_POD_FIELD_LENGTH_MISMATCH"),
                ("bytes", bytearray(4), "AUTHORING_POD_FIELD_LENGTH_MISMATCH"),
            )
            for name, bad, token in invalid:
                with pytest.raises(hs.errors.InvalidArgumentError) as error:
                    self.out.emit({**fields, name: bad})
                assert error.value.token == token
            self.out.emit(fields)
            if byte_input is bytearray:
                fields["bytes"].clear()

    class Sink(hs.Operator):
        """Retain the result to check its independent lifetime and exact Python field types."""

        inp = hs.Input("in", payload_type)

        def contract(self):
            return hs.Contract().trigger(hs.OnEach(self.inp))

        def compute(self, context):
            value = getattr(self.inp, read)()
            retained.append(value if read == "receive_data" else value.data)
            context.request_completion()

    graph = hs.Graph("pod-opaque-bytes", default_clock=hs.RealtimeClock)
    source = graph.op(Source, "source")
    sink = graph.op(Sink, "sink")
    graph.add_flow(source.out, sink["in"])
    session = hs.run(hs.compile(graph))
    assert session.diagnostics == []
    assert retained == [expected]
    assert isinstance(retained[0]["numeric"], tuple)
    assert isinstance(retained[0]["singleton"], bytes)
