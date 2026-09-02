# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = Path(__file__).with_name("gxf_entity_codec.py")
SPEC = importlib.util.spec_from_file_location("gxf_entity_codec", SCRIPT_PATH)
assert SPEC
assert SPEC.loader
codec = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = codec
SPEC.loader.exec_module(codec)


def make_tensor_header(
    *,
    storage_type=codec.MemoryStorageType.kHost.value,
    element_type=codec.PrimitiveType.kUnsigned8.value,
    bytes_per_element=1,
    rank=1,
    dims=(1,),
    strides=(1,),
):
    padded_dims = (*dims, *((1,) * (codec.Shape.kMaxRank - len(dims))))
    padded_strides = (*strides, *((0,) * (codec.Shape.kMaxRank - len(strides))))
    return codec.TensorHeader.HEADER_STRUCT.pack(
        storage_type,
        element_type,
        bytes_per_element,
        rank,
        *padded_dims,
        *padded_strides,
    )


def make_entity(tensor_header=None, payload=b"\0", *, name=b""):
    if tensor_header is None:
        tensor_header = make_tensor_header()
    entity_header = codec.EntityHeader.HEADER_STRUCT.pack(0, 0, 0, 0, 1, 0)
    component_header = codec.ComponentHeader.HEADER_STRUCT.pack(0, *codec.TensorType, len(name))
    return entity_header + component_header + name + tensor_header + payload


def write_recording(tmp_path, entity_data, *, data_size=None):
    basename = "recording"
    if data_size is None:
        data_size = len(entity_data)
    (tmp_path / f"{basename}.gxf_index").write_bytes(
        codec.EntityIndex.HEADER_STRUCT.pack(1, data_size, 0)
    )
    (tmp_path / f"{basename}.gxf_entities").write_bytes(entity_data)
    return basename


@pytest.mark.parametrize(
    ("header_type", "description"),
    [
        (codec.EntityIndex, "entity index"),
        (codec.EntityHeader, "entity header"),
        (codec.ComponentHeader, "component header"),
        (codec.TensorHeader, "tensor header"),
    ],
)
def test_fixed_size_headers_require_exact_reads(header_type, description):
    with pytest.raises(ValueError, match=rf"Truncated GXF {description}"):
        header_type(buffer=b"\0" * (header_type.HEADER_SIZE - 1))


def test_oversized_component_name_is_rejected_before_reading(tmp_path):
    entity_header = codec.EntityHeader.HEADER_STRUCT.pack(0, 0, 0, 0, 1, 0)
    component_header = codec.ComponentHeader.HEADER_STRUCT.pack(0, *codec.TensorType, 2**63)
    entity_data = entity_header + component_header + make_tensor_header()
    basename = write_recording(tmp_path, entity_data)

    with (
        codec.EntityReader(tmp_path, basename) as reader,
        pytest.raises(ValueError, match="Truncated GXF component name"),
    ):
        reader.get_entity(0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"rank": 0, "dims": (), "strides": ()}, "rank 0"),
        ({"rank": 9}, "rank 9"),
        ({"storage_type": 99}, "storage type: 99"),
        ({"element_type": 99}, "element type: 99"),
        ({"bytes_per_element": 0}, "outside the allowed range"),
        ({"bytes_per_element": 2}, "does not match kUnsigned8"),
        ({"dims": (-1,)}, "dimensions must be positive"),
        ({"dims": (2**30 + 1,)}, "payload exceeds"),
    ],
)
def test_invalid_tensor_headers_raise_bounded_value_errors(overrides, message):
    with pytest.raises(ValueError, match=message):
        codec.TensorHeader(buffer=make_tensor_header(**overrides))


@pytest.mark.parametrize(
    ("element_type", "bytes_per_element", "dtype"),
    [
        (codec.PrimitiveType.kFloat16, 2, np.float16),
        (codec.PrimitiveType.kComplex64, 8, np.complex64),
        (codec.PrimitiveType.kComplex128, 16, np.complex128),
    ],
)
def test_supported_extended_tensor_enums_are_accepted(element_type, bytes_per_element, dtype):
    header = codec.TensorHeader(
        buffer=make_tensor_header(
            storage_type=codec.MemoryStorageType.kCudaManaged.value,
            element_type=element_type.value,
            bytes_per_element=bytes_per_element,
            strides=(bytes_per_element,),
        )
    )

    assert header.storage_type is codec.MemoryStorageType.kCudaManaged
    assert header.element_type is element_type
    assert header.dtype == np.dtype(dtype)


def test_custom_tensor_type_is_rejected_before_payload_read():
    header = make_tensor_header(
        element_type=codec.PrimitiveType.kCustom.value,
        bytes_per_element=codec.MAX_DESERIALIZED_BYTES_PER_ELEMENT,
        dims=(1024,),
        strides=(16,),
    )

    with pytest.raises(ValueError, match="Custom GXF tensor element types"):
        codec.Tensor(buffer=header)


def test_tensor_payload_requires_exact_read():
    with pytest.raises(ValueError, match="Truncated GXF tensor payload"):
        codec.Tensor(buffer=make_tensor_header(dims=(4,), strides=(1,)) + b"123")


def test_tensor_strides_must_stay_within_payload():
    header = make_tensor_header(rank=2, dims=(2, 2), strides=(100, 1))

    with pytest.raises(ValueError, match="strides address .* beyond"):
        codec.Tensor(buffer=header + b"1234")


def test_index_record_must_be_complete(tmp_path):
    basename = "recording"
    (tmp_path / f"{basename}.gxf_index").write_bytes(b"\0")
    (tmp_path / f"{basename}.gxf_entities").write_bytes(b"")

    with (
        codec.EntityReader(tmp_path, basename) as reader,
        pytest.raises(ValueError, match="not a multiple"),
    ):
        _ = reader.num_entities


def test_index_data_range_must_fit_entity_file(tmp_path):
    basename = write_recording(tmp_path, b"", data_size=2**63)

    with (
        codec.EntityReader(tmp_path, basename) as reader,
        pytest.raises(ValueError, match="exceeds entity file size"),
    ):
        reader.get_entity(0)


def test_entity_data_must_be_consumed_exactly(tmp_path):
    entity_data = make_entity() + b"trailing"
    basename = write_recording(tmp_path, entity_data)

    with (
        codec.EntityReader(tmp_path, basename) as reader,
        pytest.raises(ValueError, match="unconsumed bytes"),
    ):
        reader.get_entity(0)


@pytest.mark.parametrize("timestamps", [(10, 10), (11, 10)])
def test_get_framerate_rejects_non_increasing_duration(tmp_path, timestamps):
    basename = "recording"
    index_data = b"".join(
        codec.EntityIndex.HEADER_STRUCT.pack(timestamp, 0, 0) for timestamp in timestamps
    )
    (tmp_path / f"{basename}.gxf_index").write_bytes(index_data)
    (tmp_path / f"{basename}.gxf_entities").write_bytes(b"")

    with (
        codec.EntityReader(tmp_path, basename) as reader,
        pytest.raises(ValueError, match="non-increasing"),
    ):
        reader.get_framerate()


def test_writer_recording_round_trip(tmp_path):
    frames = [
        np.arange(12, dtype=np.uint8).reshape(2, 2, 3),
        np.arange(12, 24, dtype=np.uint8).reshape(2, 2, 3),
    ]
    with codec.EntityWriter(tmp_path, "recording", framerate=30) as writer:
        for frame in frames:
            writer.add(frame)

    with codec.EntityReader(tmp_path, "recording") as reader:
        assert reader.num_entities == len(frames)
        assert reader.get_framerate() > 0
        for index, expected in enumerate(frames):
            np.testing.assert_array_equal(reader.get_frame(index), expected)


def test_raw_video_converters_round_trip(tmp_path):
    input_frames = bytes(range(24))
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH.with_name("convert_video_to_gxf_entities.py")),
            "--width",
            "2",
            "--height",
            "2",
            "--channels",
            "3",
            "--directory",
            str(tmp_path),
            "--basename",
            "recording",
        ],
        input=input_frames,
        check=True,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH.with_name("convert_gxf_entities_to_video.py")),
            "--directory",
            str(tmp_path),
            "--basename",
            "recording",
        ],
        check=True,
        capture_output=True,
    )
    assert result.stdout == input_frames
