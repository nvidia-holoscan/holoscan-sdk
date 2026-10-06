# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Holoscan Vision provider typing-artifact tests."""

from pathlib import Path

from holoscan_vision import operators as vision_ops


# REQ: PY-API-018
def test_operator_stub_surface_is_module_owned():
    # Key setup: Read the source-owned stub copied only in typing-enabled builds.
    package_directory = Path(vision_ops.__file__).parent
    assert (package_directory.parent / "py.typed").is_file()
    stub_text = (package_directory / "__init__.pyi").read_text(encoding="utf-8")

    # Key assertion: Every module-owned proxy and its refined Resize contract are present.
    for alias in vision_ops.__all__:
        assert f"{alias}: Final[_{alias}Type]" in stub_text
    assert "class _ResizeOpParams(TypedDict, total=False):" in stub_text
    assert "max_batch_size: Required[int]" in stub_text
    resize_fields = stub_text.split("class _ResizeOpParams", 1)[1].split("class ", 1)[0]
    assert "max_width" not in resize_fields
    assert "interpolation: Literal['nearest', 0, 'linear', 1, 'cubic', 2, 'area', 3]" in stub_text
    assert "class _BayerDemosaicOpParams(TypedDict, total=False):" in stub_text
    assert "bayer_grid_pos: Literal['bggr', 0, 'rggb', 1, 'gbrg', 2, 'grbg', 3]" in stub_text
