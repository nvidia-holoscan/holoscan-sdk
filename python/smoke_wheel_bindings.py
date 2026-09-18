# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualify installed stable-ABI bindings without running GPU work."""

import argparse
import importlib
import importlib.util
import sys
import sysconfig
from pathlib import Path


def check_loaded_runtime(package, library_names, maps_file=Path("/proc/self/maps"), directory=None):
    """Require Linux runtime libraries to load once from the wheel that owns them.

    ``directory`` names that owner when it is not this package's own ``lib``, as for
    libraries supplied by a sibling wheel.
    """
    mapped_paths = set()
    for line in maps_file.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            mapped_paths.add(Path(fields[5]).resolve())
    library_directory = (package / "lib" if directory is None else directory).resolve()
    for name in library_names:
        # A repaired library is named <stem>-<hash>.so.<version>, so accept either
        # separator rather than only the unversioned SONAME.
        matches = {
            path
            for path in mapped_paths
            if path.name == name or path.name.startswith((f"{name}.", f"{name}-"))
        }
        assert len(matches) == 1, (name, "expected one loaded runtime", matches)
        location = matches.pop()
        assert location.parent == library_directory, (name, location, library_directory)
        print(f"{name}: {location}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-vision",
        action="store_true",
        help="Fail unless the Vision wheel is installed alongside Core.",
    )
    parser.add_argument(
        "--vision-import-order",
        choices=("canonical", "alias"),
        default="canonical",
        help=(
            "Which name imports Vision first: 'canonical' imports holoscan_vision, "
            "'alias' imports holoscan.vision. Each order has to work on its own, so "
            "the lane runs both to catch a path that only resolves once the other "
            "name has already been imported."
        ),
    )
    args = parser.parse_args()
    assert sys.implementation.name == "cpython"
    assert (3, 12) <= sys.version_info[:2] <= (3, 14)
    assert not sysconfig.get_config_var("Py_GIL_DISABLED"), "Regular CPython is required"

    # Check the interpreter before importing native code.
    np = importlib.import_module("numpy")
    holoscan = importlib.import_module("holoscan")
    core = importlib.import_module("holoscan.core")
    _core = importlib.import_module("holoscan.core._core")

    assert _core.__file__.endswith("/_core.abi3.so")
    source = np.arange(6, dtype=np.float32).reshape(2, 3)
    tensor = core.Tensor.as_tensor(source)
    assert tuple(tensor.shape) == source.shape
    np.testing.assert_array_equal(np.from_dlpack(tensor), source)
    package = Path(holoscan.__file__).parent
    # An untyped wheel from HOLOSCAN_BUILD_PYTHON_STUBS=OFF ships no py.typed; when it
    # is present the advertised stubs must be complete.
    if (package / "py.typed").is_file():
        for name in ("__init__.pyi", "core/__init__.pyi", "core/_core.pyi"):
            assert (package / name).is_file(), name
    check_loaded_runtime(
        package,
        (
            "libholoscan_core.so",
            "libholoscan_logger.so",
            "libholomq.so",
            "libiceoryx2_cxx.so",
            "libiceoryx2_ffi_c.so",
        ),
    )

    vision_available = importlib.util.find_spec("holoscan_vision") is not None
    if args.require_vision:
        assert vision_available, "Vision package is required in this qualification lane"
    if vision_available:
        first = "holoscan_vision" if args.vision_import_order == "canonical" else "holoscan.vision"
        importlib.import_module(first)
        holoscan_vision = importlib.import_module("holoscan_vision")
        _vision = importlib.import_module("holoscan_vision._vision")
        from holoscan import vision  # noqa: PLC0415 - Check the selected import order first.

        assert vision is holoscan.vision is holoscan_vision
        assert importlib.import_module("holoscan.vision._vision") is _vision
        assert _vision.__spec__.name == "holoscan_vision._vision"

        assert _vision.__file__.endswith("/_vision.abi3.so")
        params = holoscan_vision.ResizeOpParams()
        assert params.interpolation == holoscan_vision.InterpolationMode.LINEAR
        _vision._check_native_linkage()
        package = Path(holoscan_vision.__file__).parent
        # Same posture as Core: an untyped wheel ships no py.typed, and when the marker
        # is present the advertised stubs must be complete.
        if (package / "py.typed").is_file():
            for name in ("__init__.pyi", "_vision.pyi"):
                assert (package / name).is_file(), name
        check_loaded_runtime(package, ("libholoscan_vision.so",))
        # CV-CUDA comes from the cvcuda-cu13 wheel, which auditwheel repaired: its
        # libraries carry content-derived names in a sibling <distribution>.libs.
        check_loaded_runtime(
            package,
            ("libcvcuda", "libnvcv_types"),
            directory=package.parent / "cvcuda_cu13.libs",
        )
    print(f"Qualified Core{' + Vision' if vision_available else ''} on {sys.version}")


if __name__ == "__main__":
    main()
