# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The optional extension rejects incompatible Core services before using their callbacks."""

import subprocess
import sys
import textwrap

import pytest

_CAPSULE_SETUP = """
import ctypes
import holoscan._core as core

class Prefix(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_uint32),
        ("size", ctypes.c_size_t),
        ("sdk_version", ctypes.c_char_p),
        ("compatibility_id", ctypes.c_char_p),
    ]

get_name = ctypes.pythonapi.PyCapsule_GetName
get_name.argtypes = [ctypes.py_object]
get_name.restype = ctypes.c_char_p
get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
get_pointer.restype = ctypes.c_void_p
new_capsule = ctypes.pythonapi.PyCapsule_New
new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
new_capsule.restype = ctypes.py_object
name = get_name(core._binding_services)
address = get_pointer(core._binding_services, name)
original = Prefix.from_address(address)
assert original.size >= ctypes.sizeof(Prefix)
# Keep all real callbacks and the complete table allocation intact. Only the checked prefix
# changes, and the buffer plus strings stay alive until the subprocess finishes.
storage = ctypes.create_string_buffer(ctypes.string_at(address, original.size), original.size)
prefix = Prefix.from_buffer(storage)
"""


def run_import_script(source):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "mutation",
    [
        "prefix.compatibility_id = b'other-interface-build'",
        "prefix.compatibility_id = None",
        "prefix.sdk_version = b'0.0.0'",
        "prefix.sdk_version = None",
        "prefix.abi_version += 1",
        "prefix.size -= 1",
        "name = b'holoscan.python.BindingServices.incompatible'",
    ],
)
# REQ: PY-API-026
def test_incompatible_services_fail_during_import(mutation):
    run_import_script(
        _CAPSULE_SETUP
        + "\n"
        + mutation
        + """
core._binding_services = new_capsule(ctypes.addressof(storage), name, None)
try:
    import holoscan.testing
except ImportError as error:
    assert "matching Core binding services" in str(error), str(error)
else:
    raise AssertionError("incompatible Core services were accepted")
"""
    )


@pytest.mark.parametrize("name", ["CompileOptions", "DeviceId"])
# REQ: PY-API-026
def test_incompatible_canonical_type_fails_during_import(name):
    run_import_script(
        f"""
        import holoscan._core as core
        setattr(core, {name!r}, object)
        try:
            import holoscan.testing
        except ImportError as error:
            assert "canonical Core type registration" in str(error), str(error)
        else:
            raise AssertionError("incompatible canonical type was accepted")
        """
    )


# REQ: PY-API-026
def test_matching_services_import_with_canonical_types():
    run_import_script(
        """
        import holoscan as hs
        import holoscan._core as core
        import holoscan.testing as testing
        assert hs.CompileOptions is core.CompileOptions
        assert hs.execution.DeviceId is core.DeviceId
        assert testing.OperatorTestHarness.__module__ == "holoscan.testing"
        """
    )
