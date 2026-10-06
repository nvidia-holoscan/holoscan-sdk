# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise namespace aliases with real distribution metadata and inert modules."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(params=[False, True], ids=["same-prefix", "separate-prefix"])
def environment(tmp_path, request):
    source = Path(__file__).parents[1] / "holoscan"
    # Separate prefixes exercise Python discovery only, not ELF loader behavior.
    core_site = tmp_path / "core-site"
    module_site = tmp_path / "module-site" if request.param else core_site
    package = core_site / "holoscan"
    package.mkdir(parents=True)
    module_site.mkdir(exist_ok=True)
    for name in (
        "__init__.py",
        "_module_discovery.py",
        "_public_api.py",
        "clocks.py",
        "config.py",
        "diagnostics.py",
        "errors.py",
        "execution.py",
        "lifecycle.py",
        "params.py",
        "registry.py",
        "tensor.py",
        "time.py",
        "topics.py",
    ):
        shutil.copyfile(source / name, package / name)
    operators = package / "operators"
    operators.mkdir()
    shutil.copyfile(source / "operators/__init__.py", operators / "__init__.py")
    graphs = package / "graphs"
    graphs.mkdir()
    shutil.copyfile(source / "graphs/__init__.py", graphs / "__init__.py")
    channel = package / "channel"
    channel.mkdir()
    shutil.copyfile(source / "channel/__init__.py", channel / "__init__.py")
    shutil.copyfile(source / "channel/aio.py", channel / "aio.py")
    payload = package / "payload"
    payload.mkdir()
    shutil.copyfile(source / "payload/__init__.py", payload / "__init__.py")
    shutil.copyfile(source / "payload/schema.py", payload / "schema.py")
    schema = package / "schema"
    schema.mkdir()
    (schema / "__init__.py").write_text(
        "class _Value: pass\n"
        "Header = String = TensorPacket = Time = _Value\n"
        "class geometry: pass\n"
        "__all__ = ['Header', 'String', 'TensorPacket', 'Time', 'geometry']\n"
    )
    (package / "_core.py").write_text(
        "class _Value: pass\n"
        "class Tensor: pass\n"
        "class _ChannelTensorLoan: pass\n"
        "DLDataType = DLDataTypeCode = DLDevice = DLDeviceType = Tensor\n"
        "def is_tensor_like(value): return isinstance(value, Tensor)\n"
        "def _raise_channel_async_receive_timeout(): raise TimeoutError\n"
        "def __getattr__(name):\n"
        "    from ._public_api import PUBLIC_API\n"
        "    if any(name in exports for exports in PUBLIC_API.values()):\n"
        "        return _Value\n"
        "    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')\n"
    )
    example = module_site / "holoscan_example"
    example.mkdir()
    (example / "__init__.py").write_text("from ._example import Value\n")
    (example / "_example.py").write_text("class Value: pass\n")
    metadata = module_site / "holoscan_example_cu13-5.0.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: holoscan-example-cu13\nVersion: 5.0.0\n"
    )
    (metadata / "entry_points.txt").write_text("[holoscan.modules]\nexample = holoscan_example\n")
    return core_site, module_site, metadata


def run(environment, code):
    core_site, module_site, _ = environment
    # -I excludes ambient PYTHONPATH and user site; only explicit fixture paths are added.
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"import sys; sys.path[:0] = {[str(core_site), str(module_site)]!r}\n" + code,
        ],
        env={**os.environ, "PYTHONWARNINGS": "default"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_fake_core_rejects_unknown_exports(environment):
    run(
        environment,
        """
import holoscan._core as core
assert not hasattr(core, 'MisspelledOrRemovedExport')
""",
    )


@pytest.mark.parametrize("first", ["holoscan", "holoscan_example", "holoscan.example"])
def test_aliases_preserve_identity_and_canonical_specs(environment, first):
    run(
        environment,
        f"""
import importlib
importlib.import_module({first!r})
import holoscan
import holoscan.example
from holoscan import example
import holoscan_example
assert example is holoscan.example is holoscan_example
assert example.__spec__.name == 'holoscan_example'
canonical = importlib.import_module('holoscan_example._example')
alias = importlib.import_module('holoscan.example._example')
assert canonical is alias
assert canonical.Value is example.Value
assert canonical.__spec__.name == 'holoscan_example._example'
assert holoscan.Tensor is holoscan._core.Tensor
""",
    )


def test_discovery_is_lazy_and_failed_import_does_not_break_core(environment):
    (_, module_site, _) = environment
    (module_site / "holoscan_example/__init__.py").write_text(
        "raise ImportError('missing native dependency')\n"
    )
    run(
        environment,
        """
import holoscan
assert 'example' in dir(holoscan)
assert 'holoscan_example' not in sys.modules
try:
    import holoscan.example
except ImportError as exc:
    assert 'missing native dependency' in str(exc)
else:
    raise AssertionError('Missing import error')
assert 'holoscan.example' not in sys.modules
assert holoscan.Tensor is holoscan._core.Tensor
""",
    )


@pytest.mark.parametrize(
    "entry",
    [
        "core = holoscan_example",
        "tensor = holoscan_example",
        "_private = holoscan_example",
        "bad-name = holoscan_example",
        "example = holoscan.example",
        "example = holoscan_example:Value",
    ],
)
def test_invalid_registration_is_not_loaded(environment, entry):
    environment[2].joinpath("entry_points.txt").write_text(f"[holoscan.modules]\n{entry}\n")
    result = run(
        environment,
        """
import holoscan
from holoscan import tensor
assert tensor is holoscan.tensor
assert holoscan.Tensor is holoscan._core.Tensor
assert 'holoscan_example' not in sys.modules
""",
    )
    assert "RuntimeWarning" in result.stderr


@pytest.mark.parametrize("name", ["core", "example"])
def test_rejected_registration_is_an_absent_attribute(environment, name):
    environment[2].joinpath("entry_points.txt").write_text(
        f"[holoscan.modules]\n{name} = holoscan_example:Value\n"
    )
    result = run(
        environment,
        f"""
import holoscan
assert not hasattr(holoscan, {name!r})
sentinel = object()
assert getattr(holoscan, {name!r}, sentinel) is sentinel
assert 'holoscan._core' in sys.modules
assert 'holoscan_example' not in sys.modules
""",
    )
    assert "RuntimeWarning" in result.stderr


def test_duplicates_are_rejected_instead_of_first_wins(environment):
    _, module_site, metadata = environment
    other = module_site / "other-1.dist-info"
    shutil.copytree(metadata, other)
    other.joinpath("METADATA").write_text("Metadata-Version: 2.4\nName: other\nVersion: 1\n")
    result = run(environment, "import holoscan; assert 'example' not in dir(holoscan)")
    assert "Conflicting" in result.stderr


def test_direct_extension_import_can_initialize_core(environment):
    (environment[1] / "holoscan_example/__init__.py").write_text(
        "from holoscan import Tensor\nfrom ._example import Value\n"
    )
    run(
        environment,
        """
import holoscan_example
from holoscan import example
from holoscan import Tensor
assert example is holoscan_example
assert example.Tensor is Tensor
""",
    )


def test_core_import_does_not_load_optional_native_packages(environment):
    run(
        environment,
        """
import holoscan
assert 'Tensor' in holoscan.__all__
assert 'holoscan.core' not in sys.modules
assert 'holoscan_example' not in sys.modules
assert 'holoscan._core' in sys.modules
""",
    )


def test_attribute_access_and_reload(environment):
    run(
        environment,
        """
import importlib
import holoscan
before = len(sys.meta_path)
importlib.reload(holoscan)
assert len(sys.meta_path) == before
assert holoscan.example is importlib.import_module('holoscan_example')
importlib.reload(holoscan.example)
assert holoscan.example.__spec__.name == 'holoscan_example'
""",
    )


@pytest.mark.parametrize("statement", ["import holoscan.example", "from holoscan import example"])
def test_existing_subpackage_is_not_shadowed(environment, statement):
    (environment[0] / "holoscan/example").mkdir()
    (environment[0] / "holoscan/example/__init__.py").write_text("MARKER = 'existing'\n")
    result = run(
        environment,
        f"import holoscan; {statement}\n"
        "assert holoscan.example.MARKER == 'existing'\n"
        "assert 'holoscan_example' not in sys.modules\n",
    )
    assert "shadow" in result.stderr


def test_metadata_discovery_failure_keeps_core_importable(environment):
    result = run(
        environment,
        """
from unittest.mock import patch
with patch('importlib.metadata.entry_points', side_effect=OSError('metadata unavailable')):
    import holoscan
assert holoscan.Tensor is not None
""",
    )
    assert "Unable to discover" in result.stderr


def test_circular_alias_import_is_rejected_without_stale_modules(environment):
    (environment[1] / "holoscan_example/__init__.py").write_text(
        "import holoscan.example\nfrom ._example import Value\n"
    )
    run(
        environment,
        """
try:
    import holoscan_example
except ImportError as exc:
    assert 'Circular Holoscan module alias import' in str(exc)
else:
    raise AssertionError('Circular alias initialization must fail explicitly')
assert 'holoscan_example' not in sys.modules
assert 'holoscan.example' not in sys.modules
""",
    )


def test_missing_registered_package_reports_import_error(environment):
    shutil.rmtree(environment[1] / "holoscan_example")
    run(
        environment,
        """
import holoscan
try:
    import holoscan.example
except ModuleNotFoundError as exc:
    assert exc.name == 'holoscan_example'
else:
    raise AssertionError('Missing package must fail explicitly')
assert 'holoscan.example' not in sys.modules
""",
    )


def test_alias_binds_attribute_after_first_load(environment):
    run(
        environment,
        """
import holoscan
assert 'example' not in vars(holoscan)
holoscan.example
assert vars(holoscan)['example'] is sys.modules['holoscan_example']
""",
    )


def test_failing_provider_is_imported_once_and_reports_cached_failure(environment):
    marker = environment[1] / "attempts.log"
    (environment[1] / "holoscan_example/__init__.py").write_text(
        f"open({str(marker)!r}, 'a').write('x')\nraise RuntimeError('provider is broken')\n"
    )
    run(
        environment,
        """
import holoscan
messages = []
for _ in range(3):
    try:
        holoscan.example
    except ModuleNotFoundError as exc:
        messages.append(str(exc))
    else:
        raise AssertionError('Broken provider must fail explicitly')
assert len(messages) == 3, messages
assert all('provider is broken' in message for message in messages), messages
""",
    )
    # A cached failure must not re-run a broken provider for every caller.
    assert marker.read_text() == "x"


def test_entry_point_target_must_be_a_package(environment):
    shutil.rmtree(environment[1] / "holoscan_example")
    (environment[1] / "holoscan_example.py").write_text("VALUE = 1\n")
    run(
        environment,
        """
import holoscan
try:
    holoscan.example
except ModuleNotFoundError as exc:
    assert 'package' in str(exc), exc
else:
    raise AssertionError('Non-package target must fail explicitly')
""",
    )


def test_concurrent_alias_imports_resolve_to_one_module(environment):
    run(
        environment,
        """
import threading
import holoscan
loaded, errors = [], []
def worker():
    try:
        loaded.append(holoscan.example)
    except Exception as exc:  # noqa: BLE001 - surfaced through the assertion below
        errors.append(exc)
threads = [threading.Thread(target=worker) for _ in range(8)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
assert not errors, errors
assert len(loaded) == 8
assert len({id(module) for module in loaded}) == 1
assert loaded[0] is sys.modules['holoscan_example']
""",
    )
