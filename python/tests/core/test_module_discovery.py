# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent Holoscan module discovery tests."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import holoscan as hs


def _write_package(site: Path, name: str, body: str, *, package: bool = True) -> None:
    """Write one importable test provider below an isolated site directory."""

    if package:
        package_root = site / name
        package_root.mkdir(parents=True)
        (package_root / "__init__.py").write_text(textwrap.dedent(body), encoding="utf-8")
    else:
        (site / f"{name}.py").write_text(textwrap.dedent(body), encoding="utf-8")


def _write_entry_points(site: Path, distribution: str, rows: list[str]) -> None:
    """Publish test-only ``holoscan.modules`` metadata."""

    metadata_root = site / f"{distribution}-1.0.dist-info"
    metadata_root.mkdir()
    (metadata_root / "entry_points.txt").write_text(
        "[holoscan.modules]\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )


def _run_isolated(site: Path, source: str) -> subprocess.CompletedProcess[str]:
    """Run one discovery scenario with only its temporary provider metadata added."""

    environment = os.environ.copy()
    python_paths = [str(site), str(Path(hs.__file__).resolve().parent.parent)]
    if inherited := environment.get("PYTHONPATH"):
        python_paths.append(inherited)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


@pytest.mark.parametrize("public_first", [False, True])
# REQ: PY-API-016
def test_provider_aliases_are_lazy_and_identity_preserving(tmp_path, public_first):
    # Key setup: Publish a provider package and a real module-discovery entry point.
    site = tmp_path / "site"
    site.mkdir()
    _write_package(
        site,
        "hsb_provider",
        """
        from pathlib import Path

        marker = Path(__file__).with_name("loads")
        marker.write_text(str(int(marker.read_text()) + 1) if marker.exists() else "1")
        from . import operators
        """,
    )
    _write_package(
        site / "hsb_provider",
        "operators",
        """
        class LinuxReceiverOp:
            pass
        """,
    )
    _write_entry_points(site, "hsb_provider", ["hsb = hsb_provider"])
    completed = _run_isolated(
        site,
        f"""
        import importlib
        import sys
        import holoscan

        assert "hsb_provider" not in sys.modules
        first, second = (
            ("holoscan.hsb.operators", "hsb_provider.operators")
            if {public_first!r}
            else ("hsb_provider.operators", "holoscan.hsb.operators")
        )
        importlib.import_module(first)
        importlib.import_module(second)
        import hsb_provider
        assert holoscan.hsb is hsb_provider
        assert holoscan.hsb.operators is hsb_provider.operators
        assert "LinuxReceiverOp" not in holoscan.operators.__all__
        assert not hasattr(holoscan.operators, "LinuxReceiverOp")
        assert open(hsb_provider.__file__.replace("__init__.py", "loads")).read() == "1"
        """,
    )

    # Key assertion: Either import order initializes once and preserves root and child identity.
    assert completed.returncode == 0, completed.stderr


# REQ: PY-API-016
def test_invalid_reserved_duplicate_and_attribute_entries_fail_closed(tmp_path):
    # Key setup: Advertise malformed, reserved, duplicate, and attribute-valued providers.
    site = tmp_path / "site"
    site.mkdir()
    _write_package(site, "valid_provider", "value = object()")
    _write_entry_points(
        site,
        "invalid_providers",
        [
            "bad-name = valid_provider",
            "channel = valid_provider",
            "operators = valid_provider",
            "attribute = valid_provider:value",
            "duplicate = valid_provider",
        ],
    )
    _write_entry_points(site, "duplicate_provider", ["duplicate = valid_provider"])
    completed = _run_isolated(
        site,
        """
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            import holoscan

        messages = [str(item.message) for item in caught]
        assert any("bad-name" in message for message in messages)
        assert any("channel" in message and "reserved" in message for message in messages)
        assert any("operators" in message and "reserved" in message for message in messages)
        assert any("attribute" in message and "not an attribute" in message for message in messages)
        assert any(
            "duplicate" in message and "multiple providers" in message for message in messages
        )
        assert holoscan.channel.__name__ == "holoscan.channel"
        assert holoscan.operators.__name__ == "holoscan.operators"
        for name in ("bad-name", "attribute", "duplicate"):
            try:
                getattr(holoscan, name)
            except AttributeError:
                pass
            else:
                raise AssertionError(f"rejected provider {name!r} became visible")
        """,
    )

    # Key assertion: Core still imports, warns actionably, and publishes no arbitrary winner.
    assert completed.returncode == 0, completed.stderr


# REQ: PY-API-016
def test_provider_import_failures_and_cycles_are_isolated(tmp_path):
    # Key setup: Publish a failing provider, a non-package module, and a recursive provider.
    site = tmp_path / "site"
    site.mkdir()
    _write_package(site, "broken_provider", 'raise RuntimeError("broken sentinel")')
    _write_package(site, "plain_provider", "value = 1", package=False)
    _write_package(site, "circular_provider", "from holoscan import circular")
    _write_entry_points(
        site,
        "failing_providers",
        [
            "broken = broken_provider",
            "plain = plain_provider",
            "circular = circular_provider",
        ],
    )
    completed = _run_isolated(
        site,
        """
        import holoscan

        expected = {
            "broken": "broken sentinel",
            "plain": "module package",
            "circular": "circular import",
        }
        for name, detail in expected.items():
            try:
                getattr(holoscan, name)
            except ModuleNotFoundError as error:
                assert detail in str(error), (name, error)
            else:
                raise AssertionError(f"broken provider {name!r} loaded")
        assert holoscan.Graph is not None
        """,
    )

    # Key assertion: Each failure is actionable without breaking the Core API.
    assert completed.returncode == 0, completed.stderr


# REQ: PY-API-016
def test_concurrent_alias_imports_initialize_one_provider(tmp_path):
    # Key setup: Request one registered alias concurrently from several threads.
    site = tmp_path / "site"
    site.mkdir()
    _write_package(
        site,
        "thread_provider",
        """
        from pathlib import Path

        marker = Path(__file__).with_name("loads")
        marker.write_text(str(int(marker.read_text()) + 1) if marker.exists() else "1")
        """,
    )
    _write_entry_points(site, "thread_provider", ["threaded = thread_provider"])
    completed = _run_isolated(
        site,
        """
        import importlib
        from concurrent.futures import ThreadPoolExecutor

        import holoscan

        with ThreadPoolExecutor(max_workers=8) as executor:
            modules = list(
                executor.map(lambda _: importlib.import_module("holoscan.threaded"), range(32))
            )
        assert all(module is modules[0] for module in modules)
        assert modules[0] is importlib.import_module("thread_provider")
        assert open(modules[0].__file__.replace("__init__.py", "loads")).read() == "1"
        """,
    )

    # Key assertion: Import locking and provider state prevent duplicate initialization.
    assert completed.returncode == 0, completed.stderr


# REQ: PY-API-016
def test_refresh_seam_recovers_new_metadata_and_reports_metadata_failure(tmp_path):
    # Key setup: Import Core before adding metadata, then refresh its private discovery index.
    site = tmp_path / "site"
    site.mkdir()
    _write_package(site, "late_provider", "value = 7")
    _write_entry_points(site, "late_provider", ["late = late_provider"])
    completed = _run_isolated(
        site,
        f"""
        import sys
        import warnings

        site = {str(site)!r}
        while site in sys.path:
            sys.path.remove(site)
        import holoscan
        sys.path.insert(0, site)
        from holoscan import _module_discovery

        _module_discovery.refresh_for_tests()
        assert "late" in _module_discovery.registered_names()
        assert "late" not in vars(holoscan)
        # Refreshing an indexed provider before loading it must not delete an absent attribute.
        _module_discovery.refresh_for_tests()
        assert holoscan.late.value == 7

        original = _module_discovery._metadata_entry_points
        _module_discovery._metadata_entry_points = lambda: (_ for _ in ()).throw(
            RuntimeError("metadata sentinel")
        )
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _module_discovery.refresh_for_tests()
            assert any("metadata sentinel" in str(item.message) for item in caught)
        finally:
            _module_discovery._metadata_entry_points = original
        """,
    )

    # Key assertion: Tests can rebuild discovery state and metadata failures remain non-fatal.
    assert completed.returncode == 0, completed.stderr
