#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

SDK_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_DOCKERFILE = SDK_ROOT / "Dockerfile.dev"
POLICY_EXCLUSIONS = SDK_ROOT / "dev-container/scripts/source-pip-exclusions.txt"
SCRIPT = SDK_ROOT / "dev-container/scripts/fetch-pip-sources.py"
SPEC = importlib.util.spec_from_file_location("fetch_pip_sources", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pep503_normalizes_equivalent_package_names() -> None:
    assert MODULE.pep503("Example_Package.Name") == "example-package-name"
    assert MODULE.pep503("example---package") == "example-package"


def test_load_exclude_list_normalizes_bare_and_versioned_entries(tmp_path: Path) -> None:
    exclude_list = tmp_path / "exclude.txt"
    exclude_list.write_text("Example_Package.Name\nother.package==1.2.3\n", encoding="utf-8")

    bare, exact = MODULE.load_exclude_list(str(exclude_list))

    assert bare == {"example-package-name"}
    assert exact == {"other-package==1.2.3"}


def test_download_sdist_uses_a_timeout_for_metadata_and_artifact(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, int]] = []
    metadata = {
        "urls": [{"filename": "example-1.0.tar.gz", "packagetype": "sdist", "url": "sdist-url"}]
    }

    def urlopen(url: str, timeout: int) -> io.BytesIO:
        calls.append((url, timeout))
        if url == "sdist-url":
            return io.BytesIO(b"source archive")
        return io.BytesIO(json.dumps(metadata).encode())

    monkeypatch.setattr(MODULE.urllib.request, "urlopen", urlopen)

    assert MODULE.download_sdist("Example", "1.0", tmp_path)
    assert (tmp_path / "example-1.0.tar.gz").read_bytes() == b"source archive"
    assert calls == [
        ("https://pypi.org/pypi/example/1.0/json", MODULE.DOWNLOAD_TIMEOUT_SECONDS),
        ("sdist-url", MODULE.DOWNLOAD_TIMEOUT_SECONDS),
    ]


def test_download_sdist_rejects_missing_or_yanked_sources(monkeypatch, tmp_path: Path) -> None:
    metadata = {
        "urls": [{"filename": "example-1.0.tar.gz", "packagetype": "sdist", "yanked": True}]
    }
    monkeypatch.setattr(
        MODULE.urllib.request,
        "urlopen",
        lambda _url, **_kwargs: io.BytesIO(json.dumps(metadata).encode()),
    )

    assert not MODULE.download_sdist("Example", "1.0", tmp_path)


def test_main_fails_after_attempting_all_eligible_downloads(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.sys, "argv", [str(SCRIPT)])
    monkeypatch.setattr(
        MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout='[{"name": "first", "version": "1.0"}, {"name": "second", "version": "2.0"}]'
        ),
    )
    attempted: list[tuple[str, str]] = []

    def download_sdist(name: str, version: str, _output_dir: Path) -> bool:
        attempted.append((name, version))
        return name == "second"

    monkeypatch.setattr(MODULE, "download_sdist", download_sdist)

    assert MODULE.main() == 1
    assert attempted == [("first", "1.0"), ("second", "2.0")]


def test_pytorch_source_is_not_embedded() -> None:
    development_dockerfile = DEVELOPMENT_DOCKERFILE.read_text(encoding="utf-8")
    exclusions = {
        line.strip()
        for line in POLICY_EXCLUSIONS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert "PYTORCH_SOURCE_COMMIT" not in development_dockerfile
    assert "github.com/pytorch/pytorch/archive/" not in development_dockerfile
    assert not any(name.startswith("torch==") for name in exclusions)
