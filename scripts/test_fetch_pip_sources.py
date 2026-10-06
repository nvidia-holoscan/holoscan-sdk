#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_DOCKERFILE = REPOSITORY_ROOT / "public/Dockerfile.dev"
SDK_DOCKERFILE = REPOSITORY_ROOT / "public/Dockerfile"
POLICY_EXCLUSIONS = REPOSITORY_ROOT / "public/dev-container/scripts/osrb-pip-exclusions.txt"
PYTORCH_PROVENANCE = REPOSITORY_ROOT / "public/dev-container/scripts/pytorch-source-provenance.md"
SCRIPT_CANDIDATES = (
    REPOSITORY_ROOT / "public/dev-container/scripts/fetch-pip-sources.py",
    REPOSITORY_ROOT / "docker/release/scripts/fetch-pip-sources.py",
)
SCRIPT = next(path for path in SCRIPT_CANDIDATES if path.is_file())
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


def docker_arg(dockerfile: str, name: str) -> str:
    """Return the default value of a global Docker build argument."""
    match = re.search(rf"^ARG {re.escape(name)}=(\S+)$", dockerfile, re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_pytorch_wheel_and_commit_pinned_source_stay_in_sync() -> None:
    development_dockerfile = DEVELOPMENT_DOCKERFILE.read_text(encoding="utf-8")
    sdk_dockerfile = SDK_DOCKERFILE.read_text(encoding="utf-8")
    provenance = PYTORCH_PROVENANCE.read_text(encoding="utf-8")
    exclusions = {
        line.strip()
        for line in POLICY_EXCLUSIONS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    version = docker_arg(development_dockerfile, "PYTORCH_VERSION")
    cuda_major = docker_arg(development_dockerfile, "CUDA_MAJOR")
    source_commit = docker_arg(development_dockerfile, "PYTORCH_SOURCE_COMMIT")
    source_sha256 = docker_arg(development_dockerfile, "PYTORCH_SOURCE_SHA256")

    assert version == docker_arg(sdk_dockerfile, "PYTORCH_VERSION") == "2.11.0"
    assert source_commit == "70d99e998b4955e0049d13a98d77ae1b14db1f45"
    assert source_sha256 == "9578f2a21628741278e8aaa4ee5a9dd9b636c30d6483e30093e2131b5e3481d9"
    assert f"torch=={version}+cu{cuda_major}0" in exclusions

    osrb_stage = development_dockerfile.split("FROM deps-common AS osrb-sources", maxsplit=1)[1]
    assert "ARG PYTORCH_VERSION" in osrb_stage
    assert "ARG PYTORCH_SOURCE_COMMIT" in osrb_stage
    assert "ARG PYTORCH_SOURCE_SHA256" in osrb_stage
    assert development_dockerfile.count("github.com/pytorch/pytorch/archive/") == 1
    assert "${PYTORCH_SOURCE_COMMIT}.tar.gz" in osrb_stage
    assert "${PYTORCH_SOURCE_SHA256}  ${PYTORCH_SOURCE_ARCHIVE}" in osrb_stage
    assert "sha256sum --check --strict" in osrb_stage
    assert "grep -q '/LICENSE$'" in osrb_stage
    assert "grep -q '/NOTICE$'" in osrb_stage
    assert "${SOURCES_PATH}/pytorch" in osrb_stage
    assert "${SOURCES_PATH}/scripts/osrb-pip-exclusions.txt" in osrb_stage

    assert "v2.11.0" in provenance
    assert f"torch=={version}+cu{cuda_major}0" in provenance
    assert source_commit in provenance
    assert source_sha256 in provenance
