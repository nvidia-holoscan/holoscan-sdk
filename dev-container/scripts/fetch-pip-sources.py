#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download source distributions for installed pip packages (for compliance).

Queries the current environment for all installed pip packages, then downloads
each package's source distribution (sdist) into the current directory. Any
number of --exclude-list files can be passed to skip packages we don't need
to embed (e.g., the base-image "before" snapshot so we only fetch packages
added by this build, plus any scanner-detected list so we don't duplicate
sources published externally).

Packages skipped (not downloaded):
  - pip, setuptools, wheel, triton (build/runtime bootstrap)
  - Any package whose name contains "nvidia" (proprietary; handled separately)
  - Any package matched by an entry in an --exclude-list file, where entries
    are either 'name' (skip any version) or 'name==version' (skip only that
    exact version).

Usage: Run from the directory where source tarballs should be saved. Requires
network access and a working pip in the active environment.

  fetch-pip-sources.py [--exclude-list PATH ...]
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

DOWNLOAD_TIMEOUT_SECONDS = 30


def pep503(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def download_sdist(name: str, version: str, output_dir: Path) -> bool:
    """Download a package source distribution directly from the PyPI JSON API.

    Using the distribution URL avoids invoking an sdist's build backend merely
    to obtain package metadata. Some valid older sdists, such as Cloudpickle,
    no longer build metadata with the newest backend releases.
    """
    normalized_name = pep503(name)
    api_url = f"https://pypi.org/pypi/{normalized_name}/{version}/json"
    try:
        with urllib.request.urlopen(api_url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            release = json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError) as error:
        package = f"{name}=={version}"
        message = f"WARNING: Unable to find a PyPI source distribution for {package}: {error}"
        print(message, file=sys.stderr)
        return False

    source_files = [
        file
        for file in release.get("urls", [])
        if file.get("packagetype") == "sdist" and not file.get("yanked", False)
    ]
    if not source_files:
        print(f"WARNING: No PyPI source distribution for {name}=={version}", file=sys.stderr)
        return False

    source_file = source_files[0]
    output_path = output_dir / source_file["filename"]
    try:
        with (
            urllib.request.urlopen(
                source_file["url"], timeout=DOWNLOAD_TIMEOUT_SECONDS
            ) as response,
            output_path.open("wb") as output_file,
        ):
            shutil.copyfileobj(response, output_file)
    except (OSError, urllib.error.HTTPError, urllib.error.URLError) as error:
        print(f"WARNING: Failed to download {name}=={version}: {error}", file=sys.stderr)
        return False

    print(f"Downloaded {output_path.name}")
    return True


def load_exclude_list(path: str) -> tuple[set[str], set[str]]:
    """Parse an exclude list into (bare_names, exact_name_version).

    Each line is either 'name' (any version of this package is skipped) or
    'name==version' (only this exact version is skipped). '#' comments and
    blank lines are ignored. Names are PEP 503 normalized on both sides of
    the optional '==' for stable matching against 'pip list' output.
    """
    bare: set[str] = set()
    exact: set[str] = set()
    try:
        with open(path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "==" in line:
                    name, _, version = line.partition("==")
                    exact.add(f"{pep503(name)}=={version}")
                else:
                    bare.add(pep503(line))
    except FileNotFoundError:
        print(f"WARNING: --exclude-list file not found: {path}")
    return bare, exact


def main() -> int:
    parser = argparse.ArgumentParser(description="Download sdist for installed pip packages.")
    parser.add_argument(
        "--exclude-list",
        metavar="PATH",
        action="append",
        default=[],
        help=(
            "Plain-text file of packages to skip. Each line is either 'name' "
            "(any version) or 'name==version' (exact). '#' comments and blank "
            "lines ignored. Names are PEP 503 normalized before comparison. "
            "May be repeated to supply multiple files."
        ),
    )
    args = parser.parse_args()

    # excluded_bare:  package names (PEP 503 normalized) from bare "name" entries —
    #                 matched against any installed version.
    # excluded_exact: "name==version" strings — matched only at the listed version.
    excluded_bare: set[str] = set()
    excluded_exact: set[str] = set()
    for path in args.exclude_list:
        bare, exact = load_exclude_list(path)
        excluded_bare |= bare
        excluded_exact |= exact

    result = subprocess.run(
        ["pip", "list", "--format=json"],
        capture_output=True,
        check=True,
        text=True,
    )
    pkgs = json.loads(result.stdout)
    skip = {"pip", "setuptools", "wheel", "triton"}
    skip |= {
        pep503(p["name"])
        for p in pkgs
        if "nvidia" in p["name"].lower() or "cuda" in p["name"].lower()
    }

    download_failed = False
    for p in pkgs:
        name, version = p["name"], p["version"]
        normalized = pep503(name)
        if normalized in skip:
            continue
        # bare match skips any version; exact match skips only the listed version
        if normalized in excluded_bare or f"{normalized}=={version}" in excluded_exact:
            print(f"[excluded] {name}=={version}")
            continue
        if not download_sdist(name, version, Path(".")):
            download_failed = True

    return int(download_failed)


if __name__ == "__main__":
    sys.exit(main())
