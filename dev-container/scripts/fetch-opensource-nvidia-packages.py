#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refresh opensource-nvidia-{apt,pip}-packages.txt from opensource.nvidia.com.

Discovers the most recent Holoscan release published on
https://opensource.nvidia.com/oss/teams/nvidia/holoscan/, fetches the package
listing for every architecture variant of that release, and rewrites the two
plain-text exclusion lists consumed by fetch-apt-sources.sh and
fetch-pip-sources.py.

Classification relies on scanner page ordering: pip packages appear first,
APT packages follow. The script queries PyPI for each package in page order;
the first HTTP 404 marks the boundary — all subsequent packages are classified
as APT without further queries.

Usage:
  fetch-opensource-nvidia-packages.py [--release vX.Y.Z-cudaN]
                                      [--apt-output PATH] [--pip-output PATH]

"""

import argparse
import pathlib
import re
import sys
import urllib.error
import urllib.request

BASE_URL = "https://opensource.nvidia.com/oss/teams/nvidia/holoscan/"
SCRIPTS_DIR = pathlib.Path(__file__).parent
DEFAULT_APT_OUTPUT = SCRIPTS_DIR / "opensource-nvidia-apt-packages.txt"
DEFAULT_PIP_OUTPUT = SCRIPTS_DIR / "opensource-nvidia-pip-packages.txt"


def pep503(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


_VERSION_START = re.compile(r"-\d+(?=[.:+~]|[a-z]+\d|$)")


def split_name_version(leaf: str) -> tuple[str, str]:
    """Split a 'name-<version>' leaf into (name, version).

    The version starts at the first '-<digits>' that is immediately followed
    by a version-signaling char (. : + ~), or by letters-then-digit
    (e.g. 'ubuntu1', 'dfsg2'), or by end-of-string. Examples:
      'cloudpickle-3.1.2'                         → ('cloudpickle', '3.1.2')
      'binutils-x86-64-linux-gnu-2.42-4ubuntu2.5' → ('binutils-x86-64-linux-gnu',
                                                     '2.42-4ubuntu2.5')
          (the '-64-linux' is not a version because '-linux' follows)
      'base-files-13ubuntu10.2'                   → ('base-files',
                                                     '13ubuntu10.2')
      'cpp-13-13.3.0-6ubuntu2~24.04'              → ('cpp-13',
                                                     '13.3.0-6ubuntu2~24.04')
      'bsdutils-1:2.39.3-9ubuntu6.3'              → ('bsdutils',
                                                     '1:2.39.3-9ubuntu6.3')
    """
    m = _VERSION_START.search(leaf)
    if m:
        return leaf[: m.start()], leaf[m.start() + 1 :]
    return leaf, ""


def _parse_tag(tag: str) -> tuple[int, int, int, str]:
    """Return (major, minor, patch, suffix) from a 'vX.Y.Z-suffix' tag."""
    m = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)-(.+)", tag)
    if not m:
        raise ValueError(f"Cannot parse release tag: {tag!r}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4)


def _probe_url(url: str) -> bool | None:
    """HEAD-probe a URL. Returns True (200), False (HTTP error), None (network error)."""
    try:
        urllib.request.urlopen(url, timeout=10).close()
        return True
    except urllib.error.HTTPError:
        return False
    except OSError:
        return None


def probe_release(release: str) -> bool:
    """Return True if `release` has at least one accessible arch page."""
    for arch in ("linux-arm64", "linux-amd64", "linux-aarch64"):
        result = _probe_url(f"{BASE_URL}{release}:{arch}/index.html")
        if result is True:
            return True
        if result is None:  # network error — stop trying
            return False
    return False


def discover_latest_release() -> str:
    """Return the newest published release tag.

    Parses the parent index for a baseline (the index page is rendered via
    JavaScript so urllib sees only entries up to the last cached HTML snapshot),
    then probes forward by minor version and by next major to find releases the
    index may not yet list.

    Release tags look like 'vX.Y.Z-<suffix>' where suffix includes 'cuda13'. We rank by
    (major, minor, patch, cuda_major) so cuda-suffixed releases beat
    pre-cuda ones at the same version, and higher cuda numbers win.
    """
    html = fetch(BASE_URL)
    pattern = re.compile(r"v(\d+)\.(\d+)\.(\d+)-([^/:\"<> ]+)[/:]")
    seen: list[tuple[tuple[int, int, int, int, str], str]] = []
    for m in pattern.finditer(html):
        major, minor, patch = (int(m.group(i)) for i in (1, 2, 3))
        suffix = m.group(4)
        cuda_m = re.match(r"cuda(\d+)", suffix)
        cuda_n = int(cuda_m.group(1)) if cuda_m else 0
        tag = f"v{major}.{minor}.{patch}-{suffix}"
        seen.append(((major, minor, patch, cuda_n, suffix), tag))
    if not seen:
        raise RuntimeError(f"No release tags discovered at {BASE_URL}")
    best_tag = max(seen)[1]

    # The index HTML lags behind the actual scanner (content is JS-rendered).
    # Probe for newer minor versions and the next major using direct URL checks.
    major, minor, patch, suffix = _parse_tag(best_tag)
    probe_minor = minor + 1
    while probe_release(f"v{major}.{probe_minor}.0-{suffix}"):
        best_tag = f"v{major}.{probe_minor}.0-{suffix}"
        probe_minor += 1
    if probe_release(f"v{major + 1}.0.0-{suffix}"):
        best_tag = f"v{major + 1}.0.0-{suffix}"

    return best_tag


def discover_arches(release: str) -> list[str]:
    """Find all arch subpages published for a release tag.

    Tries the parent index first; falls back to probing known arch names directly
    for releases not yet listed in the JS-rendered index HTML.
    """
    html = fetch(BASE_URL)
    arches: set[str] = set()
    pattern = re.compile(rf"{re.escape(release)}:([^/\"<> ]+)/")
    for m in pattern.finditer(html):
        arches.add(m.group(1))
    if not arches:
        for arch in ("linux-arm64", "linux-amd64", "linux-aarch64"):
            if _probe_url(f"{BASE_URL}{release}:{arch}/index.html") is True:
                arches.add(arch)
    return sorted(arches)


def fetch_packages(release: str, arch: str) -> list[tuple[str, str]]:
    """Fetch one <release>:<arch>/index.html and return (name, version) pairs in page order.

    Package entries look like:
      <a href="https://opensource.nvidia.com/oss/packages/NAME-VERSION/index.html">
    We match '/packages/<leaf>/' and split the leaf into (name, version).
    """
    url = f"{BASE_URL}{release}:{arch}/index.html"
    html = fetch(url)
    seen: dict[str, str] = {}
    for m in re.finditer(r'/packages/([^"/]+)/', html):
        name, version = split_name_version(m.group(1))
        if name and name not in seen:
            seen[name] = version
    return list(seen.items())


def load_list(path: pathlib.Path) -> set[str]:
    """Read a plain-text list file (one entry per line, '#' comments and blank
    lines ignored). Entries may be bare names ('jinja2') or versioned
    ('jinja2==3.1.6'); we return them verbatim and let callers parse.
    """
    entries: set[str] = set()
    if not path.exists():
        return entries
    with path.open() as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            entries.add(line)
    return entries


def entry_name(entry: str) -> str:
    """Return the bare name portion of a 'name' or 'name==version' entry."""
    return entry.split("==", 1)[0]


def write_list(
    path: pathlib.Path,
    entries: set[str],
    release: str,
    arches: list[str],
    kind: str,
) -> None:
    kind_label = {"apt": "APT", "pip": "PyPI"}[kind]
    name_rule = {
        "apt": "Names are Debian binary package names (matching 'dpkg --get-selections').",
        "pip": "Names are PEP 503 normalized (lowercase, '_' and '.' replaced by '-').",
    }[kind]
    header = [
        f"# {kind_label} packages detected by the NVIDIA open-source compliance scanner.",
        "#",
        "# When the released container is scanned, any package listed on",
        "# https://opensource.nvidia.com/oss/teams/nvidia/holoscan/<version>/<arch>/",
        "# is published there automatically. Manually embedding the same sources under",
        "# /opt/sources/ in the container is redundant, so fetch-apt-sources.sh and",
        "# fetch-pip-sources.py skip packages with a matching entry here.",
        "#",
        "# Format: one entry per line, either 'name' (any version) or 'name==version'",
        "# (exact version match only). Blank lines and lines starting with '#' are",
        "# ignored. Version-specific entries let us re-fetch sources when we ship a",
        "# newer package version than the scanner previously published.",
        "#",
        "# To refresh from the latest scanner listings, run:",
        "#   dev-container/scripts/fetch-opensource-nvidia-packages.py",
        "#",
        f"# Sampled release: {release} ({', '.join(arches)})",
        "#",
        f"# {name_rule}",
        "",
    ]
    body = sorted(entries)
    path.write_text("\n".join(header + body) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Refresh opensource-nvidia-{apt,pip}-packages.txt."
    )
    parser.add_argument(
        "--release",
        help="Release tag to sample (e.g. v4.1.0-cuda13). Defaults to latest discovered.",
    )
    parser.add_argument(
        "--apt-output",
        type=pathlib.Path,
        default=DEFAULT_APT_OUTPUT,
        help="Path to the APT exclusion list.",
    )
    parser.add_argument(
        "--pip-output",
        type=pathlib.Path,
        default=DEFAULT_PIP_OUTPUT,
        help="Path to the pip exclusion list.",
    )
    args = parser.parse_args()

    release = args.release or discover_latest_release()
    print(f"Using release: {release}", file=sys.stderr)

    arches = discover_arches(release)
    if not arches:
        print(f"No arch listings found for {release}", file=sys.stderr)
        return 1
    print(f"Arches: {', '.join(arches)}", file=sys.stderr)

    # Merge packages from all arches, preserving scanner page order (first-seen wins).
    detected: dict[str, str] = {}
    for arch in arches:
        pairs = fetch_packages(release, arch)
        print(f"  {arch}: {len(pairs)} packages", file=sys.stderr)
        for name, version in pairs:
            detected.setdefault(name, version)

    # Read existing output before overwriting — used only for the dropped-packages diff below.
    apt_known_names = {entry_name(e) for e in load_list(args.apt_output)}
    pip_known_names = {pep503(entry_name(e)) for e in load_list(args.pip_output)}

    # Classify in scanner page order. PyPI packages appear first on the scanner
    # page; the first HTTP 404 marks the boundary — all subsequent packages are APT.
    new_apt: set[str] = set()
    new_pip: set[str] = set()
    pypi_exhausted = False
    for name, version in detected.items():
        entry_apt = f"{name}=={version}" if version else name
        entry_pip = f"{pep503(name)}=={version}" if version else pep503(name)
        if not pypi_exhausted:
            try:
                urllib.request.urlopen(
                    f"https://pypi.org/pypi/{pep503(name)}/json", timeout=10
                ).close()
                new_pip.add(entry_pip)
                continue
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
                pypi_exhausted = True
        new_apt.add(entry_apt)

    write_list(args.apt_output, new_apt, release, arches, "apt")
    write_list(args.pip_output, new_pip, release, arches, "pip")

    # Surface what dropped vs. what was added, by bare name (version churn on
    # the same package is normal and not worth reporting each run).
    new_apt_names = {entry_name(e) for e in new_apt}
    new_pip_names = {pep503(entry_name(e)) for e in new_pip}
    dropped_apt = apt_known_names - new_apt_names
    dropped_pip = pip_known_names - new_pip_names

    print(f"Wrote {args.apt_output} ({len(new_apt)} entries)", file=sys.stderr)
    print(f"Wrote {args.pip_output} ({len(new_pip)} entries)", file=sys.stderr)

    if dropped_apt:
        print(
            f"  dropped from apt list (no longer detected): {', '.join(sorted(dropped_apt))}",
            file=sys.stderr,
        )
    if dropped_pip:
        print(
            f"  dropped from pip list (no longer detected): {', '.join(sorted(dropped_pip))}",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
