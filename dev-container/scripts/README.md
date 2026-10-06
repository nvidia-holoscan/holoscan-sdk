<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Holoscan development container — OSRB source embedding

The public Holoscan development container (`public/Dockerfile.dev`) embeds
third-party source packages under `/opt/sources/` to satisfy NVIDIA's
open-source redistribution (OSRB) requirements. After each container is registered in NSpect, NVIDIA's Pulse Curator
external scanner publishes a subset of those sources at
[opensource.nvidia.com/oss/teams/nvidia/holoscan](https://opensource.nvidia.com/oss/teams/nvidia/holoscan/).
The scripts here avoid re-embedding what the scanner will publish
independently, reducing image size without sacrificing OSRB coverage.

## How the pieces fit together

```text
BUILD: "make dev_container" pre-step
fetch-opensource-nvidia-packages.py ──→ opensource-nvidia-apt-packages.txt
                                    └──→ opensource-nvidia-pip-packages.txt

BUILD: BASE STAGE
save-package-lists.sh ──→ apt-before.txt
                      └──→ pip-before.txt

BUILD: OSRB-SOURCES STAGE
apt-before.txt ───────────────────────┐
opensource-nvidia-apt-packages.txt ───┤
osrb-apt-exclusions.txt ────────────┴──→ fetch-apt-sources.sh → /opt/sources/apt/

pip-before.txt ───────────────────────┐
opensource-nvidia-pip-packages.txt ───┤
osrb-pip-exclusions.txt ─────────────┴──→ fetch-pip-sources.py → /opt/sources/pip/
```

**Build flow:**

1. `fetch-opensource-nvidia-packages.py` via Make commands
   (`make fetch_oss_packages`, `make dev_container`) or manually
   to query opensource.nvidia.com and write `opensource-nvidia-{apt,pip}-packages.txt`
   with `name==version` for each package covered by those scan results.
2. `save-package-lists.sh` runs on top of the base container but before any new
   installs, and records every installed APT and pip package as `name==version`.
3. Container build proceeds and installs packages via `APT`, `pip`, `git`, `curl`.
4. After all installs complete, each fetch script (`fetch-apt-sources.sh`
   and `fetch-pip-sources.py`) each query the environment for installed
   packages and downloads sources, excluding the files captured in the initial steps above.

## Exclusion list format

`fetch-apt-sources.sh` and `fetch-pip-sources.py` attempt to embed sources for all packages, *except*
for packages passed via explicit exclusion lists.

All exclusion files share one format: one entry per line, blank lines and `#`
comments ignored. An entry is either:

- (Preferred) `name==version` — skip this package only at this exact version. Packages at
  opensource.nvidia.com or in the image base container will always have a version associated.
- `name` — skip this package at any version. Intended for cases where automatic version parsing
  fails or for edge cases where manual override is desired.

The fetch scripts accept any number of files and merge them at runtime for a complete list of
packages to ignore when fetching sources.

## Refreshing the scanner exclusion lists

`opensource-nvidia-{apt,pip}-packages.txt` are generated data files reflecting
packages already hosted at opensource.nvidia.com.
`make dev_container` refreshes them automatically before each build.

The `fetch-opensource-nvidia-packages.py` script discovers the latest published Holoscan release on
`opensource.nvidia.com`, fetches its package listings for all architecture variants, and
rewrites both exclusion files with versioned `name==version` entries. Classification relies
on the presentation order of scanner resultss: pip packages appear first, APT packages follow.
The script queries PyPI for each package in order; the first HTTP 404 marks the boundary —
all subsequent packages are classified as APT without further queries.

**Flags:**

- `--release vX.Y.Z-cudaN` — sample a specific release instead of the latest.
- `--apt-output PATH` / `--pip-output PATH` — write to a different path.

The script only needs outbound HTTPS to `opensource.nvidia.com` and PyPI.

## Per-script reference

### `save-package-lists.sh`

- **When:** base stage, before any new installs.
- **Outputs:** `${SOURCES_PATH}/apt-before.txt` and `${SOURCES_PATH}/pip-before.txt`, each a
  sorted list of `name[==version]` entries in the same format as all other
  exclusion files.
- **Env:** `SOURCES_PATH` (default `/opt/sources`).

### `fetch-apt-sources.sh`

- **When:** osrb-sources stage, after all apt installs.
- **Requires:** apt `deb-src` entries, `dpkg-dev` installed.
- **Outputs:** `.tar.*` and `.dsc` archives under `${SOURCES_PATH}/apt/`.
- **Failure behavior:** attempts both the installed-version and unversioned
  lookup for every package, then exits nonzero if any source remains unavailable.
- **Env:**
  - `SOURCES_PATH` — default `/opt/sources`.
  - `EXCLUDE_LISTS` — whitespace-separated paths to exclusion files. Missing
    files emit a warning but don't abort. Skipped packages log as
    `APT [excluded]: name==version`.

### `fetch-pip-sources.py`

- **When:** osrb-sources stage, after all pip installs.
- **Requires:** network access to PyPI.
- **Outputs:** sdist archives under `${SOURCES_PATH}/pip/`.
- **Failure behavior:** exits nonzero if any eligible package has no available
  source distribution or its PyPI download fails.
- **CLI:** `--exclude-list PATH` (may be repeated). Skipped packages log as
  `[excluded] name==version`.
- **Always skipped:** `pip`, `setuptools`, `wheel`, `triton`, and any package
  whose name contains `nvidia` or `cuda` (proprietary, handled separately).

### `fetch-opensource-nvidia-packages.py`

- **When:** run automatically by `make fetch_oss_packages` (a prerequisite of `make dev_container`), or invoked manually. Not run inside the Docker build itself.
- **Requires:** outbound HTTPS to `opensource.nvidia.com` and PyPI.
- **Outputs:** rewrites `opensource-nvidia-{apt,pip}-packages.txt` in place.
- **CLI:** `--release vX.Y.Z-cudaN`, `--apt-output PATH`, `--pip-output PATH`.

### `opensource-nvidia-apt-packages.txt` / `opensource-nvidia-pip-packages.txt`

**Gitignored generated files** — do not commit. Regenerated by `fetch-opensource-nvidia-packages.py`
(automatically via `make dev_container`, or manually). APT entries use Debian binary
package names and version strings; pip entries use PEP 503-normalized names
(`_` and `.` replaced by `-`) and PyPI version strings.

### `osrb-apt-exclusions.txt`

**Committed policy exclusions** for APT packages whose source should not be
embedded. Unlike the generated scanner lists, this file is maintained manually.
It excludes the proprietary Nsight Systems CLI and NVPL packages. Because the
Nsight release is part of its Debian package name, update the entry together
with `NSYS_VERSION` in `public/Dockerfile.dev`. It also excludes the two NVSHMEM
binary packages from the generic APT source fetch because their version-matched
official upstream archive is embedded separately under `/opt/sources/nvshmem/`.

### NVSHMEM source archive

The CUDA 13 packages `libnvshmem3-cuda-13` and
`libnvshmem3-dev-cuda-13`, both at version `3.7.2-1`, correspond to official
NVSHMEM release `v3.7.2-0`, commit
`3f4c6d81225f45f9c42ed9af09bdb3c61eb356d4`.
`public/Dockerfile.dev` downloads that exact commit archive from the official
NVIDIA GitHub repository and verifies its SHA-256 before preserving it under
`/opt/sources/nvshmem/`. The unmodified archive contains the upstream
`License.txt` and `NOTICE`; its adjacent README records the same provenance.

### `osrb-pip-exclusions.txt`

**Committed policy exclusions** for pip packages whose source is embedded by a
package-specific path instead of downloaded from PyPI. It currently excludes
only the exact CUDA-specific PyTorch wheel version installed in the image.

### PyTorch source archive

The `torch==2.11.0+cu130` wheel is built from official PyTorch release
`v2.11.0`, commit `70d99e998b4955e0049d13a98d77ae1b14db1f45`.
`public/Dockerfile.dev` downloads that exact commit archive from the official
PyTorch GitHub repository and verifies its SHA-256 before preserving it under
`/opt/sources/pytorch/`. The unmodified archive contains the upstream `LICENSE`
and `NOTICE`; its adjacent README records the same provenance.

## Verifying a build

1. Build the release image (`make dev_container`).
2. Grep the build log for `[excluded]` — both scripts log every skipped package.
   Expect a long list covering most base-image packages and many scanner-detected
   ones.
3. `docker run --rm <image> ls /opt/sources/pip` — should contain only packages
   we added that the scanner doesn't cover (today: typically just `networkx`).
4. `docker run --rm <image> ls /opt/sources/apt` — should contain packages
   from non-default repos (Kitware, CUDA developer downloads) and
   graphics-stack libraries the scanner doesn't detect.
5. CCCL sources are hosted externally at
   `edge.urm.nvidia.com/artifactory/sw-holoscan-thirdparty-generic-local/cccl/`
   (see !4489).

## Troubleshooting / FAQ

**I want to skip a package that is being embedded in the container.**

If the package is a new dependency, the next scan via NSight Pulse Curator may add it to
opensource.nvidia.com. In that case, future container builds will automatically skip
embedding the package in the container.

Otherwise, manually add the package name and version in a custom "exclude.txt" and pass it to
the "fetch" scripts with the `--exclude-list` argument.

**I want to embed a package that is being skipped in the container.**

The `fetch-apt-sources.sh` and `fetch-pip-sources.py` scripts do not support overriding
packages that are automatically excluded. If you believe the package should not be in the
scanner list, re-run `fetch-opensource-nvidia-packages.py` to refresh from the latest scan results.
Otherwise, consider manually embedding the package with a custom RUN stage in the Dockerfile.

**I need to permanently skip a package the generated lists don't cover.**

Create a committed plain-text file in this directory — one entry per line using
the [exclusion list format](#exclusion-list-format):

```text
# public/dev-container/scripts/my-apt-overrides.txt
some-package            # skip at any version
other-package==1.2.3    # skip only this exact version
```

Then wire it into `public/Dockerfile.dev`:

```dockerfile
# APT — add to the EXCLUDE_LISTS env var:
RUN SOURCES_PATH=... \
    EXCLUDE_LISTS="... /opt/sources/scripts/my-apt-overrides.txt" \
    /opt/sources/scripts/fetch-apt-sources.sh

# pip — add a --exclude-list flag:
RUN python3 /tmp/fetch-pip-sources.py \
        --exclude-list ... \
        --exclude-list /path/to/my-pip-overrides.txt \
        ...
```

No script changes are needed — the fetch scripts accept any number of files.

**The classification looks wrong (a pip package ended up in the apt list, or vice versa).**

Classification relies on the scanner page ordering: pip packages are expected to
appear before the first APT package. To investigate:

1. Visit the latest results at <https://opensource.nvidia.com/oss/teams/nvidia/holoscan> and
verify:
    a. The package under investigation appears in the scan results
    b. PyPI dependencies are grouped together before APT dependencies in the list
2. (Local workaround) Disable the `make fetch_oss_packages` pre-step, manually edit the
package lists, and build the development container with your custom exclusion lists.

**`fetch-opensource-nvidia-packages.py` fails with a network error.**

The script propagates PyPI errors rather than silently misclassifying packages.
Re-run once connectivity is restored. The generated files from a previous
successful run remain in place and will be reused by `make dev_container` if the
network is unavailable. In a clean workspace with no generated lists yet,
`make dev_container` creates comment-only fallback lists and continues; source
embedding then fetches all detected package sources instead of skipping packages
already published by the scanner (see *Generated files refreshed before each
build* below).

## Design notes

**Generated files refreshed before each build** via `make fetch_oss_packages`
(a prerequisite of `dev_container`). Fetched from `opensource.nvidia.com` at
build-prep time rather than inside Docker, so the Docker layer cache stays stable.
Falls back gracefully if the network is unavailable and the files already exist.

**Separate files** (APT and pip). Each fetch script operates in one
ecosystem; two plain lists are simpler to read and diff than a single file with
shared syntax.

**Generic interface** (`EXCLUDE_LISTS` / `--exclude-list`). The fetch scripts
know nothing about the scanner — they accept arbitrary exclusion files. The base
snapshot and scanner lists are generated and gitignored, so they must not be
edited by hand; `osrb-apt-exclusions.txt` is the committed list for persistent
APT policy exclusions. Pass any additional file as an entry in `EXCLUDE_LISTS`
(APT) or an additional `--exclude-list` flag (pip) in `public/Dockerfile.dev`.
No script changes are needed.

**Version-specific exclusion** (`name==version`). If we ship a
newer package version than the scanner last published, the entry doesn't match
and we still embed the new source.
