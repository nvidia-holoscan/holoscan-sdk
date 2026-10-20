#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Pyright's Python wheel contains its JavaScript; only the Node runtime is needed.
# Keep the release and architecture-specific checksums together when updating Node.
readonly NODE_VERSION=24.21.0
readonly NODE_SOURCE_SHA256=a6f54defb6fd7c84f41dba13d61e78e9b4e0961712cf61f29715c05f5ced94fc
case "${1:-}" in
    amd64)
        node_arch=x64
        node_sha256=fd8e59d5a511510f6a298afb548f18c7d2b1be404d8b4a27d94fbe49f56cb2d6
        ;;
    arm64)
        node_arch=arm64
        node_sha256=6ad1325edbdb5649c379b75a237147a666c95d4f9ae8d340fef2d1575d289ad2
        ;;
    *)
        echo "Usage: ${0##*/} {amd64|arm64}" >&2
        exit 1
        ;;
esac

node_download_dir=$(mktemp -d)
trap 'rm -rf "${node_download_dir}"' EXIT
node_release="node-v${NODE_VERSION}-linux-${node_arch}"
node_archive="${node_download_dir}/${node_release}.tar.xz"
curl -fsSL --retry 5 --retry-delay 3 \
    "https://nodejs.org/dist/v${NODE_VERSION}/${node_release}.tar.xz" \
    -o "${node_archive}"
printf '%s  %s\n' "${node_sha256}" "${node_archive}" | sha256sum --check --status
# Extract only the runtime and its license, leaving npm and Corepack out of the image.
tar -xJf "${node_archive}" -C "${node_download_dir}" \
    "${node_release}/bin/node" "${node_release}/LICENSE"
install -Dm755 "${node_download_dir}/${node_release}/bin/node" /usr/local/bin/node
install -Dm644 "${node_download_dir}/${node_release}/LICENSE" \
    /usr/local/share/doc/nodejs/LICENSE
printf '%s  node-v%s.tar.xz\n' "${NODE_SOURCE_SHA256}" "${NODE_VERSION}" \
    > /usr/local/share/doc/nodejs/source.sha256
test "$(node --version)" = "v${NODE_VERSION}"
