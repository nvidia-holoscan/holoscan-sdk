#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: generate_holoscan_meta_deb.sh --output-dir DIR [--version VERSION]
                                      [--package PAYLOAD.deb ...]

Build an empty holoscan Debian meta-package. When payload packages are supplied,
their Package and Version fields determine the exact dependencies. Without
payload packages, --version is required and all five default Holoscan packages
are used. Supplied payloads may omit the optional holoscan-testing package.
EOF
}

output_dir=
requested_version=
declare -a payload_packages=()

while (($#)); do
    case "$1" in
        --output-dir)
            [[ $# -ge 2 ]] || { echo "ERROR: --output-dir requires a value" >&2; exit 2; }
            output_dir=$2
            shift 2
            ;;
        --version)
            [[ $# -ge 2 ]] || { echo "ERROR: --version requires a value" >&2; exit 2; }
            requested_version=$2
            shift 2
            ;;
        --package)
            [[ $# -ge 2 ]] || { echo "ERROR: --package requires a value" >&2; exit 2; }
            payload_packages+=("$2")
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

[[ -n "$output_dir" ]] || { echo "ERROR: --output-dir is required" >&2; exit 2; }
command -v dpkg-deb >/dev/null || { echo "ERROR: dpkg-deb was not found" >&2; exit 1; }

readonly -a required_packages=(
    holoscan-core
    holoscan-vision
    holoscan-inference
    holoscan-visualization
)
readonly -a package_order=("${required_packages[@]}" holoscan-testing)
declare -A selected_packages=()
package_version=$requested_version

if ((${#payload_packages[@]})); then
    for payload_package in "${payload_packages[@]}"; do
        [[ -f "$payload_package" ]] || {
            echo "ERROR: payload package does not exist: $payload_package" >&2
            exit 1
        }

        payload_name=$(dpkg-deb --field "$payload_package" Package)
        payload_version=$(dpkg-deb --field "$payload_package" Version)
        case "$payload_name" in
            holoscan-core|holoscan-vision|holoscan-inference|holoscan-visualization|holoscan-testing) ;;
            *)
                echo "ERROR: unexpected payload package '$payload_name': $payload_package" >&2
                exit 1
                ;;
        esac
        if [[ -n "${selected_packages[$payload_name]:-}" ]]; then
            echo "ERROR: duplicate payload package: $payload_name" >&2
            exit 1
        fi
        if [[ -n "$package_version" && "$payload_version" != "$package_version" ]]; then
            echo "ERROR: payload package versions do not match: $payload_name is $payload_version, expected $package_version" >&2
            exit 1
        fi
        package_version=$payload_version
        selected_packages[$payload_name]=1
    done

    for package_name in "${required_packages[@]}"; do
        [[ -n "${selected_packages[$package_name]:-}" ]] || {
            echo "ERROR: granular payloads must include $package_name" >&2
            exit 1
        }
    done
else
    [[ -n "$package_version" ]] || {
        echo "ERROR: --version is required when no payload packages are supplied" >&2
        exit 2
    }
    for package_name in "${package_order[@]}"; do
        selected_packages[$package_name]=1
    done
fi

declare -a dependencies=()
for package_name in "${package_order[@]}"; do
    if [[ -n "${selected_packages[$package_name]:-}" ]]; then
        dependencies+=("$package_name (= $package_version)")
    fi
done
dependency_list=
for dependency in "${dependencies[@]}"; do
    [[ -z "$dependency_list" ]] || dependency_list+=", "
    dependency_list+=$dependency
done

mkdir -p "$output_dir"
staging_dir=$(mktemp -d)
trap 'rm -rf "$staging_dir"' EXIT
mkdir -p "$staging_dir/DEBIAN"
chmod 0755 "$staging_dir" "$staging_dir/DEBIAN"

{
    echo "Package: holoscan"
    echo "Version: $package_version"
    echo "Priority: optional"
    echo "Section: multiverse/devel"
    echo "Maintainer: cudatools <cudatools@nvidia.com>"
    echo "Depends: $dependency_list"
    echo "Architecture: all"
    echo "Description: NVIDIA Holoscan SDK"
    echo " The AI sensor processing platform"
} > "$staging_dir/DEBIAN/control"

output_package="$output_dir/holoscan_${package_version}_all.deb"
dpkg-deb --build --root-owner-group "$staging_dir" "$output_package"
echo "$output_package"
