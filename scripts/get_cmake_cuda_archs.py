#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn


def die(msg: str) -> NoReturn:
    """Logs an error message and exits with 1."""
    logging.error(msg)
    sys.exit(1)


def get_nvcc_path() -> Path:
    """Finds the nvcc executable in PATH or default location and returns its real Path object."""
    nvcc_path_str = shutil.which("nvcc")
    nvcc_path = Path(nvcc_path_str) if nvcc_path_str else Path("/usr/local/cuda/bin/nvcc")
    return nvcc_path.resolve()


def get_arch_sort_key(arch: str) -> tuple[int, str]:
    """Extracts (numeric_value, full_string) for sorting architecture strings."""
    match = re.match(r"\d+", arch)
    if not match:
        # only sort by string if no numeric match (unlikely)
        return (0, arch)

    # sort by numeric value first (ex: 90, 100), full string second (ex: 90, 90a)
    return (int(match.group()), arch)


def get_nvcc_archs(nvcc_path: Path) -> list[str]:
    """Runs `nvcc -code-ls` and parses the output."""
    cmd = [str(nvcc_path), "-code-ls"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        die(f"nvcc command '{str(nvcc_path)}' not found or not executable.")
    except subprocess.CalledProcessError as e:
        die(f"Command '{' '.join(cmd)}' failed:\n{e.stderr}")

    raw_archs: set[str] = set()
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("sm_"):
            raw_archs.add(line[3:])

    if not raw_archs:
        die("Could not parse any architectures (sm_XX) from 'nvcc -code-ls' output.")

    # Sort architectures numerically, then alphabetically (for suffixes)
    sorted_archs = sorted(raw_archs, key=get_arch_sort_key)
    logging.debug(f"nvcc supported archs: {', '.join(sorted_archs)}")
    return sorted_archs


def parse_requested_archs(req_str: str) -> list[str]:
    """Parses comma or space separated architecture string into a list."""
    if not req_str:
        return []
    archs = re.split(r"[\s,]+", req_str.strip())
    filtered_archs = list(filter(None, archs))
    return filtered_archs


def filter_archs_with_min_arch(archs: list[str], min_arch: int | None) -> list[str]:
    """Filters architecture list based on the minimum required major version."""
    if min_arch is None or min_arch <= 0:
        return list(archs)

    filtered_list = [arch for arch in archs if int(re.match(r"(\d+)", arch).group(1)) >= min_arch]
    logging.debug(f"Architectures >= sm_{min_arch}: {', '.join(filtered_list)}")
    return filtered_list


def filter_archs_for_platform(archs: list[str]) -> list[str]:
    """Filters out architectures not suitable for the current platform (iGPUs on x86_64)."""
    current_platform = platform.machine().lower()

    if current_platform not in ["x86_64", "amd64"]:
        return list(archs)  # Return a copy for consistency

    # Potential iGPU architectures (adjust if necessary for future hardware)
    igpu_archs: set[str] = {
        "72",
        "87",
        "101",
        "101a",
    }  # Xavier, Orin, Thor, DGX Spark (placeholders)

    # Log iGPU architectures to be removed
    if removed_igpus := list(set(archs) & igpu_archs):
        logging.debug(f"Removed iGPU archs from x86_64 build: {', '.join(removed_igpus)}")

    # Filter out iGPU architectures
    filtered_list = [arch for arch in archs if arch not in igpu_archs]
    logging.debug(f"Platform supported archs: {', '.join(filtered_list)}")

    return filtered_list


def filter_feature_specific_archs(archs: list[str]) -> list[str]:
    """Filters out architectures like sm_89 and those with 'a' or 'f' suffixes by default.

    sm_89 is excluded based on information like that found in PyTorch discussions
    (e.g., https://github.com/pytorch/pytorch/issues/152690#issuecomment-2847723785),
    where it's noted that sm_89 is SASS compatible with sm_86 and typically only
    required for specific features like FP8.
    Architectures with 'a' or 'f' suffixes usually denote specific hardware
    sub-variants not intended for a general 'all' selection.
    """
    excluded_values: tuple[str, ...] = "89"
    excludes_suffixes: tuple[str, ...] = ("a", "f")

    filtered_archs: list[str] = []
    excluded_archs: list[str] = []

    for arch in archs:
        exclude = arch.startswith(excluded_values) or arch.endswith(excludes_suffixes)

        if exclude:
            excluded_archs.append(arch)
        else:
            filtered_archs.append(arch)

    if excluded_archs:
        removed_str = ", ".join(sorted(excluded_archs, key=get_arch_sort_key))
        logging.debug(f"Removed feature-specific targets: {removed_str}")

    logging.debug(f"Non-feature-specific architectures: {', '.join(filtered_archs)}")
    return filtered_archs


def filter_major_archs(archs: list[str]) -> list[str]:
    """Filters architecture list to include only major versions (ending in 0)."""
    filtered_list = [arch for arch in archs if re.match(r"^\d+0$", arch)]
    logging.debug(f"Major architectures only: {', '.join(filtered_list)}")
    return filtered_list


def validate_user_archs(
    user_archs: list[str],
    nvcc_supported_archs: list[str],
    min_filtered_archs: list[str],
    platform_filtered_archs: list[str],
    non_specific_archs: list[str],
    min_arch_value: int | None,
    allow_specific_archs: bool,
) -> list[str]:
    """Validates user-provided architectures against supported and filtered lists."""
    if not user_archs:
        die("Requested architecture list is empty.")

    nvcc_supported_set = set(nvcc_supported_archs)
    min_filtered_set = set(min_filtered_archs)
    platform_filtered_set = set(platform_filtered_archs)
    non_specific_set = set(non_specific_archs)

    final_valid_archs = platform_filtered_set if allow_specific_archs else non_specific_set
    final_valid_archs_str = ", ".join(sorted(list(final_valid_archs), key=get_arch_sort_key))
    if not final_valid_archs_str:
        final_valid_archs_str = "<None>"

    validated_user_archs: list[str] = []
    for arch in user_archs:
        if arch not in nvcc_supported_set:
            die(
                f"Requested architecture '{arch}' is not supported by this version of nvcc. "
                f"Valid architectures: {final_valid_archs_str}"
            )
        if arch not in min_filtered_set:
            die(
                f"Requested architecture '{arch}' does not meet minimum requirement "
                f"(sm_{min_arch_value}). Valid architectures: {final_valid_archs_str}"
            )
        if arch not in platform_filtered_set:
            die(
                f"Requested architecture '{arch}' corresponds to an iGPU not supported "
                f"on this platform (x86_64). Valid architectures: {final_valid_archs_str}"
            )
        if arch not in non_specific_set:
            # Prepare reason for specific architecture exclusion
            reason = ""
            if arch == "89":
                reason = (
                    "is typically only needed for fp8 support, sm_86 is compatible on "
                    "Ada Lovelace for general usecases"
                )
            else:
                variant_type = (
                    "family-"
                    if arch.endswith("f")
                    else "architecture-"
                    if arch.endswith("a")
                    else ""
                )
                reason = (
                    f"is a {variant_type}variant needed for features "
                    "which are not generally needed for all cuda kernels"
                )
            if not allow_specific_archs:
                die(
                    f"Requested architecture '{arch}' is filtered out by default as it {reason} "
                    f"\nUse --allow-specific-archs to allow it."
                    f"\nValid architectures: {final_valid_archs_str}"
                )
            else:
                logging.warning(
                    f"Requested architecture '{arch}' {reason}. It is recommended to only use it "
                    f"for specific cuda kernels that need said features instead of globally."
                )
        validated_user_archs.append(arch)
    return validated_user_archs


def generate_sass_ptx_arch_list(target_archs: list[str]) -> list[str]:
    """Formats the final list with -real and -virtual suffixes for CMake."""
    if not target_archs:
        die("Cannot generate SASS/PTX list from empty target architectures.")

    # Ensure architectures are sorted numerically, then alphabetically
    sorted_archs = sorted(target_archs, key=get_arch_sort_key)

    # Generate SASS targets for all architectures
    final_archs = [f"{arch}-real" for arch in sorted_archs]

    # Find the highest architecture not ending with 'a' or 'f' for PTX target
    ptx_target_base = None
    for arch in reversed(sorted_archs):
        if not arch.endswith(("a", "f")):
            ptx_target_base = arch
            break

    # Only add PTX target if a suitable base architecture was found
    if ptx_target_base:
        final_archs.append(f"{ptx_target_base}-virtual")

    return final_archs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Determine and format CUDA architectures based on nvcc output and user request."
    )
    parser.add_argument(
        "requested_archs",
        help=(
            "Requested architectures: 'all', 'all-major', 'native', or a comma/space-separated "
            "list (e.g., '75 86 90a')."
        ),
    )
    parser.add_argument(
        "--nvcc-path",
        "-n",
        help=(
            "Path to the nvcc executable. Defaults to checking hint, PATH, "
            "then /usr/local/cuda/bin/nvcc."
        ),
    )
    parser.add_argument(
        "--min-arch",
        "-m",
        type=int,
        default=None,
        help=(
            "Minimum major CUDA architecture to consider (e.g., 70 for Volta+). "
            "Set to 0 or omit to disable."
        ),
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose debug logging to stderr."
    )
    parser.add_argument(
        "--allow-specific-archs",
        "-f",
        action="store_true",
        help=(
            "Allow architectures like sm_89 or those with 'a'/'f' suffixes (e.g., 90a) "
            "which are normally filtered out by default."
        ),
    )

    args = parser.parse_args()

    # Configure logging so verbose -> debug
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = "%(message)s" if log_level == logging.DEBUG else "%(levelname)s: %(message)s"
    logging.basicConfig(level=log_level, format=log_format, stream=sys.stderr)

    logging.debug(f"Requested CUDA architectures: {args.requested_archs}")

    # Return right away if requesting native
    req_lower = args.requested_archs.lower()
    if req_lower == "native":
        print("native", end="")
        sys.exit(0)

    # Get nvcc path
    nvcc_path: Path | None = None
    if args.nvcc_path:
        nvcc_path = Path(args.nvcc_path).resolve()
        logging.debug(f"Using user-provided nvcc path: {nvcc_path}")
    else:
        logging.debug("No nvcc path provided, attempting automatic search...")
        try:
            nvcc_path = get_nvcc_path()
            logging.debug(f"Using nvcc at: {nvcc_path}")
        except FileNotFoundError:
            die(
                "Could not find 'nvcc' automatically. Please provide path via --nvcc-path "
                "or ensure it's in PATH or /usr/local/cuda/bin/nvcc."
            )

    # Get supported architectures
    nvcc_supported_archs = get_nvcc_archs(nvcc_path)
    min_filtered_archs = filter_archs_with_min_arch(nvcc_supported_archs, args.min_arch)
    platform_filtered_archs = filter_archs_for_platform(min_filtered_archs)
    non_specific_archs = filter_feature_specific_archs(platform_filtered_archs)

    # Filter based on requested architectures
    target_archs: list[str] = []
    if req_lower == "all":
        target_archs = non_specific_archs
        logging.debug(f"Using all default-filtered CUDA architectures: {', '.join(target_archs)}")
    elif req_lower == "all-major":
        target_archs = filter_major_archs(non_specific_archs)
    else:
        user_archs = parse_requested_archs(args.requested_archs)
        target_archs = validate_user_archs(
            user_archs,
            nvcc_supported_archs,
            min_filtered_archs,
            platform_filtered_archs,
            non_specific_archs,
            args.min_arch,
            args.allow_specific_archs,
        )

    # Error if no valid architectures
    if not target_archs:
        error_msg = (
            f"No valid CUDA architectures could be determined for request '{args.requested_archs}' "
            f"with current filters (min_arch={args.min_arch}, platform={platform.machine()}, "
            f"allow_specific_archs={args.allow_specific_archs})."
        )
        die(error_msg)

    # Generate final list with SASS/PTX suffixes
    final_archs_list = generate_sass_ptx_arch_list(target_archs)

    # Print final list
    final_archs_str = ";".join(final_archs_list)
    logging.debug(f"Selected CUDA architectures: {final_archs_str}")
    print(final_archs_str, end="")


if __name__ == "__main__":
    main()
