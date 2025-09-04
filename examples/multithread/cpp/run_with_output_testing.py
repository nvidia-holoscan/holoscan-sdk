"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

import argparse
import os
import re
import subprocess
import sys

import yaml

# Expected port map output
EXPECTED_PORT_MAP = """input_to_output:
  mx0.in:
    - tx.out
  mx1.in:
    - tx.out
  mx2.in:
    - tx.out
  mx3.in:
    - tx.out
  rx.names:0:
    - mx0.out_name
  rx.names:1:
    - mx1.out_name
  rx.names:2:
    - mx2.out_name
  rx.names:3:
    - mx3.out_name
  rx.values:0:
    - mx0.out_val
  rx.values:1:
    - mx1.out_val
  rx.values:2:
    - mx2.out_val
  rx.values:3:
    - mx3.out_val
output_to_input:
  mx0.out_name:
    - rx.names:0
  mx0.out_val:
    - rx.values:0
  mx1.out_name:
    - rx.names:1
  mx1.out_val:
    - rx.values:1
  mx2.out_name:
    - rx.names:2
  mx2.out_val:
    - rx.values:2
  mx3.out_name:
    - rx.names:3
  mx3.out_val:
    - rx.values:3
  tx.out:
    - mx0.in
    - mx1.in
    - mx2.in
    - mx3.in"""

# Required patterns for tracking and data logging
EXPECTED_TRACKING_PATTERNS = [
    r"Data Flow Tracking Results:",
    r"tx->out: 1",  # count: 1 in the config
]
EXPECTED_DATA_LOGGING_PATTERNS = [
    r"BasicConsoleLogger\[ID:tx\.out\]",
    r"BasicConsoleLogger\[ID:mx15\.in\]",
    r"BasicConsoleLogger\[ID:mx15\.out_val\]",
    r"BasicConsoleLogger\[ID:mx15\.out_name\]",
    r"BasicConsoleLogger\[ID:rx\.values:15\]",
    r"Category:Message \(std::any\)",  # holoscan::Message type for int and string
]

# Failure patterns for data logging
FAILURE_PATTERNS_DATA_LOGGING = [
    r"BasicConsoleLogger\[ID:rx\.names:15\]",  # excluded by denylist
]


def test_port_map(output):
    """Test port map output"""
    # Extract the port map section using regex
    pattern = r"====== PORT MAPPING =======\n(.+?)(?=\n\[|$)"
    match = re.search(pattern, output, re.DOTALL)

    if not match:
        print("Port mapping section not found in output")
        print("Full output:")
        print(output)
        return False

    actual_port_map_str = match.group(1).strip()

    # Parse both actual and expected port maps as YAML for robust comparison
    try:
        actual_port_map = yaml.safe_load(actual_port_map_str)
        expected_port_map = yaml.safe_load(EXPECTED_PORT_MAP)
    except yaml.YAMLError as e:
        print(f"Failed to parse YAML: {e}")
        print("Actual port map string:")
        print(actual_port_map_str)
        return False

    # Compare the parsed data structures
    if actual_port_map == expected_port_map:
        print("Port map output matches expected result")
        return True
    else:
        print("Port map output does not match expected result")
        print("Expected:")
        print(yaml.dump(expected_port_map, default_flow_style=False))
        print("Actual:")
        print(yaml.dump(actual_port_map, default_flow_style=False))
        return False


def test_tracking_and_data_logging(output):
    """Test tracking output"""
    missing_patterns = []
    failure_patterns_found = []

    # Check for required patterns
    for pattern in EXPECTED_TRACKING_PATTERNS + EXPECTED_DATA_LOGGING_PATTERNS:
        if not re.search(pattern, output):
            missing_patterns.append(pattern)  # noqa: PERF401

    # Check for failure patterns (patterns that should NOT be present)
    for pattern in FAILURE_PATTERNS_DATA_LOGGING:
        if re.search(pattern, output):
            failure_patterns_found.append(pattern)  # noqa: PERF401

    # Report results
    if failure_patterns_found:
        print("Test failed: Found patterns that should not be present:")
        for pattern in failure_patterns_found:
            print(f"  - {pattern}")
        print("\nFull output:")
        print(output)
        return False
    elif missing_patterns:
        print("Missing expected tracking patterns:")
        for pattern in missing_patterns:
            print(f"  - {pattern}")
        print("\nFull output:")
        print(output)
        return False
    else:
        print("All expected patterns found and no failure patterns detected")
        return True


def main():
    parser = argparse.ArgumentParser(description="Test multithread example output")
    parser.add_argument("multithread_exe", help="Path to the multithread executable")
    parser.add_argument(
        "--config", help="Path to the configuration file (default: auto-determined based on mode)"
    )
    parser.add_argument(
        "--mode",
        choices=["port_map", "tracking"],
        default="port_map",
        help="Test mode: port_map or tracking (default: port_map)",
    )

    args = parser.parse_args()

    # Set default config file if not specified
    if args.config is None:
        exe_dir = os.path.dirname(args.multithread_exe)
        if args.mode == "port_map":
            args.config = os.path.join(exe_dir, "multithread_port_map.yaml")
        elif args.mode == "tracking":
            args.config = os.path.join(exe_dir, "multithread_tracking.yaml")

    # Validate the config file path
    if not os.path.isfile(args.config):
        print(f"Error: YAML config file not found: {args.config}")
        return 1

    # Validate the C++ application binary path
    if not os.path.isfile(args.multithread_exe):
        print(f"Error: C++ application binary not found: {args.multithread_exe}")
        return 1

    # Normalize the paths
    config_path = os.path.abspath(args.config)
    binary_path = os.path.abspath(args.multithread_exe)

    # Run the multithread example and capture output
    try:
        result = subprocess.run(
            [binary_path, config_path],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,  # 30 second timeout
        )
        output = result.stderr  # Holoscan logs go to stderr
        output += result.stdout  # Data flow tracking prints to stdout
    except subprocess.CalledProcessError as e:
        print(f"Failed to run multithread example: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return 1

    # Run the appropriate test based on mode
    if args.mode == "port_map":
        success = test_port_map(output)
    elif args.mode == "tracking":
        success = test_tracking_and_data_logging(output)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
