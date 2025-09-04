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
import os.path
import re
import subprocess
import sys

# Required patterns for tracking and data logging
EXPECTED_DATA_LOGGING_PATTERNS = [
    r"sum of received values: 496",
    r"BasicConsoleLogger\[ID:tx\.out\]",
    r"BasicConsoleLogger\[ID:delay15\.in\]",
    r"BasicConsoleLogger\[ID:delay15\.out_name\]",
    r"BasicConsoleLogger\[ID:rx\.values:\d+\]",
    r"Category:Message \(std::any\)",  # holoscan::Message type for int and string
    r"Python\(int\): 15",  # logged message contains the Python type and repr
    r"Python\(str\): 'delay15'",  # logged message contains the Python type and repr
]

# Failure patterns for data logging
FAILURE_PATTERNS_DATA_LOGGING = [
    r"BasicConsoleLogger\[ID:rx\.names:15\]",  # excluded by denylist
    r"BasicConsoleLogger\[ID:delay15\.out_val\]",  # excluded by denylist
]


def test_data_logging(output):
    """Test tracking output"""
    missing_patterns = []
    failure_patterns_found = []

    # Check for required patterns
    for pattern in EXPECTED_DATA_LOGGING_PATTERNS:
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
    parser.add_argument("multithread_py", help="Path to the multithread.py python script")
    parser.add_argument(
        "--mode",
        choices=["logging"],
        default="logging",
        help="Test mode: logging (default: logging)",
    )

    # use parse_known_args so additional arguments can be passed to multithread.py
    args, additional_args = parser.parse_known_args()

    # Validate the target script path
    if not os.path.isfile(args.multithread_py):
        print(f"Error: Script not found: {args.multithread_py}")
        return 1

    # Normalize the path
    script_path = os.path.abspath(args.multithread_py)

    # Build command line arguments for the multithread.py script
    cmd_args = ["python", script_path] + additional_args

    # Run the multithread example and capture output
    try:
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,  # 30 second timeout
        )
        output = result.stderr  # Holoscan logs go to stderr
        output += result.stdout  # Data flow tracking and Python print statements go to stdout
    except subprocess.CalledProcessError as e:
        print(f"Failed to run multithread example: {e}")
        print(f"Command: {' '.join(cmd_args)}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return 1

    # Run the appropriate test based on mode
    if args.mode == "logging":
        success = test_data_logging(output)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
