#!/usr/bin/env python3
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

--------------------

Test Pattern Validation Utility

This script validates multiple regular expressions against a log file.
It allows running an application once with output captured to a log file,
then validating multiple pass/fail patterns from that single log file.

This is a workaround for the CTest limitation that any PASS_REGULAR_EXPRESSION
match will mark the entire test as passed, while we typically want to require
all expressions are matched.

Usage Examples:
    # Basic usage with log file
    python3 test_pattern_validation.py --log-file app.log --pass-pattern "Success" --fail-pattern "Error"

    # Using stdin (pipe output directly) - default behavior
    python3 my_app.py | python3 test_pattern_validation.py --pass-pattern "Success"

    # Multiple patterns with log file
    python3 test_pattern_validation.py --log-file app.log \
        --pass-pattern "Connection established" \
        --pass-pattern "Data processed" \
        --fail-pattern "FATAL:" \
        --fail-pattern "initialized independent of a parent entity"

    # Only check for failure patterns from stdin
    some_command | python3 test_pattern_validation.py --fail-pattern "ERROR:"
"""  # noqa: E501

import argparse
import re
import sys
from pathlib import Path


def validate_log_patterns(
    log_file_path=None, pass_patterns=None, fail_patterns=None, verbose=False
):
    """
    Validate log file or stdin against multiple pass and fail patterns.

    Args:
        log_file_path: Path to the log file to validate, or None to read from stdin
        pass_patterns: List of regex patterns that should be found (all must match)
        fail_patterns: List of regex patterns that should NOT be found (none should match)
        verbose: Print additional debug information

    Returns:
        bool: True if validation passes, False otherwise
    """
    # Read from stdin if no file path is provided
    if log_file_path is None:
        try:
            log_content = sys.stdin.read()
        except Exception as e:
            print(f"ERROR: Failed to read from stdin: {e}")
            return False
        if verbose:
            print(f"Read from stdin ({len(log_content)} characters)")
    else:
        if not Path(log_file_path).exists():
            print(f"ERROR: Log file {log_file_path} does not exist")
            return False

        # Read the entire log file
        try:
            with open(log_file_path, encoding="utf-8", errors="ignore") as f:
                log_content = f.read()
        except Exception as e:
            print(f"ERROR: Failed to read log file {log_file_path}: {e}")
            return False

        if verbose:
            print(f"Read log file: {log_file_path} ({len(log_content)} characters)")

    # Check fail patterns first - any match means failure
    if fail_patterns:
        for pattern in fail_patterns:
            match = re.search(pattern, log_content, re.MULTILINE)
            if match:
                print(f"FAIL: Found prohibited pattern: '{pattern}'")
                if verbose:
                    print(f"  Match: {match.group()}")
                    # Show context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(log_content), match.end() + 50)
                    context = log_content[start:end].replace("\n", "\\n")
                    print(f"  Context: ...{context}...")
                return False

        print(f"PASS: No prohibited patterns found ({len(fail_patterns)} patterns checked)")

    # Check pass patterns - all must match
    if pass_patterns:
        for pattern in pass_patterns:
            match = re.search(pattern, log_content, re.MULTILINE)
            if not match:
                print(f"FAIL: Required pattern not found: '{pattern}'")
                if verbose:
                    print(f"  Searched in {len(log_content)} characters")
                return False
            elif verbose:
                print(f"  Found required pattern: '{pattern}' -> {match.group()}")

        print(f"PASS: All required patterns found ({len(pass_patterns)} patterns checked)")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate multiple regex patterns against a log file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check for success and failure patterns from log file
  %(prog)s --log-file app.log --pass-pattern "Success" --fail-pattern "Error"

  # Read from stdin (pipe application output directly) - default behavior
  python3 my_app.py | %(prog)s --pass-pattern "Success"

  # Multiple patterns of each type. Must match all patterns to pass.
  %(prog)s --log-file app.log \\
    --pass-pattern "Connection established" \\
    --pass-pattern "Data processed" \\
    --fail-pattern "FATAL:" \\
    --fail-pattern "ERROR:"

  # Only check for failure patterns from stdin
  some_command | %(prog)s --fail-pattern "initialized independent of a parent entity"
        """,
    )

    parser.add_argument(
        "--log-file",
        dest="log_file",
        help="Path to the log file to validate (if not provided, reads from stdin)",
    )
    parser.add_argument(
        "--pass-pattern",
        action="append",
        dest="pass_patterns",
        help="Regex pattern that must be found (can be specified multiple times)",
    )
    parser.add_argument(
        "--fail-pattern",
        action="append",
        dest="fail_patterns",
        help="Regex pattern that must NOT be found (can be specified multiple times)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed information about pattern matching",
    )

    args = parser.parse_args()

    if not args.pass_patterns and not args.fail_patterns:
        print("ERROR: At least one --pass-pattern or --fail-pattern must be specified")
        return 1

    success = validate_log_patterns(
        args.log_file, args.pass_patterns, args.fail_patterns, args.verbose
    )

    if success:
        print("SUCCESS: All pattern validations passed")
        return 0
    else:
        print("FAILURE: Pattern validation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
