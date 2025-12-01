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

See ArgumentParser epilog in main for usage examples.
"""  # noqa: E501

import argparse
import re
import sys
from pathlib import Path


def validate_log_patterns(
    log_file_path=None, pass_patterns=None, fail_patterns=None, verbose=False, strip_quotes=False
):
    """
    Validate log file or stdin against multiple pass and fail patterns.

    Args:
        log_file_path: Path to the log file to validate, or None to read from stdin
        pass_patterns: List of regex patterns that should be found (all must match)
        fail_patterns: List of regex patterns that should NOT be found (none should match)
        verbose: Print additional debug information
        strip_quotes: Remove all single and double quotes from log content before pattern matching

    Returns:
        bool: True if validation passes, False otherwise
    """
    # Read from stdin if no file path is provided
    if log_file_path is None:
        try:
            # Read stdin as bytes first to handle encoding issues
            stdin_bytes = sys.stdin.buffer.read()
            # Try to decode with UTF-8, replacing invalid sequences
            log_content = stdin_bytes.decode("utf-8", errors="replace")

            # Check for replacement characters that indicate binary/invalid UTF-8
            if "\ufffd" in log_content:  # Unicode replacement character
                replacement_count = log_content.count("\ufffd")
                print(f"WARNING: Found {replacement_count} invalid UTF-8 byte(s) in stdin")

            if verbose:
                print(f"Read from stdin ({len(log_content)} characters)")
        except Exception as e:
            print(f"ERROR: Failed to read from stdin: {e}")
            return False
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

    # Strip quotes if requested
    if strip_quotes:
        original_length = len(log_content)
        log_content = log_content.replace('"', "").replace("'", "")
        if verbose:
            print(f"Stripped quotes: {original_length} -> {len(log_content)} characters")

    success = True
    # Check fail patterns first - any match means failure
    if fail_patterns:
        for pattern in fail_patterns:
            match = re.search(pattern, log_content, re.MULTILINE)
            if match:
                success = False
                print(f"FAIL: Found prohibited pattern: '{pattern}'")
                print(f"  Match: {match.group()}")
                print("\n" + "=" * 80)
                # Show context around the match
                start = max(0, match.start() - 20)
                end = min(len(log_content), match.end() + 20)
                context = log_content[start:end].replace("\n", "\\n")
                print("LOCAL LOG CONTEXT AROUND THE FAILURE:\n\n")
                print(context)

    # Check pass patterns - all must match
    if pass_patterns:
        for pattern in pass_patterns:
            match = re.search(pattern, log_content, re.MULTILINE)
            if not match:
                success = False
                print(f"FAIL: Required pattern not found: '{pattern}'")
            elif verbose:
                print(f"  Found required pattern: '{pattern}' -> {match.group()}")

    if not success:
        print("\n" + "=" * 80)
        print("FULL LOG CONTENT:\n\n")
        print(log_content)
        print("=" * 80 + "\n")
        return False

    if fail_patterns:
        print(f"PASS: No prohibited patterns found ({len(fail_patterns)} patterns checked)")
    if pass_patterns:
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

  # Strip quotes from log content before pattern matching
  python3 my_app.py | %(prog)s --strip-quotes \\
    --pass-pattern 'messages received on in1: [tx1, tx1, tx1, tx1]'

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
    parser.add_argument(
        "--strip-quotes",
        action="store_true",
        help="Remove all single and double quotes from log content before pattern matching",
    )

    args = parser.parse_args()

    if not args.pass_patterns and not args.fail_patterns:
        print("ERROR: At least one --pass-pattern or --fail-pattern must be specified")
        return 1

    success = validate_log_patterns(
        args.log_file, args.pass_patterns, args.fail_patterns, args.verbose, args.strip_quotes
    )

    if success:
        print("SUCCESS: All pattern validations passed")
        return 0
    else:
        print("FAILURE: Pattern validation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
