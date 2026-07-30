"""
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501


def remove_ignored_errors(captured_error: str) -> str:
    """Utility to remove specific, known errors from a captured stderr string"""

    err_lines = captured_error.split("\n")

    errors_to_ignore = [
        # some versions of the UCX extension print this error during application shutdown
        "Connection dropped with status -25",
    ]

    for err in errors_to_ignore:
        err_lines = [line for line in err_lines if err not in line]

    return "\n".join(err_lines)
