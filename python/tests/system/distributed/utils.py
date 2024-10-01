"""
SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
