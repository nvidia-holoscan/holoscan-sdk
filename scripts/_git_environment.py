# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment isolation for commands operating on a different Git repository."""

import os
import subprocess


def git_environment() -> dict[str, str]:
    """Copy the environment without Git's repository-local overrides.

    Pass this to every subprocess working with the other repository, including
    shell scripts. For in-process fixture code that also launches Git, scope
    the copy with unittest.mock.patch.dict(os.environ, ..., clear=True).
    This does not isolate user-global Git configuration or credentials.
    """
    environment = os.environ.copy()
    # cwd and git -C do not override the repository paths exported by hooks.
    # https://git-scm.com/docs/githooks#_description
    local_variables = subprocess.run(
        ["git", "rev-parse", "--local-env-vars"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    for name in local_variables:
        environment.pop(name, None)
    return environment
