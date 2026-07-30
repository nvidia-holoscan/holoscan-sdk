"""
SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
"""  # noqa: E501

import os
import warnings
from contextlib import contextmanager

from holoscan.logger import log_level, set_log_level


@contextmanager
def env_var_context(env_var_settings=None):
    """Context manager for temporarily setting environment variables.

    Example usage:

    >>> env_var_settings = {
    ...     ("HOLOSCAN_CHECK_RECESSION_PERIOD_MS", "10"),
    ...     ("HOLOSCAN_MAX_DURATION_MS", "10000"),
    ...     ("HOLOSCAN_UCX_NETWORK_CONNECTION_TIMEOUT", "10000"),
    ...     ("HOLOSCAN_STOP_ON_DEADLOCK_TIMEOUT", "1000")
    ... }
    >>> with env_var_context(env_var_settings):
    ...     app = DistributedVideoReplayerApp()
    ...     app.run()
    """
    if env_var_settings is None:
        env_var_settings = {}

    # remember existing environment variables
    orig_env_vars = {env_var: os.environ.get(env_var) for env_var, _ in env_var_settings}
    orig_log_level = log_level()
    # Warning: The GXF logger level cannot be accessed or modified through Python APIs. If
    # 'HOLOSCAN_EXECUTOR_LOG_LEVEL' is modified here, we won't be able to restore it to its
    # original value when exiting this context.
    for env_var, value in env_var_settings:
        if env_var == "HOLOSCAN_EXECUTOR_LOG_LEVEL":
            # Skip setting GXF logger level as it's not accessible via Python APIs
            warnings.warn(
                "Cannot set HOLOSCAN_EXECUTOR_LOG_LEVEL through this context manager. "
                "The GXF logger level is not accessible through Python APIs, making it "
                "impossible to restore its original value when exiting this context.",
                stacklevel=2,
            )
            continue
        os.environ[env_var] = value
    try:
        yield
    finally:
        # restore the environment variable
        for env_var, _ in env_var_settings:
            orig_env = orig_env_vars.get(env_var)
            if orig_env is None:
                os.environ.pop(env_var, None)
            else:
                os.environ[env_var] = orig_env
        set_log_level(orig_log_level)
