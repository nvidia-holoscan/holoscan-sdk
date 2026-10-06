# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Narrow helpers shared by standalone Channel client tests."""

import uuid


def unique_channel(prefix):
    """Return a backend-safe name unique to this test process."""
    return f"{prefix}-{uuid.uuid4().hex}"
