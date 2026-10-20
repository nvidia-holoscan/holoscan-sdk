# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scheduling allowances for observable progress; not operator timeout contracts."""

OBSERVATION_SECONDS = 10.0
OBSERVATION_MS = 10_000
# Held callbacks allow the controller several observations before releasing them.
CALLBACK_RELEASE_SECONDS = 3 * OBSERVATION_SECONDS
