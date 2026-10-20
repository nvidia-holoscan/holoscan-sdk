# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, topology-independent helpers for Runtime Graph benchmark runners."""

from __future__ import annotations

import argparse
import statistics


def parse_integer_list(
    value: str,
    *,
    invalid_message: str,
    empty_message: str,
) -> list[int]:
    """Parse a comma-delimited integer list while preserving caller-specific errors."""
    items = value.split(",")
    if not items or any(not item for item in items):
        raise argparse.ArgumentTypeError(empty_message)
    try:
        return [int(item) for item in items]
    except ValueError as error:
        raise argparse.ArgumentTypeError(invalid_message) from error


def summarize_values(values: list[float], *, median_key: str) -> dict[str, float | int | None]:
    """Summarize repeated measurements using a caller-selected median field name."""
    mean = statistics.fmean(values)
    return {
        "count": len(values),
        median_key: statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": mean,
        "coefficient_of_variation": (
            statistics.stdev(values) / mean if len(values) > 1 and mean != 0.0 else None
        ),
    }
