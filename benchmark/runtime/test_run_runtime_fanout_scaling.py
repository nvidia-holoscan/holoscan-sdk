# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Result-contract checks for the concurrent producer benchmark runner."""

import json

import pytest
from run_runtime_fanout_scaling import source_state, validate_result


@pytest.fixture
def result():
    timing = {"samples": 8, "p50": 100, "max": 200, "p9999": None}
    return {
        "schema": "holoscan.fanout_scaling.v1",
        "valid": True,
        "completed": True,
        "diagnostics": [],
        "producer_count": 2,
        "fanout": 3,
        "samples": 8,
        "warmup": 2,
        "shared_destinations": True,
        "lane_startup": [{}] * 5,
        "measured_window_ns": 1000,
        "publications_per_second": 16e6,
        "producers": [
            {
                "index": index,
                "publications": 10,
                "latency_ns": dict(timing),
                "emit_ns": dict(timing),
            }
            for index in range(2)
        ],
        "sinks": [{"index": index, "deliveries": 10} for index in range(6)],
        "phase_counters": {"enabled": True, "cuts": 80, "cut_drains": 20, "cut_skips": 60},
    }


def test_accepts_complete_shared_destinations(result):
    validate_result(result)


@pytest.mark.parametrize("field", ["valid", "completed"])
def test_rejects_failed_or_stopped_session(result, field):
    result[field] = False
    with pytest.raises(ValueError, match="complete naturally"):
        validate_result(result)


def test_rejects_missing_recipient_tail(result):
    result["sinks"][-1]["deliveries"] -= 1
    with pytest.raises(ValueError, match="sink delivery"):
        validate_result(result)


def test_accepts_a_zero_median_from_one_clock_tick(result):
    result["producers"][0]["emit_ns"]["p50"] = 0
    validate_result(result)


def test_rejects_truncated_timing(result):
    result["producers"][1]["latency_ns"]["samples"] -= 1
    with pytest.raises(ValueError, match="timing samples"):
        validate_result(result)


def test_rejects_p9999_without_enough_samples(result):
    result["producers"][0]["emit_ns"]["p9999"] = 180
    with pytest.raises(ValueError, match="P99.99"):
        validate_result(result)


def test_rejects_inconsistent_cut_counts(result):
    result["phase_counters"]["cut_skips"] += 1
    with pytest.raises(ValueError, match="activation-cut"):
        validate_result(result)


def test_container_source_state(tmp_path):
    state = {"revision": "abc123", "tracked_changes": " M benchmark/source.cpp\n"}
    path = tmp_path / "source.json"
    path.write_text(json.dumps(state))
    assert source_state(path) == state


def test_source_state_requires_dirty_status(tmp_path):
    path = tmp_path / "source.json"
    path.write_text(json.dumps({"revision": "abc123"}))
    with pytest.raises(ValueError, match="tracked_changes"):
        source_state(path)
