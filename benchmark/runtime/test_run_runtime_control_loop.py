# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Result-contract, ordering, and summary checks for the control-loop benchmark runner."""

import json
import os
import subprocess

import pytest
from run_runtime_control_loop import (
    deltas,
    load_arms,
    markdown,
    round_orders,
    summarize_arm,
    validate_result,
)


def make_result(p50, samples=8):
    timing = {
        "samples": samples,
        "p50": p50,
        "p90": p50 + 1,
        "p99": p50 + 2,
        "p999": None,
        "p9999": None,
        "max": p50 + 3,
    }
    return {
        "schema": "holoscan.control_loop.v1",
        "valid": True,
        "diagnostics": [],
        "samples": samples,
        "warmup": 2,
        "counts": {"measured_ticks": samples, "missing_ticks": 0, "late": 1, "coalesced": 0},
        "kpis_ns": {
            "control_loop_latency": dict(timing),
            "active_path": dict(timing),
            "entry_latency": dict(timing),
        },
        "phase_counters": {
            "enabled": True,
            "wait_calls": 20,
            "wake_signals": 50,
            "partitions": [
                {"index": 0, "owner_tid": 7, "role": "control", "waits": 10},
                {"index": 1, "owner_tid": 8, "role": "target", "waits": 4},
            ],
        },
    }


def test_valid_result_passes():
    validate_result(make_result(1000))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("valid",), False, "invalid run"),
        (("diagnostics",), [{"token": "X"}], "diagnostics"),
        (("counts", "missing_ticks"), 1, "tick count"),
        (("kpis_ns", "active_path", "samples"), 7, "active_path"),
    ],
)
def test_invalid_result_is_rejected(path, value, message):
    result = make_result(1000)
    target = result
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_result(result)


def test_p9999_requires_100000_samples():
    result = make_result(1000)
    result["kpis_ns"]["control_loop_latency"]["p9999"] = 1
    with pytest.raises(ValueError, match="P99.99"):
        validate_result(result)


def test_every_round_runs_every_arm_once_reproducibly():
    orders = round_orders(["a", "b", "c"], 6, seed=3)
    assert len(orders) == 6
    assert all(sorted(order) == ["a", "b", "c"] for order in orders)
    assert orders == round_orders(["a", "b", "c"], 6, seed=3)


def test_arms_file_requires_unique_names_and_runner_owned_output(tmp_path):
    path = tmp_path / "arms.json"
    path.write_text(
        json.dumps({"arms": [{"name": "a", "binary": "x"}, {"name": "a", "binary": "y"}]})
    )
    with pytest.raises(ValueError, match="unique"):
        load_arms(path)
    for args in (["--output", "o"], ["--output=o"]):
        path.write_text(json.dumps({"arms": [{"name": "a", "binary": "x", "args": args}]}))
        with pytest.raises(ValueError, match="--output"):
            load_arms(path)
    malformed = (("args", "--samples 5"), ("args", [5]), ("env", ["A=1"]), ("env", {"A": 1}))
    for field, value in malformed:
        path.write_text(json.dumps({"arms": [{"name": "a", "binary": "x", field: value}]}))
        with pytest.raises(ValueError, match="string"):
            load_arms(path)
    path.write_text(json.dumps({"reference": "b", "arms": [{"name": "a", "binary": "x"}]}))
    with pytest.raises(ValueError, match="reference"):
        load_arms(path)
    path.write_text(json.dumps({"arms": [{"name": "a", "binary": "x"}]}))
    assert load_arms(path) == ("a", [{"name": "a", "binary": "x", "args": [], "env": {}}])


def test_summary_medians_ranges_counts_and_deltas():
    reference = summarize_arm([make_result(1000), make_result(1200), make_result(1100)])
    faster = summarize_arm([make_result(500), make_result(700)])
    loop = reference["kpis_ns"]["control_loop_latency"]["p50"]
    assert loop == {"median": 1100, "min": 1000, "max": 1200}
    assert "p999" not in reference["kpis_ns"]["control_loop_latency"]
    assert reference["counts"]["late"] == 3
    assert reference["phase_counters_per_tick"]["wait_calls"] == pytest.approx(60 / 30)
    assert "partitions" not in reference["phase_counters_per_tick"]
    assert reference["partition_counters_per_tick"]["control"] == {"waits": pytest.approx(1)}
    assert reference["partition_counters_per_tick"]["target"] == {"waits": pytest.approx(12 / 30)}
    result = deltas({"ref": reference, "fast": faster}, "ref")
    assert result["fast"]["control_loop_latency.p50"]["delta_ns"] == -500
    assert result["fast"]["control_loop_latency.p50"]["ratio"] == pytest.approx(600 / 1100)
    table = markdown({"ref": reference, "fast": faster}, "ref")
    assert "| ref (reference) | 1.10 |" in table


# The benchmark binary, set by the CTest row that runs these tests; the runner-only tests above do
# not need it.
BENCHMARK = os.environ.get("CONTROL_LOOP_BENCHMARK")


@pytest.mark.skipif(not BENCHMARK, reason="CONTROL_LOOP_BENCHMARK names the benchmark binary")
@pytest.mark.parametrize(
    "operand",
    ['quote"d', "back\\slash", "tab\there", "line\nbreak", "bell\x07"],
)
def test_authored_guard_text_stays_valid_json(operand):
    guard = f"plan.joint_state!={operand}"
    completed = subprocess.run(
        [BENCHMARK, "--mode", "split", "--topology", "rcb", "--plan-only", "--expect", guard],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["guards"]["passed"] is True
    assert result["guards"]["results"][0]["expect"] == guard


@pytest.mark.skipif(not BENCHMARK, reason="CONTROL_LOOP_BENCHMARK names the benchmark binary")
def test_failed_plan_guard_exits_one():
    # The CTest guard self-test matches the failing guard's output but ignores the exit status.
    completed = subprocess.run(
        [
            BENCHMARK,
            "--mode",
            "split",
            "--topology",
            "rcb",
            "--plan-only",
            "--expect",
            "plan.joint_state==drop_oldest:pump",
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 1, completed.stderr
    result = json.loads(completed.stdout)
    assert result["guards"]["passed"] is False
    assert result["guards"]["results"][0]["actual"] == "drop_oldest:direct"
