#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Google Benchmark CTest runner."""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).with_name("run_benchmark.py")


def make_fake_benchmark(tmp_path: Path) -> Path:
    benchmark = tmp_path / "fake_benchmark.py"
    benchmark.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

output = next((arg.removeprefix("--benchmark_out=") for arg in sys.argv
               if arg.startswith("--benchmark_out=")), os.environ.get("BENCHMARK_OUT"))
if output is None:
    raise SystemExit("missing benchmark output path")
Path(output).write_text(os.environ["FAKE_BENCHMARK_REPORT"], encoding="utf-8")
record_path = os.environ.get("FAKE_BENCHMARK_OUTPUT_PATH")
if record_path:
    Path(record_path).write_text(output, encoding="utf-8")
raise SystemExit(int(os.environ.get("FAKE_BENCHMARK_EXIT_CODE", "0")))
""",
        encoding="utf-8",
    )
    benchmark.chmod(benchmark.stat().st_mode | stat.S_IXUSR)
    return benchmark


def run_runner(
    benchmark: Path,
    report: dict,
    *,
    exit_code: int = 0,
    arguments: list[str] | None = None,
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    environment = {
        **os.environ,
        "FAKE_BENCHMARK_REPORT": json.dumps(report),
        "FAKE_BENCHMARK_EXIT_CODE": str(exit_code),
    }
    environment.pop("HOLOSCAN_BENCHMARK_ARGS", None)
    environment.pop("BENCHMARK_OUT", None)
    environment.pop("BENCHMARK_OUT_FORMAT", None)
    environment.update(extra_environment or {})
    return subprocess.run(
        [sys.executable, str(RUNNER), str(benchmark), *(arguments or [])],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )


def test_runner_accepts_successful_benchmark_report(tmp_path: Path) -> None:
    completed = run_runner(
        make_fake_benchmark(tmp_path),
        {"benchmarks": [{"name": "BM_Succeeds", "error_occurred": False}]},
    )

    assert completed.returncode == 0


def test_runner_ignores_parent_benchmark_output_environment(tmp_path: Path, monkeypatch) -> None:
    parent_output_path = tmp_path / "parent-benchmark.json"
    monkeypatch.setenv("BENCHMARK_OUT", str(parent_output_path))
    monkeypatch.setenv("BENCHMARK_OUT_FORMAT", "json")

    completed = run_runner(
        make_fake_benchmark(tmp_path),
        {"benchmarks": [{"name": "BM_Succeeds", "error_occurred": False}]},
    )

    assert completed.returncode == 0
    assert not parent_output_path.exists()


def test_runner_fails_when_google_benchmark_reports_an_error(tmp_path: Path) -> None:
    completed = run_runner(
        make_fake_benchmark(tmp_path),
        {
            "benchmarks": [
                {"name": "BM_Succeeds", "error_occurred": False},
                {"name": "BM_Fails", "error_occurred": True},
            ]
        },
    )

    assert completed.returncode == 1
    assert "BM_Fails" in completed.stderr


def test_runner_validates_a_caller_provided_json_report(tmp_path: Path) -> None:
    output_path = tmp_path / "benchmark.json"
    completed = run_runner(
        make_fake_benchmark(tmp_path),
        {"benchmarks": [{"name": "BM_Succeeds", "error_occurred": False}]},
        arguments=[f"--benchmark_out={output_path}", "--benchmark_out_format=json"],
    )

    assert completed.returncode == 0
    assert output_path.exists()


def test_runner_uses_benchmark_output_environment(tmp_path: Path) -> None:
    output_path = tmp_path / "benchmark.json"
    completed = run_runner(
        make_fake_benchmark(tmp_path),
        {"benchmarks": [{"name": "BM_Succeeds", "error_occurred": False}]},
        extra_environment={
            "BENCHMARK_OUT": str(output_path),
            "BENCHMARK_OUT_FORMAT": "json",
        },
    )

    assert completed.returncode == 0
    assert output_path.exists()


def test_runner_rejects_non_json_benchmark_output_environment(tmp_path: Path) -> None:
    completed = run_runner(
        make_fake_benchmark(tmp_path),
        {"benchmarks": []},
        extra_environment={
            "BENCHMARK_OUT": str(tmp_path / "benchmark.csv"),
            "BENCHMARK_OUT_FORMAT": "csv",
        },
    )

    assert completed.returncode == 2
    assert "requires --benchmark_out_format=json" in completed.stderr


def test_runner_rejects_stale_caller_provided_report(tmp_path: Path) -> None:
    output_path = tmp_path / "benchmark.json"
    stale_report = {"benchmarks": [{"name": "BM_Stale", "error_occurred": False}]}
    output_path.write_text(json.dumps(stale_report), encoding="utf-8")

    completed = run_runner(
        Path("/bin/true"),
        {"benchmarks": []},
        arguments=[f"--benchmark_out={output_path}", "--benchmark_out_format=json"],
    )

    assert completed.returncode == 1
    assert "valid JSON report" in completed.stderr
    assert json.loads(output_path.read_text(encoding="utf-8")) == stale_report


def test_runner_preserves_benchmark_process_failures(tmp_path: Path) -> None:
    completed = run_runner(
        make_fake_benchmark(tmp_path),
        {"benchmarks": [{"name": "BM_Succeeds", "error_occurred": False}]},
        exit_code=7,
    )

    assert completed.returncode == 7


def test_runner_preserves_existing_report_when_benchmark_process_fails(tmp_path: Path) -> None:
    output_path = tmp_path / "benchmark.json"
    stale_report = {"benchmarks": [{"name": "BM_Previous", "error_occurred": False}]}
    output_path.write_text(json.dumps(stale_report), encoding="utf-8")

    completed = run_runner(
        make_fake_benchmark(tmp_path),
        {"benchmarks": [{"name": "BM_Replacement", "error_occurred": False}]},
        exit_code=7,
        arguments=[f"--benchmark_out={output_path}", "--benchmark_out_format=json"],
    )

    assert completed.returncode == 7
    assert json.loads(output_path.read_text(encoding="utf-8")) == stale_report


def test_runner_rejects_an_invalid_benchmark_report(tmp_path: Path) -> None:
    benchmark = make_fake_benchmark(tmp_path)
    environment = {**os.environ, "FAKE_BENCHMARK_REPORT": "not-json"}
    environment.pop("HOLOSCAN_BENCHMARK_ARGS", None)
    environment.pop("BENCHMARK_OUT", None)
    environment.pop("BENCHMARK_OUT_FORMAT", None)

    completed = subprocess.run(
        [sys.executable, str(RUNNER), str(benchmark)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert completed.returncode == 1
    assert "valid JSON report" in completed.stderr


def test_runner_removes_its_temporary_json_report(tmp_path: Path) -> None:
    benchmark = make_fake_benchmark(tmp_path)
    output_record = tmp_path / "output-path"
    environment = {
        **os.environ,
        "FAKE_BENCHMARK_REPORT": json.dumps({"benchmarks": []}),
        "FAKE_BENCHMARK_OUTPUT_PATH": str(output_record),
    }
    environment.pop("HOLOSCAN_BENCHMARK_ARGS", None)
    environment.pop("BENCHMARK_OUT", None)
    environment.pop("BENCHMARK_OUT_FORMAT", None)

    completed = subprocess.run(
        [sys.executable, str(RUNNER), str(benchmark)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert completed.returncode == 0
    assert not Path(output_record.read_text(encoding="utf-8")).exists()
