# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("run_partition_planning_benchmark.py")
SPEC = importlib.util.spec_from_file_location("partition_planning_runner", MODULE_PATH)
assert SPEC
assert SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_checked_result_accepts_complete_probe(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        '{"schema":"holoscan.ea2.partition-planning-benchmark/v1",'
        '"topology":"linear","operator_count":16,'
        '"warmup_compile":"accepted_zero_work","compile_ns":10,'
        '"allocation_count":2,"cumulative_requested_bytes":30,'
        '"peak_live_requested_bytes":20,"live_requested_bytes_after_compile":12,'
        '"live_requested_bytes_after_plan_release":0,"process_max_rss_kib":100,'
        '"edge_count":15,"partition_count":1,"authored_partition_count":0,'
        '"automatic_partition_count":1,"authored_member_count":0,'
        '"automatic_member_count":16,"partitioning_mode":"automatic",'
        '"plan_digest":"' + "a" * 64 + '"}',
        encoding="utf-8",
    )
    assert runner.checked_result(path, "linear", 16)["partition_count"] == 1


def test_checked_result_rejects_wrong_identity(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid partition-planning result"):
        runner.checked_result(path, "linear", 16)


@pytest.mark.parametrize("partitioning_mode", [None, "manual", "unknown"])
def test_checked_result_rejects_wrong_partitioning_mode(
    tmp_path: Path, partitioning_mode: str | None
) -> None:
    path = tmp_path / "result.json"
    row = {
        "schema": runner.SCHEMA,
        "topology": "linear",
        "operator_count": 16,
        "warmup_compile": "accepted_zero_work",
        "compile_ns": 10,
        "allocation_count": 2,
        "cumulative_requested_bytes": 30,
        "peak_live_requested_bytes": 20,
        "live_requested_bytes_after_compile": 12,
        "live_requested_bytes_after_plan_release": 0,
        "process_max_rss_kib": 100,
        "edge_count": 15,
        "partition_count": 1,
        "authored_partition_count": 0,
        "automatic_partition_count": 1,
        "authored_member_count": 0,
        "automatic_member_count": 16,
        "plan_digest": "a" * 64,
    }
    if partitioning_mode is not None:
        row["partitioning_mode"] = partitioning_mode
    path.write_text(json.dumps(row), encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid partition-planning result"):
        runner.checked_result(path, "linear", 16)


def test_checked_result_accepts_constrained_automatic_hybrid_mode(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    row = {
        "schema": runner.SCHEMA,
        "topology": "hybrid",
        "operator_count": 4,
        "warmup_compile": "accepted_zero_work",
        "compile_ns": 10,
        "allocation_count": 2,
        "cumulative_requested_bytes": 30,
        "peak_live_requested_bytes": 20,
        "live_requested_bytes_after_compile": 12,
        "live_requested_bytes_after_plan_release": 0,
        "process_max_rss_kib": 100,
        "edge_count": 3,
        "partition_count": 2,
        "authored_partition_count": 1,
        "automatic_partition_count": 1,
        "authored_member_count": 1,
        "automatic_member_count": 3,
        "partitioning_mode": "constrained_automatic",
        "plan_digest": "a" * 64,
    }
    path.write_text(json.dumps(row), encoding="utf-8")

    assert runner.checked_result(path, "hybrid", 4)["partitioning_mode"] == (
        "constrained_automatic"
    )


def test_summarize_reports_order_statistics() -> None:
    assert runner.summarize([5, 1, 3, 2, 4]) == {
        "minimum": 1,
        "median": 3,
        "p95": 5,
        "p99": 5,
        "maximum": 5,
    }


def test_validate_args_accepts_same_binary_empty_baseline() -> None:
    runner.validate_args(
        type(
            "Args",
            (),
            {
                "topology": ["linear"],
                "operators": [0, 16],
                "repetitions": 1,
                "process_timeout_sec": 1.0,
                "source_snapshot_sha256": None,
            },
        )()
    )


def test_validate_args_rejects_values_beyond_uint32_admission_domain() -> None:
    with pytest.raises(ValueError, match="uint32 admission domain"):
        runner.validate_args(
            type(
                "Args",
                (),
                {
                    "topology": ["linear"],
                    "operators": [runner.MAX_OPERATOR_COUNT + 1],
                    "repetitions": 1,
                    "process_timeout_sec": 1.0,
                    "source_snapshot_sha256": None,
                },
            )()
        )


def test_validate_args_rejects_invalid_source_snapshot_digest() -> None:
    with pytest.raises(ValueError, match="source-snapshot-sha256"):
        runner.validate_args(
            type(
                "Args",
                (),
                {
                    "topology": ["linear"],
                    "operators": [16],
                    "repetitions": 1,
                    "process_timeout_sec": 1.0,
                    "source_snapshot_sha256": "not-a-digest",
                },
            )()
        )


@pytest.mark.parametrize("operators", [1, 2, 3])
def test_validate_args_rejects_hybrid_without_a_nonempty_relay_anchor(operators: int) -> None:
    with pytest.raises(ValueError, match="hybrid topology requires zero or at least four"):
        runner.validate_args(
            type(
                "Args",
                (),
                {
                    "topology": ["hybrid"],
                    "operators": [operators],
                    "repetitions": 1,
                    "process_timeout_sec": 1.0,
                    "source_snapshot_sha256": None,
                },
            )()
        )


def test_build_metadata_reads_nearest_cmake_cache(tmp_path: Path) -> None:
    binary = tmp_path / "benchmark" / "graph" / "partition_planning_benchmark"
    binary.parent.mkdir(parents=True)
    binary.touch()
    (tmp_path / "CMakeCache.txt").write_text(
        "CMAKE_BUILD_TYPE:STRING=Release\n"
        "CMAKE_CXX_COMPILER_ID:STRING=GNU\n"
        "UNRELATED:STRING=ignored\n",
        encoding="utf-8",
    )

    assert runner.build_metadata(binary) == {
        "cmake_cache": str(tmp_path / "CMakeCache.txt"),
        "CMAKE_BUILD_TYPE": "Release",
        "CMAKE_CXX_COMPILER_ID": "GNU",
    }


@pytest.mark.parametrize("failure_kind", ["unavailable", "timeout"])
def test_build_metadata_treats_unavailable_compiler_version_as_optional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_kind: str
) -> None:
    binary = tmp_path / "benchmark" / "graph" / "partition_planning_benchmark"
    binary.parent.mkdir(parents=True)
    binary.touch()
    (tmp_path / "CMakeCache.txt").write_text(
        "CMAKE_CXX_COMPILER:FILEPATH=/missing/compiler\n", encoding="utf-8"
    )

    def unavailable(*args: object, **kwargs: object) -> None:
        assert kwargs["timeout"] == 5
        if failure_kind == "timeout":
            raise runner.subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])
        raise OSError("compiler disappeared")

    monkeypatch.setattr(runner.subprocess, "run", unavailable)

    assert runner.build_metadata(binary) == {
        "cmake_cache": str(tmp_path / "CMakeCache.txt"),
        "CMAKE_CXX_COMPILER": "/missing/compiler",
        "compiler_version_output": None,
    }


def test_file_sha256_identifies_exact_benchmark_binary(tmp_path: Path) -> None:
    binary = tmp_path / "probe"
    binary.write_bytes(b"partition-probe")

    assert runner.file_sha256(binary) == (
        "f4e0b706e8bf75ab46f9ef5b36d95c6ab1d2d823296a183c8a9970aa069a8d03"
    )


def test_run_case_preserves_command_and_partial_output_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def timed_out(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(
            cmd=["partition-probe"], timeout=3.0, output="partial stdout", stderr=b"partial stderr"
        )

    monkeypatch.setattr(runner.subprocess, "run", timed_out)

    with pytest.raises(RuntimeError, match="partition-planning probe timed out"):
        runner.run_case(tmp_path / "probe", "linear", 4, 2, tmp_path, 3.0)

    case_dir = tmp_path / "linear_4" / "repetition_002"
    assert json.loads((case_dir / "command.json").read_text(encoding="utf-8"))[0] == str(
        tmp_path / "probe"
    )
    assert (case_dir / "stdout.log").read_text(encoding="utf-8") == "partial stdout"
    assert (case_dir / "stderr.log").read_text(encoding="utf-8") == "partial stderr"
