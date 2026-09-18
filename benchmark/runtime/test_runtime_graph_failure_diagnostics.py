# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        raise RuntimeError("expected the runtime_graph_benchmark path")

    with tempfile.TemporaryDirectory(prefix="runtime-graph-failure-diagnostics-") as directory:
        result_path = Path(directory) / "result.json"
        completed = subprocess.run(
            [
                sys.argv[1],
                "--workload",
                "latency",
                "--scenario",
                "sample_forward",
                "--relay-count",
                "1",
                "--samples",
                "8",
                "--warmup",
                "0",
                "--publisher-period-us",
                "1",
                "--queue-depth",
                "1",
                "--notification-capacity",
                "1",
                "--payload-validation",
                "none",
                "--test-fail-sink-after-relay-pressure",
                "--timeout-ms",
                "1000",
                "--output",
                str(result_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

        if completed.returncode == 0:
            raise RuntimeError("the injected callback failure unexpectedly succeeded")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        session = result["session"]
        if session["termination"] != "failed":
            raise RuntimeError("the injected callback failure did not fail the RunSession")

        sink = next(
            (
                diagnostic
                for diagnostic in session["diagnostics"]
                if diagnostic.get("detail", {}).get("operator_path") == "sink"
            ),
            None,
        )
        if sink is None:
            raise RuntimeError("the result omitted the sink callback diagnostic")
        if any(
            diagnostic.get("token") == "RUNTIME_EXECUTION_LANE_PLAN_INVALID"
            for diagnostic in session["diagnostics"]
        ):
            raise RuntimeError("the callback failure was mislabeled as an invalid lane plan")
        if sink["id"] <= 0 or sink["severity"] != "error":
            raise RuntimeError("the sink diagnostic omitted its stable identity or severity")
        if sink["token"] != "RUNTIME_OPERATOR_CALLBACK_FAILED":
            raise RuntimeError("the sink diagnostic reported an unexpected token")
        if sink["source"]["graph_generation"] <= 0 or sink["source"]["vertex"] <= 0:
            raise RuntimeError("the sink diagnostic omitted its graph source coordinates")

        detail = sink["detail"]
        if detail["operation"] != "compute" or detail["role"] != "primary":
            raise RuntimeError("the sink diagnostic omitted its callback operation or role")
        if detail["error"] != {
            "code": "kFailure",
            "numeric_code": 1,
            "message": "intentional sink failure after relay pressure",
        }:
            raise RuntimeError("the sink diagnostic did not preserve the original Error")
        if detail["truncated"] is not False:
            raise RuntimeError("the complete sink diagnostic was incorrectly marked truncated")

        stderr = completed.stderr
        if not all(
            field in stderr
            for field in (
                "error RUNTIME_OPERATOR_CALLBACK_FAILED",
                "operator=sink",
                "operation=kCompute",
                "cause=kFailure: intentional sink failure after relay pressure",
            )
        ):
            raise RuntimeError(f"stderr omitted the formatted callback evidence:\n{stderr}")
        if "RUNTIME_EXECUTION_LANE_PLAN_INVALID" in stderr:
            raise RuntimeError("stderr included the obsolete invalid-lane-plan fallback")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
