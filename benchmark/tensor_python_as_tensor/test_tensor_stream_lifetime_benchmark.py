#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise benchmark completion control without importing CuPy or initializing CUDA."""

import argparse
import ast
import contextlib
import copy
import io
import math
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).with_name("tensor_stream_lifetime_benchmark.py")
TREE = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
WAIT = next(
    node
    for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "wait_for_delivery"
)
MAIN = next(node for node in TREE.body if isinstance(node, ast.FunctionDef) and node.name == "main")
# Execute the benchmark's argument checks, stopping before its CUDA availability probe.
PREFIX = []
for statement in MAIN.body:
    if isinstance(statement, ast.Try):
        break
    PREFIX.append(copy.deepcopy(statement))
PARSE = ast.FunctionDef(
    name="parse",
    args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
    body=PREFIX + [ast.Return(value=ast.Name(id="args", ctx=ast.Load()))],
    decorator_list=[],
)
CPU_CODE = compile(
    ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(WAIT), PARSE], type_ignores=[])),
    str(SOURCE),
    "exec",
)


class DeliveryWaitTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.waits = []
        self.namespace = {
            "argparse": argparse,
            "threading": threading,
            "Path": Path,
            "__doc__": "Completion-control tests",
            "time": SimpleNamespace(monotonic=lambda: self.now),
        }
        exec(CPU_CODE, self.namespace)

    def event(self, succeed=False):
        def wait(timeout):
            self.waits.append(timeout)
            if not succeed:
                self.now += timeout
            return succeed

        return SimpleNamespace(wait=wait)

    def parse(self, *arguments):
        with contextlib.redirect_stderr(io.StringIO()):
            saved = sys.argv
            try:
                sys.argv = [str(SOURCE), "--output", "unused.json", *arguments]
                return self.namespace["parse"]()
            finally:
                sys.argv = saved

    def test_completed_delivery_does_not_inspect_the_session(self):
        session = SimpleNamespace()
        assert self.namespace["wait_for_delivery"](self.event(True), session, 10)
        assert self.now == 0

    def test_session_failure_is_reported_before_the_delivery_deadline(self):
        session = SimpleNamespace(is_running=False, diagnostics=["forward: kBackpressured"])
        with pytest.raises(RuntimeError, match="forward: kBackpressured"):
            self.namespace["wait_for_delivery"](self.event(), session, 60)
        assert math.isclose(self.now, 0.1, abs_tol=1e-09)
        assert len(self.waits) == 1

    def test_delivery_timeout_uses_one_deadline(self):
        session = SimpleNamespace(is_running=True)
        assert not self.namespace["wait_for_delivery"](self.event(), session, 0.25)
        assert math.isclose(self.now, 0.25, abs_tol=1e-09)
        assert len(self.waits) == 3
        assert math.isclose(self.waits[-1], 0.05, abs_tol=1e-09)

    def test_session_stop_is_reported_without_waiting_for_missing_frames(self):
        session = SimpleNamespace(is_running=False, diagnostics=[])
        with pytest.raises(RuntimeError, match="Graph stopped before delivery completed"):
            self.namespace["wait_for_delivery"](self.event(), session, 60)
        assert math.isclose(self.now, 0.1, abs_tol=1e-09)

    def test_default_depth_and_deadlines_are_preserved(self):
        args = self.parse()
        assert args.queue_depth == 2
        assert (args.delivery_timeout_seconds, args.retirement_timeout_seconds) == (60, 10)

    def test_finite_smoke_capacity_and_deadlines_can_be_selected(self):
        args = self.parse(
            "--frames=40",
            "--warmup=10",
            "--queue-depth=50",
            "--delivery-timeout-seconds=10",
            "--retirement-timeout-seconds=10",
        )
        assert args.queue_depth == args.frames + args.warmup
        assert (args.delivery_timeout_seconds, args.retirement_timeout_seconds) == (10, 10)

    def test_invalid_depth_is_rejected_before_cuda_setup(self):
        for value in ("0", "-1"):
            with self.subTest(value=value), pytest.raises(SystemExit) as error:
                self.parse("--queue-depth=" + value)
            assert error.value.code == 2

    def test_invalid_deadlines_are_rejected_before_cuda_setup(self):
        for option in ("--delivery-timeout-seconds", "--retirement-timeout-seconds"):
            for value in ("0", "-1", "nan", "inf", "-inf", str(threading.TIMEOUT_MAX * 2)):
                with (
                    self.subTest(option=option, value=value),
                    pytest.raises(SystemExit) as error,
                ):
                    self.parse(option + "=" + value)
                assert error.value.code == 2

    def test_long_workload_deadlines_are_accepted(self):
        args = self.parse("--delivery-timeout-seconds=300", "--retirement-timeout-seconds=60")
        assert (args.delivery_timeout_seconds, args.retirement_timeout_seconds) == (300, 60)


if __name__ == "__main__":
    unittest.main()
