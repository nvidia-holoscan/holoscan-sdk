# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-level ownership boundaries cannot be established by in-process mocks."""

import subprocess
import sys
import textwrap

from observation import OBSERVATION_MS, OBSERVATION_SECONDS

PREAMBLE = """
import gc
import os
import threading
import holoscan as hs
from holoscan.testing import OperatorTestHarness, TestRunOptions, FrameRelease
class Echo(hs.Operator):
    inp = hs.Input('in', hs.payload.uint32)
    out = hs.Output('out', hs.payload.uint32)
    def compute(self, context):
        self.out.emit(self.inp.receive_data())
"""


def run_child(script, timeout=15):
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "ignore::DeprecationWarning",
            "-c",
            PREAMBLE
            + f"\nOBSERVATION_MS = {OBSERVATION_MS}\nOBSERVATION_SECONDS = {OBSERVATION_SECONDS}\n"
            + textwrap.dedent(script),
        ],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


# REQ: PY-API-031
def test_forked_owner_rejects_use_and_does_not_join_parent_work():
    run_child("""
        h = OperatorTestHarness(Echo)
        h.push_frame({'in': 1})
        running = h.run_async(TestRunOptions(frames=FrameRelease.MANUAL, expect_stop=True))
        child = os.fork()
        if child == 0:
            try:
                running.finish()
            except hs.HoloscanError as error:
                assert 'PYTHON_TEST_FIXTURE_FORKED_PROCESS' in str(error)
                assert isinstance(error, hs.errors.InvalidArgumentError)
                assert 'parent process' in str(error)
            else:
                os._exit(2)
            del running
            gc.collect()
            os._exit(0)
        _, status = os.waitpid(child, 0)
        assert status == 0
        running.close()
        assert running.finish().ok
        h.close()
        del running, h
    """)


# REQ: PY-API-031
def test_shutdown_of_deferred_noncooperating_work_is_bounded():
    result = run_child(
        """
        entered = threading.Event()
        blocked = threading.Event()
        owners = []
        class Noncooperating(Echo):
            def compute(self, context):
                owners.pop()
                gc.collect()
                entered.set()
                blocked.wait()
        h = OperatorTestHarness(Noncooperating)
        h.push_frame({'in': 1})
        owners.append(h.run_async(TestRunOptions(frames=FrameRelease.MANUAL)))
        owners[0].release_frame(timeout_ms=OBSERVATION_MS)
        assert entered.wait(OBSERVATION_SECONDS)
        h.close()
        print('shutdown reached', flush=True)
    """,
        timeout=15,
    )
    assert "shutdown reached" in result.stdout
    assert "REAPER_EXIT_DRAIN_TIMEOUT" in result.stderr + result.stdout
