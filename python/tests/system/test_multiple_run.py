"""
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""  # noqa: E501

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.operators import PingRxOp

# Number of times the application will be run in the test
GLOBAL_RUN_COUNT = 2000
# The statistics count limit is based on experimental observations.
# This is an approximate threshold rather than a strict boundary, designed to monitor
# that Python object reference counts don't grow excessively.
# Local testing showed that even with GLOBAL_RUN_COUNT exceeding 10000, the 'count' values
# for all statistical items remain below 1000. Values up to 1058 were observed on CI test runs.
GLOBAL_STAT_COUNT_LIMIT = 1500


class PingTxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        self.index = 1
        # Need to call the base class constructor last
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: OperatorSpec):
        spec.output("out")
        spec.param("recess_period", 0)  # add parameter for test

    def compute(self, op_input, op_output, context):
        op_output.emit(self.index, "out")
        self.index += 1


class MyPingApp(Application):
    def compose(self):
        # Define the tx and rx operators, allowing tx to execute 10 times
        tx = PingTxOp(self, CountCondition(self, 10), name="tx")
        rx = PingRxOp(self, name="rx")

        # Define the workflow:  tx -> rx
        self.add_flow(tx, rx)


def test_multiple_run():
    import gc
    import tracemalloc

    tracemalloc.start()

    app = MyPingApp()
    for _ in range(GLOBAL_RUN_COUNT):
        app.run()
        gc.collect()

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    print("[ Top 10 differences ]")

    is_passed = True
    for stat in top_stats[:10]:
        # For each stat, check if 'count' part is less than `GLOBAL_STAT_COUNT_LIMIT`
        print(stat)
        if stat.count > GLOBAL_STAT_COUNT_LIMIT:
            print(f"warning: stat.count: {stat.count} exceeds the limit: {GLOBAL_STAT_COUNT_LIMIT}")
            print(f"warning: stat.traceback: {stat.traceback}")
            is_passed = False
    assert is_passed


def test_multiple_run_async():
    import gc
    import tracemalloc

    app = MyPingApp()
    for _ in range(GLOBAL_RUN_COUNT):
        future = app.run_async()
        future.result()
        gc.collect()

    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    print("[ Top 10 differences ]")

    is_passed = True
    for stat in top_stats[:10]:
        # For each stat, check if 'count' part is less than `GLOBAL_STAT_COUNT_LIMIT`
        print(stat)
        if stat.count > GLOBAL_STAT_COUNT_LIMIT:
            print(f"warning: stat.count: {stat.count} exceeds the limit: {GLOBAL_STAT_COUNT_LIMIT}")
            print(f"warning: stat.traceback: {stat.traceback}")
            is_passed = False
    assert is_passed


def main():
    import gc
    import tracemalloc

    tracemalloc.start()

    app = MyPingApp()
    for _ in range(1000):
        future = app.run_async()
        future.result()
        gc.collect()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics("lineno")

    print("[ Top 10 differences ]")

    is_passed = True
    for stat in top_stats[:10]:
        # For each stat, check if 'count' part is less than `GLOBAL_STAT_COUNT_LIMIT`
        print(stat)
        if stat.count > GLOBAL_STAT_COUNT_LIMIT:
            print(f"warning: stat.count: {stat.count} exceeds the limit: {GLOBAL_STAT_COUNT_LIMIT}")
            print(f"warning: stat.traceback: {stat.traceback}")
            is_passed = False
    assert is_passed


if __name__ == "__main__":
    main()
