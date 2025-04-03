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

import datetime

from holoscan.conditions import CountCondition, PeriodicCondition
from holoscan.core import Application
from holoscan.operators import PingRxOp, PingTxOp
from holoscan.resources import OrConditionCombiner


class MyConditionCombinerApp(Application):
    def __init__(
        self,
        *args,
        count1=10,
        count2=20,
        period=None,
        **kwargs,
    ):
        self.count1 = count1
        self.count2 = count2
        self.period = period
        super().__init__(*args, **kwargs)

    def compose(self):
        count1 = CountCondition(self, self.count1, name="count1")
        count2 = CountCondition(self, self.count2, name="count2")
        or_combiner = OrConditionCombiner(fragment=self, terms=[count1, count2])
        tx = PingTxOp(
            self,
            or_combiner,
            PeriodicCondition(self, recess_period=self.period),
            name="tx",
        )

        rx = PingRxOp(self, name="rx")
        self.add_flow(tx, rx)


def test_condition_combiner_app(capfd):
    count1 = 10
    count2 = 20
    app = MyConditionCombinerApp(
        count1=count1,
        count2=count2,
        period=datetime.timedelta(seconds=0.1),
    )
    app.run()

    # assert that the expected number of messages were received
    captured = capfd.readouterr()

    # will stop when either CountCondition reaches NEVER state
    min_count = min(count1, count2)
    assert f"Rx message value: {min_count}" in captured.out
    assert f"Rx message value: {min_count + 1}" not in captured.out
