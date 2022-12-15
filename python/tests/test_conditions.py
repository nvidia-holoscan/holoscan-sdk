# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from holoscan.conditions import (
    BooleanCondition,
    CountCondition,
    DownstreamMessageAffordableCondition,
    MessageAvailableCondition,
)
from holoscan.core import Condition
from holoscan.gxf import GXFCondition


class TestBooleanCondition:
    def test_kwarg_based_initialization(self, app, capfd):
        cond = BooleanCondition(
            fragment=app,
            name="boolean",
            enable_tick=True,
        )
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::BooleanSchedulingTerm"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_enable_tick(self, app, capfd):
        cond = BooleanCondition(
            fragment=app,
            name="boolean",
            enable_tick=True,
        )
        cond.disable_tick()
        assert not cond.check_tick_enabled()

        cond.enable_tick()
        assert cond.check_tick_enabled()

    def test_default_initialization(self, app):
        BooleanCondition(app)

    def test_positional_initialization(self, app):
        BooleanCondition(app, False, "bool")


class TestCountCondition:
    def test_kwarg_based_initialization(self, app, capfd):
        cond = CountCondition(
            fragment=app,
            name="count",
            count=100,
        )
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::CountSchedulingTerm"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_count(self, app, capfd):
        cond = CountCondition(
            fragment=app,
            name="count",
            count=100,
        )
        cond.count = 10
        assert cond.count == 10

    def test_default_initialization(self, app):
        CountCondition(app)

    def test_positional_initialization(self, app):
        CountCondition(app, 100, "counter")


class TestDownstreamMessageAffordableCondition:
    def test_kwarg_based_initialization(self, app, capfd):
        cond = DownstreamMessageAffordableCondition(
            fragment=app,
            name="downstream_affordable",
            min_size=10,
        )
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::DownstreamReceptiveSchedulingTerm"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_default_initialization(self, app):
        DownstreamMessageAffordableCondition(app)

    def test_positional_initialization(self, app):
        DownstreamMessageAffordableCondition(app, 4, "affordable")


class TestMessageAvailableCondition:
    def test_kwarg_based_initialization(self, app, capfd):
        cond = MessageAvailableCondition(
            fragment=app,
            name="message_available",
            min_size=1,
            front_stage_max_size=10,
        )
        assert isinstance(cond, GXFCondition)
        assert isinstance(cond, Condition)
        assert cond.gxf_typename == "nvidia::gxf::MessageAvailableSchedulingTerm"

        # assert no warnings or errors logged
        captured = capfd.readouterr()
        assert "error" not in captured.err
        assert "warning" not in captured.err

    def test_default_initialization(self, app):
        MessageAvailableCondition(app)

    def test_positional_initialization(self, app):
        MessageAvailableCondition(app, 1, 4, "available")
