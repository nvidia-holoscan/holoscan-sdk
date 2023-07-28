# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from holoscan.core import ComponentSpec, Scheduler
from holoscan.gxf import GXFScheduler
from holoscan.resources import ManualClock, RealtimeClock
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler


class TestGreedyScheduler:
    def test_default_init(self, app):
        e = GreedyScheduler(app)
        assert isinstance(e, GXFScheduler)
        assert isinstance(e, Scheduler)
        assert isinstance(e.spec, ComponentSpec)
        assert "<uninitialized>" in repr(e)
        assert repr(e).startswith("GreedyScheduler(self, clock=")

    @pytest.mark.parametrize("ClockClass", [ManualClock, RealtimeClock])
    def test_init_kwargs(self, app, ClockClass):
        e = GreedyScheduler(
            app,
            clock=ClockClass(app),
            stop_on_deadlock=True,
            max_duration_ms=10000,
            check_recession_period_ms=1.0,
            stop_on_deadlock_timeout=-1,
            name="scheduler",
        )
        assert isinstance(e, GXFScheduler)

    def test_clock(self, app):
        e = GreedyScheduler(app)
        with pytest.raises(RuntimeError) as err:
            # value will only be initialized by executor once app.run() is called
            e.clock
        assert "'clock' is not set" in str(err.value)

    def test_max_duration_ms(self, app):
        e = GreedyScheduler(app)

        # max_duration_ms is optional and will report -1 if not set
        assert e.max_duration_ms == -1

    def test_stop_on_deadlock(self, app):
        e = GreedyScheduler(app)
        with pytest.raises(RuntimeError):
            # value will only be initialized by executor once app.run() is called
            e.stop_on_deadlock

    def test_check_recession_period_ms(self, app):
        e = GreedyScheduler(app)
        with pytest.raises(RuntimeError):
            # value will only be initialized by executor once app.run() is called
            e.check_recession_period_ms

    def test_stop_on_deadlock_timeout(self, app):
        e = GreedyScheduler(app)
        with pytest.raises(RuntimeError):
            e.stop_on_deadlock_timeout


class TestMultiThreadScheduler:
    def test_default_init(self, app):
        e = MultiThreadScheduler(app)
        assert isinstance(e, GXFScheduler)
        assert isinstance(e, Scheduler)
        assert isinstance(e.spec, ComponentSpec)
        assert "<uninitialized>" in repr(e)
        assert repr(e).startswith("MultiThreadScheduler(self, clock=")

    @pytest.mark.parametrize("ClockClass", [ManualClock, RealtimeClock])
    def test_init_kwargs(self, app, ClockClass):
        e = MultiThreadScheduler(
            app,
            clock=ClockClass(app),
            worker_thread_number=4,
            stop_on_deadlock=True,
            check_recession_period_ms=2.0,
            max_duration_ms=10000,
            stop_on_deadlock_timeout=10,
            name="scheduler",
        )
        assert isinstance(e, GXFScheduler)

    def test_clock(self, app):
        e = MultiThreadScheduler(app)
        with pytest.raises(RuntimeError) as err:
            # value will only be initialized by executor once app.run() is called
            e.clock
        assert "'clock' is not set" in str(err.value)

    def test_worker_thread_number(self, app):
        e = MultiThreadScheduler(app)
        with pytest.raises(RuntimeError):
            # value will only be initialized by executor once app.run() is called
            e.worker_thread_number

    def test_max_duration_ms(self, app):
        e = MultiThreadScheduler(app)

        # max_duration_ms is optional and will report -1 if not set
        assert e.max_duration_ms == -1

    def test_stop_on_deadlock(self, app):
        e = MultiThreadScheduler(app)
        with pytest.raises(RuntimeError):
            # value will only be initialized by executor once app.run() is called
            e.stop_on_deadlock

    def test_check_recession_period_ms(self, app):
        e = MultiThreadScheduler(app)
        with pytest.raises(RuntimeError):
            # value will only be initialized by executor once app.run() is called
            e.check_recession_period_ms

    def test_stop_on_deadlock_timeout(self, app):
        e = MultiThreadScheduler(app)
        with pytest.raises(RuntimeError):
            # value will only be initialized by executor once app.run() is called
            e.stop_on_deadlock_timeout
