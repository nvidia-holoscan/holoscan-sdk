"""
 SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest

from holoscan.core import ComponentSpec, Scheduler
from holoscan.gxf import GXFScheduler
from holoscan.resources import ManualClock, RealtimeClock
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler, MultiThreadScheduler


class TestGreedyScheduler:
    def test_default_init(self, app):
        scheduler = GreedyScheduler(app)
        assert isinstance(scheduler, GXFScheduler)
        assert isinstance(scheduler, Scheduler)
        assert isinstance(scheduler.spec, ComponentSpec)

    @pytest.mark.parametrize("ClockClass", [ManualClock, RealtimeClock])
    def test_init_kwargs(self, app, ClockClass):  # noqa: N803
        name = "greedy-scheduler"
        scheduler = GreedyScheduler(
            app,
            clock=ClockClass(app),
            stop_on_deadlock=True,
            max_duration_ms=10000,
            check_recession_period_ms=1.0,
            stop_on_deadlock_timeout=-1,
            name=name,
        )
        assert isinstance(scheduler, GXFScheduler)
        assert f"name: {name}" in repr(scheduler)

    def test_clock(self, app):
        scheduler = GreedyScheduler(app)
        with pytest.raises(RuntimeError) as err:
            # value will only be initialized by executor once app.run() is called
            scheduler.clock  # noqa: B018
        assert "'clock' is not set" in str(err.value)

    def test_max_duration_ms(self, app):
        scheduler = GreedyScheduler(app)

        # max_duration_ms is optional and will report -1 if not set
        assert scheduler.max_duration_ms == -1

    def test_stop_on_deadlock(self, app):
        scheduler = GreedyScheduler(app)
        with pytest.raises(RuntimeError):
            # value will only be initialized by executor once app.run() is called
            scheduler.stop_on_deadlock  # noqa: B018

    def test_check_recession_period_ms(self, app):
        scheduler = GreedyScheduler(app)
        with pytest.raises(RuntimeError):
            # value will only be initialized by executor once app.run() is called
            scheduler.check_recession_period_ms  # noqa: B018

    def test_stop_on_deadlock_timeout(self, app):
        scheduler = GreedyScheduler(app)
        with pytest.raises(RuntimeError):
            scheduler.stop_on_deadlock_timeout  # noqa: B018


class TestMultiThreadScheduler:
    def test_default_init(self, app):
        scheduler = MultiThreadScheduler(app)
        assert isinstance(scheduler, GXFScheduler)
        assert isinstance(scheduler, Scheduler)
        assert isinstance(scheduler.spec, ComponentSpec)

    @pytest.mark.parametrize("ClockClass", [ManualClock, RealtimeClock])
    def test_init_kwargs(self, app, ClockClass):  # noqa: N803
        name = "multithread-scheduler"
        scheduler = MultiThreadScheduler(
            app,
            clock=ClockClass(app),
            worker_thread_number=4,
            stop_on_deadlock=True,
            check_recession_period_ms=2.0,
            max_duration_ms=10000,
            stop_on_deadlock_timeout=10,
            name=name,
        )
        assert isinstance(scheduler, GXFScheduler)
        assert f"name: {name}" in repr(scheduler)

    def test_clock(self, app):
        scheduler = MultiThreadScheduler(app)
        with pytest.raises(RuntimeError) as err:
            # value will only be initialized by executor once app.run() is called
            scheduler.clock  # noqa: B018
        assert "'clock' is not set" in str(err.value)

    def test_worker_thread_number(self, app):
        scheduler = MultiThreadScheduler(app)
        with pytest.raises(RuntimeError):
            # value will only be initialized by executor once app.run() is called
            scheduler.worker_thread_number  # noqa: B018

    def test_max_duration_ms(self, app):
        scheduler = MultiThreadScheduler(app)

        # max_duration_ms is optional and will report -1 if not set
        assert scheduler.max_duration_ms == -1

    def test_stop_on_deadlock(self, app):
        scheduler = MultiThreadScheduler(app)
        with pytest.raises(RuntimeError):
            # value will only be initialized by executor once app.run() is called
            scheduler.stop_on_deadlock  # noqa: B018

    def test_check_recession_period_ms(self, app):
        scheduler = MultiThreadScheduler(app)
        with pytest.raises(RuntimeError):
            # value will only be initialized by executor once app.run() is called
            scheduler.check_recession_period_ms  # noqa: B018

    def test_stop_on_deadlock_timeout(self, app):
        scheduler = MultiThreadScheduler(app)
        with pytest.raises(RuntimeError):
            # value will only be initialized by executor once app.run() is called
            scheduler.stop_on_deadlock_timeout  # noqa: B018


class TestEventBasedScheduler:
    def test_default_init(self, app):
        scheduler = EventBasedScheduler(app)
        assert isinstance(scheduler, GXFScheduler)
        assert isinstance(scheduler, Scheduler)
        assert isinstance(scheduler.spec, ComponentSpec)

    @pytest.mark.parametrize("ClockClass", [ManualClock, RealtimeClock])
    def test_init_kwargs(self, app, ClockClass):  # noqa: N803
        name = "event-based-scheduler"
        scheduler = EventBasedScheduler(
            app,
            clock=ClockClass(app),
            worker_thread_number=4,
            stop_on_deadlock=True,
            max_duration_ms=10000,
            stop_on_deadlock_timeout=10,
            name=name,
        )
        assert isinstance(scheduler, GXFScheduler)
        assert f"name: {name}" in repr(scheduler)

    def test_clock(self, app):
        scheduler = EventBasedScheduler(app)
        with pytest.raises(RuntimeError) as err:
            # value will only be initialized by executor once app.run() is called
            scheduler.clock  # noqa: B018
        assert "'clock' is not set" in str(err.value)

    def test_worker_thread_number(self, app):
        scheduler = EventBasedScheduler(app)
        with pytest.raises(RuntimeError):
            # value will only be initialized by executor once app.run() is called
            scheduler.worker_thread_number  # noqa: B018

    def test_max_duration_ms(self, app):
        scheduler = EventBasedScheduler(app)

        # max_duration_ms is optional and will report -1 if not set
        assert scheduler.max_duration_ms == -1

    def test_stop_on_deadlock(self, app):
        scheduler = EventBasedScheduler(app)
        with pytest.raises(RuntimeError):
            # value will only be initialized by executor once app.run() is called
            scheduler.stop_on_deadlock  # noqa: B018

    def test_stop_on_deadlock_timeout(self, app):
        scheduler = EventBasedScheduler(app)
        with pytest.raises(RuntimeError):
            # value will only be initialized by executor once app.run() is called
            scheduler.stop_on_deadlock_timeout  # noqa: B018
