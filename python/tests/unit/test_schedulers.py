"""
SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from holoscan.core import Scheduler
from holoscan.core._core import ComponentSpec as ComponentSpecBase
from holoscan.gxf import GXFScheduler
from holoscan.resources import ManualClock, RealtimeClock, SyntheticClock
from holoscan.schedulers import EventBasedScheduler, GreedyScheduler, MultiThreadScheduler


class TestGreedyScheduler:
    def test_default_init(self, app):
        scheduler = GreedyScheduler(app)
        assert isinstance(scheduler, GXFScheduler)
        assert isinstance(scheduler, Scheduler)
        assert issubclass(ComponentSpecBase, type(scheduler.spec))

    @pytest.mark.parametrize("ClockClass", [ManualClock, RealtimeClock, SyntheticClock])
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

    def test_clock_default(self, app):
        scheduler = GreedyScheduler(app)
        # the default clock is not initialized by the executor until app.run() is called
        assert scheduler.clock is None

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
        assert issubclass(ComponentSpecBase, type(scheduler.spec))

    @pytest.mark.parametrize("ClockClass", [ManualClock, RealtimeClock, SyntheticClock])
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
            strict_job_thread_pinning=True,
            name=name,
        )
        assert isinstance(scheduler, GXFScheduler)
        assert f"name: {name}" in repr(scheduler)

    def test_clock(self, app):
        scheduler = MultiThreadScheduler(app)
        # the default clock is not initialized by the executor until app.run() is called
        assert scheduler.clock is None

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
        assert issubclass(ComponentSpecBase, type(scheduler.spec))

    @pytest.mark.parametrize("ClockClass", [ManualClock, RealtimeClock, SyntheticClock])
    def test_init_kwargs(self, app, ClockClass):  # noqa: N803
        name = "event-based-scheduler"
        scheduler = EventBasedScheduler(
            app,
            clock=ClockClass(app),
            worker_thread_number=4,
            stop_on_deadlock=True,
            max_duration_ms=10000,
            stop_on_deadlock_timeout=10,
            pin_cores=[0, 1],
            enable_queue_stealing=False,
            steal_scan_limit=4,
            enable_worker_postcheck_fastpath=False,
            postcheck_fallback_notify_interval=128,
            postcheck_fallback_notify_min_workers=4,
            postcheck_fallback_notify_min_period_ns=50000,
            internal_event_shard_count=2,
            dispatcher_internal_pop_batch_size=16,
            wait_state_shard_count=2,
            log_perf_stats=True,
            name=name,
        )
        assert isinstance(scheduler, GXFScheduler)
        assert f"name: {name}" in repr(scheduler)

    def test_clock(self, app):
        scheduler = EventBasedScheduler(app)
        # the default clock is not initialized by the executor until app.run() is called
        assert scheduler.clock is None

    def test_worker_thread_number(self, app):
        scheduler = EventBasedScheduler(app)
        with pytest.raises(RuntimeError):
            # value will only be initialized by executor once app.run() is called
            scheduler.worker_thread_number  # noqa: B018

    def test_max_duration_ms(self, app):
        scheduler = EventBasedScheduler(app)

        # max_duration_ms is optional and will report -1 if not set
        assert scheduler.max_duration_ms == -1

    def test_pin_cores(self, app):
        scheduler = EventBasedScheduler(app)

        # pin_cores is optional and will report an empty list if not set
        assert scheduler.pin_cores == []

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

    def test_enable_queue_stealing(self, app):
        scheduler = EventBasedScheduler(app)
        with pytest.raises(RuntimeError):
            scheduler.enable_queue_stealing  # noqa: B018

    def test_steal_scan_limit(self, app):
        scheduler = EventBasedScheduler(app)
        with pytest.raises(RuntimeError):
            scheduler.steal_scan_limit  # noqa: B018

    def test_enable_worker_postcheck_fastpath(self, app):
        scheduler = EventBasedScheduler(app)
        with pytest.raises(RuntimeError):
            scheduler.enable_worker_postcheck_fastpath  # noqa: B018

    def test_postcheck_fallback_notify_interval(self, app):
        scheduler = EventBasedScheduler(app)
        with pytest.raises(RuntimeError):
            scheduler.postcheck_fallback_notify_interval  # noqa: B018

    def test_postcheck_fallback_notify_min_workers(self, app):
        scheduler = EventBasedScheduler(app)
        with pytest.raises(RuntimeError):
            scheduler.postcheck_fallback_notify_min_workers  # noqa: B018

    def test_postcheck_fallback_notify_min_period_ns(self, app):
        scheduler = EventBasedScheduler(app)
        with pytest.raises(RuntimeError):
            scheduler.postcheck_fallback_notify_min_period_ns  # noqa: B018

    def test_internal_event_shard_count(self, app):
        scheduler = EventBasedScheduler(app)
        with pytest.raises(RuntimeError):
            scheduler.internal_event_shard_count  # noqa: B018

    def test_dispatcher_internal_pop_batch_size(self, app):
        scheduler = EventBasedScheduler(app)
        with pytest.raises(RuntimeError):
            scheduler.dispatcher_internal_pop_batch_size  # noqa: B018

    def test_wait_state_shard_count(self, app):
        scheduler = EventBasedScheduler(app)
        with pytest.raises(RuntimeError):
            scheduler.wait_state_shard_count  # noqa: B018

    def test_log_perf_stats(self, app):
        scheduler = EventBasedScheduler(app)
        with pytest.raises(RuntimeError):
            scheduler.log_perf_stats  # noqa: B018
