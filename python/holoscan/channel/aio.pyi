# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native ``asyncio`` adapters for standalone Channel endpoints."""

from datetime import timedelta as _timedelta
from typing import Any

from holoscan import Sample as _Sample
from holoscan import SampleFlags as _SampleFlags
from holoscan.channel import Publisher as _Publisher
from holoscan.channel import Subscriber as _Subscriber
from holoscan.time import Duration as _Duration
from holoscan.time import TimePoint as _TimePoint

type _Timeout = float | _timedelta | _Duration

class Subscriber:
    def __init__(self, subscriber: _Subscriber) -> None: ...
    async def receive(self, timeout: _Timeout | None = None) -> _Sample[Any]: ...
    def detach(self) -> None: ...

class Publisher:
    def __init__(self, publisher: _Publisher) -> None: ...
    async def emit(
        self,
        value: object,
        *,
        capture_time: _TimePoint | None = None,
        frame_id: int | None = None,
        flags: _SampleFlags | None = None,
        timeout: _Timeout | None = None,
    ) -> int:
        """Emit a value and return the number of subscribers that accepted it.

        A timeout retries only pre-submit readiness or backpressure outcomes against one deadline.
        """

__all__: list[str] = ["Publisher", "Subscriber"]
