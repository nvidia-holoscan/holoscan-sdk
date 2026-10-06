# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# fmt: off

"""Clock authoring support types."""

from holoscan.time import TimePoint as _time_TimePoint



class ClockType:
    """A registered SDK Clock type token."""

    @property
    def name(self) -> str: ...

    def __repr__(self) -> str: ...

class ClockKey:
    """A named cold-authoring Clock declaration key."""

    @property
    def name(self) -> str: ...

    @property
    def is_default(self) -> bool: ...

    def __repr__(self) -> str: ...

class ClockCapabilities:
    @property
    def can_read_now(self) -> bool: ...

    @property
    def can_wait_until(self) -> bool: ...

    @property
    def steady(self) -> bool: ...

    @property
    def has_outgoing_conversions(self) -> bool: ...

    @property
    def has_incoming_conversions(self) -> bool: ...

class ClockRef:
    @property
    def id(self) -> int: ...

    @property
    def capabilities(self) -> ClockCapabilities: ...

    def now(self) -> _time_TimePoint: ...

    def at_ns(self, timestamp_ns: int) -> _time_TimePoint: ...

class ClockRateRatio:
    def __init__(self, numerator: int = 1, denominator: int = 1) -> None: ...

    @property
    def numerator(self) -> int: ...

    @numerator.setter
    def numerator(self, arg: int, /) -> None: ...

    @property
    def denominator(self) -> int: ...

    @denominator.setter
    def denominator(self, arg: int, /) -> None: ...

class AffineClockConversion:
    def __init__(self, scale: ClockRateRatio = ..., offset_ns: int = 0) -> None: ...

    @property
    def scale(self) -> ClockRateRatio: ...

    @scale.setter
    def scale(self, arg: ClockRateRatio, /) -> None: ...

    @property
    def offset_ns(self) -> int: ...

    @offset_ns.setter
    def offset_ns(self, arg: int, /) -> None: ...

__all__: list[str] = ['ClockType', 'ClockKey', 'ClockCapabilities', 'ClockRef', 'ClockRateRatio', 'AffineClockConversion']
