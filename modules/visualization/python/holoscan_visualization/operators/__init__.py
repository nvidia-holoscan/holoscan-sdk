# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Holoscan Visualization operator bindings."""

from datetime import timedelta as _timedelta
from typing import Any as _Any

from holoscan.registry import resolve as _resolve
from holoscan.time import Duration as _Duration


def _duration_nanoseconds(value: _Duration | _timedelta) -> int:
    if isinstance(value, _Duration):
        return value.nanoseconds
    if not isinstance(value, _timedelta):
        raise TypeError("duration must be datetime.timedelta or holoscan.time.Duration")
    return (
        value.days * 86_400_000_000_000 + value.seconds * 1_000_000_000 + value.microseconds * 1_000
    )


def _normalize_trigger(trigger: object) -> object:
    """Copy and lower the public clock duration without mutating caller-owned input."""

    if isinstance(trigger, dict) and "period_ns" in trigger:
        raise TypeError("use period with datetime.timedelta or holoscan.time.Duration")
    if not isinstance(trigger, dict) or trigger.get("kind") != "clock" or "period" not in trigger:
        return trigger
    normalized = dict(trigger)
    normalized["period_ns"] = _duration_nanoseconds(normalized.pop("period"))
    return normalized


class _HolovizOpType:
    """Callable module facade for the registered C++ HolovizOp."""

    def __init__(self) -> None:
        self._registered = _resolve("holoscan.visualization.HolovizOp", kind="operator")

    @property
    def Handle(self) -> object:  # noqa: N802
        return self._registered.Handle

    @property
    def name(self) -> str:
        return self._registered.name

    @property
    def description(self) -> str:
        return self._registered.description

    @property
    def parameter_introspection(self) -> str:
        return self._registered.parameter_introspection

    @property
    def parameter_summary(self) -> str:
        return self._registered.parameter_summary

    @property
    def ports(self) -> object:
        return self._registered.ports

    @property
    def parameters(self) -> object:
        return self._registered.parameters

    @property
    def required_parameters(self) -> frozenset[str]:
        return self._registered.required_parameters

    def __call__(
        self,
        graph: _Any,
        instance_name: str,
        *,
        layers: object,
        sink: object,
        trigger: object | None = None,
        max_input_age: _Duration | _timedelta | None = None,
    ) -> object:
        params = {"layers": layers, "sink": sink}
        if trigger is not None:
            params["trigger"] = _normalize_trigger(trigger)
        if max_input_age is not None:
            params["max_input_age_ns"] = _duration_nanoseconds(max_input_age)
        return self._registered(graph, instance_name, **params)

    def __repr__(self) -> str:
        return repr(self._registered)


HolovizOp = _HolovizOpType()

__all__ = ["HolovizOp"]
