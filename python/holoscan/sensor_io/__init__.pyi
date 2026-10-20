# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# fmt: off

"""Sensor-operator sequence and provenance utilities."""

import enum

import holoscan


class CaptureTimePolicy(enum.Enum):
    """Choose which contributor capture time a combined result inherits."""

    EARLIEST = 0
    LATEST = 1


class ProvenanceAccumulator:
    """
    Combine metadata from one output group's input samples into options for a fresh
    ``Output.emit()``.

    Retained instances keep every observation until ``reset()`` is called.
    """

    def __init__(self) -> None: ...

    def observe(self, metadata: holoscan.SampleMetadata) -> None: ...

    def emit_options(self, policy: CaptureTimePolicy) -> holoscan.EmitOptions:
        """
        Return authoritative provenance for a fused result published with ``Output.emit()``.

        This method does not reset the accumulator. Call ``reset()`` after publishing one output
        group and before observing contributors for the next group. Passing the returned options
        to ``Output.forward()`` raises ``InvalidArgumentError`` because forwarding is a one-to-one
        operation whose ordinary options inherit absent fields from its source.
        """

    @property
    def observed(self) -> int: ...

    @property
    def mixed_domains(self) -> bool: ...

    @property
    def has_unstamped_contributor(self) -> bool: ...

    def reset(self) -> None:
        """Discard every contributor observed for the current output group."""


class SequenceObservation:
    """What one observed device sequence number implies about missing samples."""

    @property
    def sequence(self) -> int: ...

    @property
    def gap(self) -> int: ...

    @property
    def first(self) -> bool: ...

    @property
    def regressed(self) -> bool: ...


class SequenceTracker:
    """Track a device sequence counter and report gaps and regressions."""

    def __init__(self) -> None: ...

    def observe(self, sequence: int) -> SequenceObservation: ...

    @property
    def total_gap(self) -> int: ...

    @property
    def previous(self) -> int: ...

    @property
    def seen(self) -> bool: ...

    def reset(self) -> None: ...


__all__: list[str] = ['CaptureTimePolicy', 'ProvenanceAccumulator', 'SequenceObservation', 'SequenceTracker']
