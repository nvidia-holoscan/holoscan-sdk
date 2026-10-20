#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exchange a generated multi-Tensor dataclass with the C++ relay graph."""

from __future__ import annotations

import argparse
from contextlib import suppress
from pathlib import Path

import external_frame
import numpy as np

import holoscan as hs

DATA_SEQUENCE = 1
ACK_SEQUENCE = 2
CLIENT_SOURCE = "standalone-client"
GRAPH_SOURCE = "graph-relay"
PIXELS = np.arange(1, 9, dtype=np.uint8).reshape(2, 4)
CALIBRATION = np.array([0.5, 1.0, -0.5], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    """Parse the graph-produced catalog and unique endpoint instance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--instance", required=True)
    return parser.parse_args()


def frame(sequence: int) -> external_frame.ExternalFrame:
    """Build one generated descriptor-schema value with both canonical fields."""
    return external_frame.ExternalFrame(
        sequence=sequence,
        source=CLIENT_SOURCE,
        pixels=PIXELS,
        calibration=CALIBRATION,
    )


def main() -> None:
    """Publish, validate the relayed frame, acknowledge, and close explicitly."""
    args = parse_args()
    catalog = hs.topics.EndpointCatalog.load(args.catalog)
    import_endpoint = f"descriptor-tensor-import-{args.instance}"
    export_endpoint = f"descriptor-tensor-export-{args.instance}"

    with hs.channel.Runtime() as runtime:
        publisher = runtime.publisher(catalog, import_endpoint)
        subscriber = runtime.subscriber(catalog, export_endpoint)
        publisher.wait_for_subscribers(10.0)
        subscriber.refresh_connections()

        if publisher.emit(frame(DATA_SEQUENCE), timeout=10.0) != 1:
            raise RuntimeError("the graph did not accept exactly one input frame")
        sample = subscriber.receive(10.0)
        value = sample.data
        if not isinstance(value, external_frame.ExternalFrame):
            raise TypeError(f"unexpected generated value: {type(value)!r}")
        if value.sequence != DATA_SEQUENCE or value.source != GRAPH_SOURCE:
            raise RuntimeError(f"unexpected relayed metadata: {value!r}")
        if set(value.tensors) != {"pixels", "calibration"}:
            raise RuntimeError(f"unexpected Tensor field inventory: {value.tensors!r}")
        if not value.pixels.is_read_only or not value.calibration.is_read_only:
            raise RuntimeError("received Tensor fields must be read-only")
        np.testing.assert_array_equal(np.asarray(value.pixels), PIXELS)
        np.testing.assert_array_equal(np.asarray(value.calibration), CALIBRATION)

        if publisher.emit(frame(ACK_SEQUENCE), timeout=10.0) != 1:
            raise RuntimeError("the graph did not accept exactly one acknowledgment")
        # The graph stops after the acknowledgment, so end-of-stream may find its subscriber
        # already gone; NotReady and Closed are benign here, as in the C++ client.
        with suppress(hs.errors.NotReadyError, hs.errors.ClosedError):
            publisher.close()
        subscriber.close()

    print("Python descriptor Tensor external-topic client complete")


if __name__ == "__main__":
    main()
