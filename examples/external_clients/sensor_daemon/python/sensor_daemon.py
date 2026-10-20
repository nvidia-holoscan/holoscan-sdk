#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Publish captured uint32 samples through a standalone Python Channel client."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import holoscan as hs


def positive_integer(value: str) -> int:
    """Parse one positive command-line integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def uint32(value: str) -> int:
    """Parse one uint32 command-line integer."""
    parsed = int(value)
    if parsed < 0 or parsed > 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("value must fit uint32")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse the C++ sensor daemon's catalog-first command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("endpoint")
    parser.add_argument("count", nargs="?", type=positive_integer, default=64)
    parser.add_argument("period_ms", nargs="?", type=positive_integer, default=5)
    parser.add_argument("first_value", nargs="?", type=uint32, default=1)
    args = parser.parse_args()
    if args.first_value + args.count - 1 > 0xFFFFFFFF:
        parser.error("first-value plus count exceeds the uint32 sample range")
    return args


def main() -> None:
    """Attach, publish a finite captured stream, and send ordered EOS."""
    args = parse_args()
    catalog = hs.topics.EndpointCatalog.load(args.catalog)
    recipient_deliveries = 0
    with hs.channel.Runtime() as runtime:
        publisher = runtime.publisher(catalog, args.endpoint)
        publisher.wait_for_subscribers(5.0)
        for index in range(args.count):
            capture_time = publisher.make_capture_time(time.time_ns())
            recipient_deliveries += publisher.emit(
                args.first_value + index,
                capture_time=capture_time,
                timeout=5.0,
            )
            time.sleep(args.period_ms / 1000.0)
        publisher.close()

    print(
        f"published={args.count} recipient_deliveries={recipient_deliveries} "
        f"eos=ordered catalog_digest={catalog.digest}"
    )


if __name__ == "__main__":
    main()
