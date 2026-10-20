#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Record scalar results through a standalone Python Channel client."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import threading
from pathlib import Path

import holoscan as hs


def positive_integer(value: str) -> int:
    """Parse one positive command-line integer."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    """Parse the C++ recorder's catalog-first command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("endpoint")
    parser.add_argument("max_samples", nargs="?", type=positive_integer)
    parser.add_argument(
        "--async",
        dest="use_asyncio",
        action="store_true",
        help="receive through the native-readiness asyncio adapter",
    )
    return parser.parse_args()


def print_sample(sample: hs.channel.Sample[object], catalog: hs.topics.EndpointCatalog) -> None:
    """Print one received value and its transport metadata."""
    metadata = sample.metadata
    print(
        f"value={sample.data} sequence={metadata.sequence} "
        f"capture_time={metadata.capture_time} publish_time={metadata.publish_time} "
        f"frame_id={metadata.frame_id} flags={metadata.flags} "
        f"catalog_digest={catalog.digest}"
    )


def record(
    subscriber: hs.channel.Subscriber,
    catalog: hs.topics.EndpointCatalog,
    max_samples: int | None,
) -> tuple[int, bool]:
    """Synchronously receive until producer EOS or the optional sample limit."""
    received = 0
    remote_end_of_stream = False
    print("subscriber ready", flush=True)
    while max_samples is None or received < max_samples:
        try:
            sample = subscriber.receive(10.0)
        except hs.errors.ClosedError:
            remote_end_of_stream = True
            break
        received += 1
        print_sample(sample, catalog)
    return received, remote_end_of_stream


async def record_async(
    subscriber: hs.channel.Subscriber,
    catalog: hs.topics.EndpointCatalog,
    max_samples: int | None,
) -> tuple[int, bool]:
    """Receive through native readiness without creating a Python helper thread."""
    async_subscriber = hs.channel.aio.Subscriber(subscriber)
    received = 0
    remote_end_of_stream = False
    python_threads = threading.active_count()
    pending = asyncio.create_task(async_subscriber.receive(timeout=10.0))
    await asyncio.sleep(0)
    if threading.active_count() != python_threads:
        raise RuntimeError("asyncio recorder created a helper thread while attaching")
    print("subscriber ready", flush=True)
    try:
        while max_samples is None or received < max_samples:
            try:
                sample = await pending
            except hs.errors.ClosedError:
                remote_end_of_stream = True
                break
            received += 1
            print_sample(sample, catalog)
            if threading.active_count() != python_threads:
                raise RuntimeError("asyncio recorder created a helper thread while receiving")
            if max_samples is None or received < max_samples:
                pending = asyncio.create_task(async_subscriber.receive(timeout=10.0))
    finally:
        if not pending.done():
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pending
        async_subscriber.detach()
    if threading.active_count() != python_threads:
        raise RuntimeError("asyncio recorder retained a helper thread after detach")
    print("asyncio helper threads=0")
    return received, remote_end_of_stream


def main() -> None:
    """Receive until producer EOS or the optional local sample limit."""
    args = parse_args()
    catalog = hs.topics.EndpointCatalog.load(args.catalog)
    with hs.channel.Runtime() as runtime:
        subscriber = runtime.subscriber(catalog, args.endpoint)
        try:
            if args.use_asyncio:
                received, remote_end_of_stream = asyncio.run(
                    record_async(subscriber, catalog, args.max_samples)
                )
            else:
                received, remote_end_of_stream = record(subscriber, catalog, args.max_samples)
        finally:
            subscriber.close()

    eos = "remote" if remote_end_of_stream else "not-observed"
    stop = " stop=sample-limit" if not remote_end_of_stream else ""
    print(f"received={received} eos={eos}{stop}")


if __name__ == "__main__":
    main()
