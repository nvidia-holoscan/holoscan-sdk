# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify rebuilt external schema registration against only the installed Core package."""

import subprocess
import sys
import uuid
from pathlib import Path

import byte_channel_payload

import holoscan as hs

# REQ: SCHEMA-PY-007
assert Path(hs.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())
name = f"installed-bytes-{uuid.uuid4().hex}"
options = hs.channel.ChannelOptions(enable_event_wake=True)
with hs.channel.Runtime() as runtime:
    subscriber = runtime.subscriber(name + "-native", byte_channel_payload.ByteChannelPayload, options=options)
    publisher = runtime.publisher(name + "-python", byte_channel_payload.ByteChannelPayload, options=options)
    peer = subprocess.Popen([sys.argv[1], name])
    try:
        publisher.wait_for_subscribers(5.0)
        subscriber.refresh_connections()
        value = subscriber.receive(5.0).data
        assert type(value.blob) is bytes
        assert value.blob == bytes(range(256))
        publisher.emit(value, timeout=5.0)
        status = peer.wait(timeout=10)
        assert status == 0, f"native peer exit status: {status}"
    except Exception as error:
        raise AssertionError(f"installed provider round trip failed; native peer status: {peer.poll()}") from error
    finally:
        if peer.poll() is None:
            peer.terminate()
            peer.wait(timeout=5)
assert value.blob == bytes(range(256))
