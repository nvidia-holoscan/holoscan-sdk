# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pass an installed-consumer constructor handle through the installed Holoscan package."""

import gc

import installed_handle_provider as provider

import holoscan as hs


class Channel:
    """Expose one provider-table channel through the constructor-handle protocol."""

    def __init__(self, channel_id: int) -> None:
        self.key = provider.new_channel(channel_id)

    def __holoscan_constructor_handle__(self, type_name: str) -> object:
        return provider.export_channel(self.key, type_name)


provider.register_rows()
source_type = hs.registry.resolve("holoscan.installed.ChannelSource", kind="operator")
assert list(source_type.constructor_handles) == ["channel"], source_type.constructor_handles

graph = hs.Graph("installed-handle-provider")
channel = Channel(1)
source = source_type(graph, "source", channel=channel)
assert "channel=<holoscan.installed.Channel: process-local>" in repr(source), repr(source)

# The Graph's factory now owns the C++ object; drop the provider's own reference.
provider.forget_channel(channel.key)
del channel, source
gc.collect()
assert provider.live_channels() == 1, provider.live_channels()
del graph
gc.collect()
assert provider.live_channels() == 0, provider.live_channels()
print("installed constructor-handle provider: ok")
