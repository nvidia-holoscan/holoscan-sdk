// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <holoscan/channel/runtime.hpp>
#include "byte_channel_payload_schema_traits.hpp"

// REQ: SCHEMA-PY-007
int main(int argc, char** argv) {
  if (argc != 2) {
    return 1;
  }
  const std::string name{argv[1]};
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    std::cerr << "runtime creation failed: " << holoscan::runtime_error_message(runtime.error())
              << '\n';
    return 2;
  }
  using Payload = holoscan::test::ByteChannelPayloadT;
  holoscan::ChannelOptions options{};
  options.enable_event_wake = true;
  auto publisher = runtime->create_publisher<Payload>(name + "-native", options);
  auto subscriber = runtime->create_subscriber<Payload>(name + "-python", options);
  if (!publisher || !subscriber) {
    std::cerr << "endpoint creation failed: "
              << holoscan::runtime_error_message(!publisher ? publisher.error()
                                                            : subscriber.error())
              << '\n';
    return 3;
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{10};
  while (!publisher->has_subscribers() || !subscriber->has_publishers()) {
    if (std::chrono::steady_clock::now() >= deadline || !publisher->update_connections() ||
        !subscriber->refresh_connections()) {
      std::cerr << "endpoint rendezvous failed\n";
      return 4;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  Payload value;
  for (unsigned index = 0; index < 256U; ++index) {
    value.blob.push_back(static_cast<std::uint8_t>(index));
  }
  if (!publisher->emit(value)) {
    return 5;
  }
  auto received = subscriber->wait_and_receive_sample(deadline);
  if (!received || received->data.blob != value.blob) {
    return 6;
  }
  return publisher->close() ? 0 : 7;
}
