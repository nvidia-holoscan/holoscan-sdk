/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>

#include "control_payload_generated.h"
#include "control_payload_schema_traits.hpp"
#include "stereo_payload_schema_traits.hpp"

namespace {

using namespace std::chrono_literals;
using ControlPayload = python::fixture::ControlPayloadT;
using StereoPayload =
    holoscan::schema::DescriptorFlatBufferPayloadFor<python::fixture::StereoPayload>;

template <typename Publisher, typename Value>
holoscan::expected<std::uint32_t, holoscan::Error> emit_when_attached(Publisher& publisher,
                                                                      const Value& value) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = publisher.update_connections();
    if (!refreshed) {
      return holoscan::make_unexpected(std::move(refreshed).error());
    }
    if (publisher.has_subscribers()) {
      auto emitted = publisher.emit(value);
      if (emitted || (emitted.error().code != holoscan::ErrorCode::kNotReady &&
                      emitted.error().code != holoscan::ErrorCode::kBackpressured)) {
        return emitted;
      }
    }
    std::this_thread::sleep_for(2ms);
  }
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout});
}

template <typename Subscriber>
auto receive_when_ready(Subscriber& subscriber) {
  using Result = decltype(subscriber.receive_sample());
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    Result sample = subscriber.receive_sample();
    if (sample || sample.error().code != holoscan::ErrorCode::kNotReady) {
      return sample;
    }
    std::this_thread::sleep_for(2ms);
  }
  return Result{holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout})};
}

holoscan::expected<holoscan::ChannelResolvedTensorBundle<StereoPayload>, holoscan::Error>
receive_stereo_when_ready(holoscan::ChannelTensorSchemaSubscriber<StereoPayload>& subscriber) {
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto ready = subscriber.has_samples();
    if (!ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
    if (*ready) {
      auto sample = subscriber.receive_tensors();
      if (sample || sample.error().code != holoscan::ErrorCode::kNotReady) {
        return sample;
      }
    }
    std::this_thread::sleep_for(2ms);
  }
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout});
}

template <typename Subscriber>
bool wait_for_publisher(Subscriber& subscriber) {
  const auto deadline = std::chrono::steady_clock::now() + 5s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = subscriber.refresh_connections();
    if (!refreshed) {
      return false;
    }
    if (subscriber.has_publishers()) {
      return true;
    }
    std::this_thread::sleep_for(2ms);
  }
  return false;
}

int run_channel_client_peer(std::string_view prefix) {
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    std::cerr << "runtime creation failed\n";
    return 20;
  }
  holoscan::ChannelOptions options{};
  options.enable_event_wake = true;
  std::string scalar_output{prefix};
  scalar_output.append("-scalar-from-cpp");
  std::string scalar_input{prefix};
  scalar_input.append("-scalar-to-cpp");
  std::string schema_output{prefix};
  schema_output.append("-schema-from-cpp");
  std::string schema_input{prefix};
  schema_input.append("-schema-to-cpp");
  auto scalar_publisher = runtime->create_publisher<std::uint32_t>(scalar_output, options);
  auto scalar_subscriber = runtime->create_subscriber<std::uint32_t>(scalar_input, options);
  auto schema_publisher = runtime->create_publisher<ControlPayload>(schema_output, options);
  auto schema_subscriber = runtime->create_subscriber<ControlPayload>(schema_input, options);
  if (!scalar_publisher || !scalar_subscriber || !schema_publisher || !schema_subscriber) {
    std::cerr << "name-based endpoint creation failed\n";
    return 21;
  }

  ControlPayload outbound;
  outbound.sequence = 43U;
  outbound.gain = 2.5F;
  outbound.labels = {"cpp", "channel"};
  const auto scalar_emitted = emit_when_attached(*scalar_publisher, std::uint32_t{41U});
  const auto schema_emitted = emit_when_attached(*schema_publisher, outbound);
  if (!scalar_emitted || !schema_emitted || !wait_for_publisher(*scalar_subscriber) ||
      !wait_for_publisher(*schema_subscriber)) {
    std::cerr << "peer attachment or emit failed\n";
    return 22;
  }

  auto scalar_received =
      scalar_subscriber->wait_and_receive_sample(std::chrono::steady_clock::now() + 5s);
  auto schema_received =
      schema_subscriber->wait_and_receive_sample(std::chrono::steady_clock::now() + 5s);
  if (!scalar_received || !schema_received) {
    std::cerr << "peer receive failed\n";
    return 23;
  }
  const bool valid = scalar_received->data == 42U && schema_received->data.sequence == 44U &&
                     schema_received->data.gain == 3.5F &&
                     schema_received->data.labels == std::vector<std::string>{"python", "channel"};
  auto scalar_closed = scalar_publisher->close();
  auto schema_closed = schema_publisher->close();
  if (!scalar_closed || !schema_closed) {
    std::cerr << "peer EOS failed\n";
    return 24;
  }
  std::cout << "standalone channel C++ peer complete\n";
  return valid ? 0 : 25;
}

int run_stereo_subscriber(const char* catalog_path, const char* endpoint_name) {
  auto catalog = holoscan::ChannelEndpointCatalog::load_json(catalog_path);
  if (!catalog) {
    std::cerr << "catalog load failed: " << holoscan::error_code_name(catalog.error().code) << '\n';
    return 1;
  }
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    std::cerr << "runtime creation failed: " << holoscan::error_code_name(runtime.error().code)
              << '\n';
    return 2;
  }
  auto subscriber =
      runtime->create_tensor_schema_subscriber<StereoPayload>(*catalog, endpoint_name);
  if (!subscriber) {
    std::cerr << "endpoint attachment failed: "
              << holoscan::error_code_name(subscriber.error().code) << '\n';
    return 3;
  }
  auto sample = receive_stereo_when_ready(*subscriber);
  if (!sample) {
    std::cerr << "receive failed: " << holoscan::error_code_name(sample.error().code) << '\n';
    return 4;
  }

  const auto* root = sample->root();
  const auto* header = root == nullptr ? nullptr : root->header();
  const auto* source = header == nullptr ? nullptr : header->source();
  const auto left_ready = sample->tensors[0].synchronize_for_host();
  const auto right_ready = sample->tensors[1].synchronize_for_host();
  const auto* left = sample->tensors[0].data_as<std::uint8_t>();
  const auto* right = sample->tensors[1].data_as<std::uint8_t>();
  constexpr std::array<std::uint8_t, 4> kExpectedLeft{1U, 2U, 3U, 4U};
  constexpr std::array<std::uint8_t, 4> kExpectedRight{5U, 6U, 7U, 8U};
  const bool valid = root != nullptr && header != nullptr && source != nullptr &&
                     header->sequence() == 9U && source->string_view() == "python" &&
                     root->exposure_us() == 2500U && left_ready && right_ready &&
                     sample->tensors[0].nbytes() == 4 && sample->tensors[1].nbytes() == 4 &&
                     left != nullptr && right != nullptr &&
                     std::equal(kExpectedLeft.begin(), kExpectedLeft.end(), left) &&
                     std::equal(kExpectedRight.begin(), kExpectedRight.end(), right);
  auto closed = subscriber->close();
  if (!closed && closed.error().code != holoscan::ErrorCode::kClosed) {
    std::cerr << "close failed: " << holoscan::error_code_name(closed.error().code) << '\n';
    return 5;
  }
  if (!valid) {
    std::cerr << "generated stereo frame validation failed\n";
    return 6;
  }
  std::cout << "generated stereo C++ peer complete\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::string_view(argv[1]) == "--channel-client") {
    return run_channel_client_peer(argv[2]);
  }
  if (argc == 4 && std::string_view(argv[1]) == "--stereo") {
    return run_stereo_subscriber(argv[2], argv[3]);
  }
  if (argc != 4) {
    std::cerr << "usage: python_core_schema_payload_peer <catalog.json> <input> <output>\n"
                 "       python_core_schema_payload_peer --channel-client <prefix>\n"
                 "       python_core_schema_payload_peer --stereo <catalog.json> <endpoint>\n";
    return 64;
  }
  auto catalog = holoscan::ChannelEndpointCatalog::load_json(argv[1]);
  if (!catalog) {
    std::cerr << "catalog load failed: " << holoscan::error_code_name(catalog.error().code) << '\n';
    return 1;
  }
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    std::cerr << "runtime creation failed: " << holoscan::error_code_name(runtime.error().code)
              << '\n';
    return 2;
  }
  auto publisher = runtime->create_publisher<ControlPayload>(*catalog, argv[2]);
  auto subscriber = runtime->create_subscriber<ControlPayload>(*catalog, argv[3]);
  if (!publisher || !subscriber) {
    std::cerr << "endpoint attachment failed\n";
    return 3;
  }
  ControlPayload value;
  value.sequence = 41U;
  value.gain = 1.25F;
  value.labels = {"cpp", "peer"};
  auto emitted = emit_when_attached(*publisher, value);
  if (!emitted) {
    std::cerr << "publish failed: " << holoscan::error_code_name(emitted.error().code) << '\n';
    return 4;
  }
  auto sample = receive_when_ready(*subscriber);
  if (!sample) {
    std::cerr << "receive failed: " << holoscan::error_code_name(sample.error().code) << '\n';
    return 5;
  }
  const bool valid = sample->data.sequence == 42U && sample->data.gain == 1.25F &&
                     sample->data.labels == std::vector<std::string>{"cpp", "peer"};
  std::cout << "sequence=" << sample->data.sequence << " labels=" << sample->data.labels.size()
            << '\n'
            << std::flush;
  // Keep the required peer endpoints alive until the graph has begun its orderly stop. Closing the
  // publisher sends end of stream, which would let the graph complete before the stop request. A
  // caller that does not provide a control pipe observes immediate EOF and retains the standalone
  // behavior.
  std::string release;
  (void)std::getline(std::cin, release);
  auto closed = publisher->close();
  if (!closed && closed.error().code != holoscan::ErrorCode::kNotReady) {
    std::cerr << "close failed: " << holoscan::error_code_name(closed.error().code) << '\n';
    return 6;
  }
  return valid ? 0 : 7;
}
