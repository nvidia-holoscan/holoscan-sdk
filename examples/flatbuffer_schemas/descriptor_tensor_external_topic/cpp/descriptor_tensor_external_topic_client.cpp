/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/core/errors.hpp>

#include "external_topic_common.hpp"

namespace {

using namespace std::chrono_literals;

struct Arguments {
  std::filesystem::path catalog;
  std::string instance;
};

template <typename Publisher>
class TopicCloseGuard {
 public:
  explicit TopicCloseGuard(Publisher& publisher) noexcept : publisher_(publisher) {}
  TopicCloseGuard(const TopicCloseGuard&) = delete;
  TopicCloseGuard& operator=(const TopicCloseGuard&) = delete;
  ~TopicCloseGuard() { static_cast<void>(close()); }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> close() {
    if (!armed_) {
      return {};
    }
    armed_ = false;
    auto result = publisher_.close();
    if (!result && result.error().code != holoscan::ErrorCode::kNotReady &&
        result.error().code != holoscan::ErrorCode::kClosed) {
      return holoscan::make_unexpected(std::move(result).error());
    }
    return {};
  }

 private:
  Publisher& publisher_;
  bool armed_{true};
};

[[nodiscard]] bool parse_arguments(const int argc, char** argv, Arguments& result) {
  for (int index = 1; index + 1 < argc; index += 2) {
    const std::string_view option{argv[index]};
    if (option == "--catalog") {
      result.catalog = argv[index + 1];
    } else if (option == "--instance") {
      result.instance = argv[index + 1];
    } else {
      return false;
    }
  }
  return argc == 5 && !result.catalog.empty() && !result.instance.empty();
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> wait_for_connections(
    holoscan::ChannelTensorSchemaPublisher<descriptor_tensor_external_topic::Payload>& publisher,
    holoscan::ChannelTensorSchemaSubscriber<descriptor_tensor_external_topic::Payload>&
        subscriber) {
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = publisher.update_connections();
    if (!refreshed) {
      return holoscan::make_unexpected(std::move(refreshed).error());
    }
    if (publisher.has_subscribers() && subscriber.has_publishers()) {
      return {};
    }
    std::this_thread::sleep_for(1ms);
  }
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout});
}

[[nodiscard]] holoscan::expected<
    holoscan::ChannelResolvedTensorBundle<descriptor_tensor_external_topic::Payload>,
    holoscan::Error>
receive_result(holoscan::ChannelTensorSchemaSubscriber<descriptor_tensor_external_topic::Payload>&
                   subscriber) {
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto ready = subscriber.has_samples();
    if (!ready) {
      return holoscan::make_unexpected(std::move(ready).error());
    }
    if (*ready) {
      auto received = subscriber.receive_tensors();
      if (received || received.error().code != holoscan::ErrorCode::kNotReady) {
        return received;
      }
    }
    std::this_thread::sleep_for(1ms);
  }
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout});
}

[[nodiscard]] holoscan::expected<void, holoscan::Error> publish(
    holoscan::ChannelTensorSchemaPublisher<descriptor_tensor_external_topic::Payload>& publisher,
    const std::span<const std::byte> metadata, const std::span<const holoscan::Tensor> tensors) {
  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    auto emitted = publisher.emit(metadata, tensors);
    if (emitted) {
      return *emitted == 1U ? holoscan::expected<void, holoscan::Error>{}
                            : holoscan::expected<void, holoscan::Error>{holoscan::make_unexpected(
                                  holoscan::Error{holoscan::ErrorCode::kProtocolError,
                                                  "unexpected external-topic recipient count"})};
    }
    if (emitted.error().code != holoscan::ErrorCode::kNotReady &&
        emitted.error().code != holoscan::ErrorCode::kBackpressured) {
      return holoscan::make_unexpected(std::move(emitted).error());
    }
    auto refreshed = publisher.update_connections();
    if (!refreshed) {
      return holoscan::make_unexpected(std::move(refreshed).error());
    }
    std::this_thread::sleep_for(1ms);
  }
  return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kTimeout});
}

}  // namespace

int main(int argc, char** argv) {
  Arguments arguments;
  if (!parse_arguments(argc, argv, arguments)) {
    std::cerr << "usage: descriptor_tensor_external_topic_client --catalog PATH --instance NAME\n";
    return 2;
  }

  auto catalog = holoscan::ChannelEndpointCatalog::load_json(arguments.catalog);
  if (!catalog) {
    std::cerr << "failed to load endpoint catalog: "
              << holoscan::runtime_error_message(catalog.error()) << '\n';
    return 3;
  }
  auto runtime = holoscan::ChannelRuntime::create();
  if (!runtime) {
    std::cerr << "failed to create ChannelRuntime: "
              << holoscan::runtime_error_message(runtime.error()) << '\n';
    return 4;
  }
  const std::string import_name =
      descriptor_tensor_external_topic::import_endpoint_name(arguments.instance);
  const std::string export_name =
      descriptor_tensor_external_topic::export_endpoint_name(arguments.instance);
  auto publisher =
      runtime->create_tensor_schema_publisher<descriptor_tensor_external_topic::Payload>(
          *catalog, import_name);
  if (!publisher) {
    std::cerr << "failed to attach publisher: "
              << holoscan::runtime_error_message(publisher.error()) << '\n';
    return 5;
  }
  // Every exit after attachment attempts explicit EOS; kNotReady/kClosed are benign when the
  // graph has already begun tearing down after the acknowledgment.
  TopicCloseGuard close_guard{*publisher};
  auto subscriber =
      runtime->create_tensor_schema_subscriber<descriptor_tensor_external_topic::Payload>(
          *catalog, export_name);
  if (!subscriber) {
    std::cerr << "failed to attach subscriber: "
              << holoscan::runtime_error_message(subscriber.error()) << '\n';
    return 6;
  }
  if (auto connected = wait_for_connections(*publisher, *subscriber); !connected) {
    std::cerr << "external endpoints did not connect: "
              << holoscan::runtime_error_message(connected.error()) << '\n';
    return 7;
  }

  auto tensors = descriptor_tensor_external_topic::make_client_tensors();
  if (!tensors) {
    std::cerr << "failed to create client Tensors: "
              << holoscan::runtime_error_message(tensors.error()) << '\n';
    return 8;
  }
  auto metadata = descriptor_tensor_external_topic::build_metadata(
      descriptor_tensor_external_topic::kDataSequence,
      descriptor_tensor_external_topic::kClientSource,
      *tensors);
  if (!metadata) {
    std::cerr << "failed to build schema metadata: "
              << holoscan::runtime_error_message(metadata.error()) << '\n';
    return 9;
  }
  if (auto emitted = publish(*publisher, *metadata, *tensors); !emitted) {
    std::cerr << "failed to publish input frame: "
              << holoscan::runtime_error_message(emitted.error()) << '\n';
    return 10;
  }

  auto received = receive_result(*subscriber);
  if (!received || received->root() == nullptr ||
      received->root()->sequence() != descriptor_tensor_external_topic::kDataSequence ||
      received->root()->source() == nullptr ||
      received->root()->source()->string_view() != descriptor_tensor_external_topic::kGraphSource ||
      !descriptor_tensor_external_topic::tensors_match(received->tensors)) {
    std::cerr << "received descriptor-schema frame is invalid";
    if (!received) {
      std::cerr << ": " << holoscan::runtime_error_message(received.error());
    }
    std::cerr << '\n';
    return 11;
  }

  auto ack_metadata = descriptor_tensor_external_topic::build_metadata(
      descriptor_tensor_external_topic::kAckSequence,
      descriptor_tensor_external_topic::kClientSource,
      *tensors);
  if (!ack_metadata) {
    std::cerr << "failed to build acknowledgment metadata: "
              << holoscan::runtime_error_message(ack_metadata.error()) << '\n';
    return 12;
  }
  if (auto acknowledged = publish(*publisher, *ack_metadata, *tensors); !acknowledged) {
    std::cerr << "failed to publish acknowledgment: "
              << holoscan::runtime_error_message(acknowledged.error()) << '\n';
    return 13;
  }
  if (auto closed = close_guard.close(); !closed) {
    std::cerr << "failed to close input topic: " << holoscan::runtime_error_message(closed.error())
              << '\n';
    return 14;
  }

  std::cout << "descriptor Tensor external-topic client complete\n";
  return 0;
}
