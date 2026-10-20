/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/schema/descriptor_flatbuffer_payload.hpp>
#include <holoscan/schema/sdk_schemas.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/manual_clock.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "image_generated.h"

#include "channel/channel_common.hpp"
#include "channel/channel_payload_frame.hpp"
#include "channel/process_identity.hpp"
#include "core/graph_compiler_access.hpp"
#include "runtime/channel_runtime_binding_access.hpp"
#include "runtime/graph_exec.hpp"
#include "runtime/runtime_channel_binding_test_access.hpp"
#include "runtime/runtime_phase_counters.hpp"
#include "runtime_external_channel_test_support.hpp"
#include "runtime_test_graph.hpp"

namespace holoscan::test::external_channel {

using namespace std::chrono_literals;

[[nodiscard]] expected<Tensor, Error> make_external_image_tensor(const std::uint8_t value) {
  auto* storage = new (std::nothrow) std::uint8_t[8U];
  if (storage == nullptr) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  std::fill_n(storage, 8U, value);
  auto tensor = make_tensor(kExternalImageShape,
                            kExternalTensorDtype,
                            std::nullopt,
                            MemoryStorageType::kSystem,
                            storage,
                            [](void* pointer) -> expected<void, Error> {
                              delete[] static_cast<std::uint8_t*>(pointer);
                              return {};
                            });
  if (!tensor) {
    delete[] storage;
    return make_unexpected(std::move(tensor).error());
  }
  return tensor;
}

[[nodiscard]] DescriptorTensorTopicContract<schema::ImageT> external_image_contract() {
  DescriptorTensorTopicContract<schema::ImageT> result;
  result.field(kExternalImageData,
               TensorBoundaryContract{
                   .representation = {.memory_kind = MemoryKind::kHost,
                                      .dtype = kExternalTensorDtype,
                                      .rank = 2U,
                                      .strides = TensorStridePolicy::kContiguousOnly},
                   .bounds = tensor_bounds(kExternalImageShape),
                   .transfer = TensorTransferPolicy{.foreign_emit_policy =
                                                        InterfaceTensorEmitPolicy::kCopyIfNeeded},
               });
  return result;
}

[[nodiscard]] TopicOptions external_image_import_options(std::string_view endpoint_name) {
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.bounds.publisher = PublisherEndpointBounds{.max_loaned_samples = 1U};
  return options;
}

[[nodiscard]] TopicOptions optional_tensor_export_options(std::string_view endpoint_name) {
  TopicOptions options;
  options.endpoint_name = endpoint_name;
  options.bounds.subscriber =
      SubscriberEndpointBounds{.queue_depth = 1U, .max_borrowed_samples = 1U};
  options.tensor = TensorBoundaryContract{
      .representation = {.memory_kind = MemoryKind::kHost,
                         .dtype = kExternalTensorDtype,
                         .rank = 1U,
                         .strides = TensorStridePolicy::kContiguousOnly},
      .bounds =
          TensorBounds{
              .max_rank = 1U,
              .max_byte_span = 8U,
              .axes = kExternalTensorAxes,
          },
      .transfer =
          TensorTransferPolicy{.foreign_emit_policy = InterfaceTensorEmitPolicy::kCopyIfNeeded},
  };
  return options;
}

[[nodiscard]] std::string unique_endpoint_name() {
  static std::atomic<std::uint64_t> ordinal{};
  return "runtime_external_import_" + std::to_string(channel_internal::current_process_identity()) +
         "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "_" +
         std::to_string(ordinal.fetch_add(1U, std::memory_order_relaxed));
}

[[nodiscard]] expected<void, Error> publish_external_host_tensor(
    ChannelPublisher<Tensor>& publisher, std::optional<TimePoint> capture_time,
    const std::byte fill) {
  auto loan = publisher.loan_tensor(
      TensorLoanRequest{.shape = kExternalTensorShape, .dtype = kExternalTensorDtype});
  if (!loan) {
    return make_unexpected(std::move(loan).error());
  }
  auto writer = loan->write_host();
  if (!writer) {
    return make_unexpected(std::move(writer).error());
  }
  std::fill_n(static_cast<std::byte*>(writer->data()), writer->byte_size(), fill);
  auto committed = std::move(*writer).commit();
  if (!committed) {
    return make_unexpected(std::move(committed).error());
  }
  EmitOptions options;
  options.capture_time = capture_time;
  auto emitted = publisher.emit(std::move(*loan), options);
  return emitted ? expected<void, Error>{}
                 : expected<void, Error>{make_unexpected(std::move(emitted).error())};
}

[[nodiscard]] expected<channel_internal::CatalogReceivedOperation, Error>
wait_for_catalog_tensor_schema_operation(
    channel_internal::TensorSchemaPayloadSubscriber& subscriber,
    std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto operation = detail::RuntimeChannelBindingAccess::receive_catalog_tensor_schema_operation(
        subscriber, true);
    if (!operation || operation->kind != channel_internal::CatalogReceivedOperationKind::kNone) {
      return operation;
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

[[nodiscard]] bool wait_for_session_terminal(const RunSession& session,
                                             std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(1ms);
  }
  return !session.is_running();
}

[[nodiscard]] expected<Sample<std::uint32_t>, Error> wait_for_sample(
    ChannelSubscriber<std::uint32_t>& subscriber, std::chrono::milliseconds timeout) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto ready = subscriber.has_samples();
    if (!ready) {
      return make_unexpected(std::move(ready.error()));
    }
    if (*ready) {
      auto sample = subscriber.receive_sample();
      if (sample || sample.error().code != ErrorCode::kNotReady) {
        return sample;
      }
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

}  // namespace holoscan::test::external_channel
