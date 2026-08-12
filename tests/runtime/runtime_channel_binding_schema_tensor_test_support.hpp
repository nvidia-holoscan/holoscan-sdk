/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/channel/channel.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/schema/flatbuffer_schema_package.hpp>

#include "compiler/execution_plan_state.hpp"
#include "runtime/predicate_evaluator.hpp"
#include "runtime/runtime_channel_binding.hpp"

#include "runtime_channel_binding_test_support.hpp"

namespace holoscan::detail::runtime_channel_binding_schema_tensor_test {

using namespace std::chrono_literals;
using namespace runtime_channel_binding_test;

template <std::size_t Size>
struct PredicateComputeSelectionFixture {
  std::array<InputSelection, Size> selections{};

  [[nodiscard]] static expected<std::span<const InputSelection>, Error> select(void* state,
                                                                               TimePoint) noexcept {
    auto& self = *static_cast<PredicateComputeSelectionFixture*>(state);
    return std::span<const InputSelection>{self.selections};
  }

  [[nodiscard]] PredicateComputeSelectionRequest request() noexcept {
    return {.planned_input_count = Size, .state = this, .select = &select};
  }
};

using GeneratedSchemaSourceOperator = EmptySource<schema::StringT>;
using GeneratedSchemaSinkOperator = EmptySink<schema::StringT>;
using ImageSourceOperator = EmptySource<schema::ImageT>;
using ImageSinkOperator = EmptySink<schema::ImageT>;
using TensorPacketSourceOperator = EmptySource<schema::TensorPacketT>;
using TensorPacketSinkOperator = EmptySink<schema::TensorPacketT>;
using TensorSourceOperator = EmptySource<Tensor>;
using TensorSinkOperator = EmptySink<Tensor>;

// Pause typed ingress after irreversible transport/import work but before cache commit for
// RawTensorIngressRetainsExactLossyDestinationAcrossMaterialization and
// DescriptorTensorSchemaIngressFencesSelectionAcrossSidecarAcquisition.
struct BlockingTypedIngressObserver {
  static void observe(void* const opaque) noexcept {
    auto& state = *static_cast<BlockingTypedIngressObserver*>(opaque);
    state.calls.fetch_add(1U, std::memory_order_relaxed);
    state.entered.store(true, std::memory_order_release);
    while (state.block.load(std::memory_order_acquire) &&
           !state.release.load(std::memory_order_acquire)) {
      std::this_thread::yield();
    }
  }

  std::atomic<std::uint32_t> calls{};
  std::atomic<bool> entered{};
  std::atomic<bool> block{};
  std::atomic<bool> release{};
};

class ContractTensorSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;

  Output<Tensor> output{"output"};
};

class DeviceContractTensorSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;

  Output<Tensor> output{"output"};
};

class RequireExportableTensorSourceOperator final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;

  Output<Tensor> output{"output"};
};

[[nodiscard]] const PortPlanRecord* planned_operator_port(const ExecutionPlanState& plan,
                                                          PortDirection direction);

// Reuse the scalar fixture's already-audited topology and finite capacities, replacing only the
// typed declarations and payload geometry needed to exercise the generated-schema codec family.
struct GeneratedSchemaSessionFixture {
  SessionFixture base;
  GeneratedSchemaSourceOperator source;
  GeneratedSchemaSinkOperator sink;
  OperatorSpec source_spec;
  OperatorSpec sink_spec;
  std::array<RuntimeChannelPortReplayRecord, 2> replay;

  GeneratedSchemaSessionFixture();
  void use_cross_process_edge();
};

// ImageT is a generated owning value whose bounded FlatBuffer metadata and embedded Tensor take
// separate transport paths. The process-local service remains one typed endpoint: metadata uses
// the fixed schema envelope while Tensor storage stays under the retained descriptor authority.
struct ImageSessionFixture {
  SessionFixture base;
  ImageSourceOperator source;
  ImageSinkOperator sink;
  ImageSinkOperator second_sink;
  OperatorSpec source_spec;
  OperatorSpec sink_spec;
  OperatorSpec second_sink_spec;
  std::array<RuntimeChannelPortReplayRecord, 3> replay;

  ImageSessionFixture();
  void enable_reuse_last();
  void enable_bracketed_window(std::uint32_t max_selected_samples);
  void use_direct_retained_delivery();
  void add_blocking_fanout();
  void use_one_slot_blocking_transport();
  void use_cross_process_staging(std::uint64_t max_byte_span,
                                 MemoryKind memory_kind = MemoryKind::kHost, DeviceId device = {});
};

// A heterogeneous descriptor root reuses the same service contract and staging pool family as
// Image; only its generated metadata operations differ.
struct TensorPacketSessionFixture {
  SessionFixture base;
  TensorPacketSourceOperator source;
  TensorPacketSinkOperator sink;
  OperatorSpec source_spec;
  OperatorSpec sink_spec;
  std::array<RuntimeChannelPortReplayRecord, 2> replay;

  TensorPacketSessionFixture();
  void use_cross_process_staging(std::uint64_t max_byte_span);
};

// Reuse the scalar fixture's topology and finite temporal capacities, but bind Tensor-specific
// runtime operations and a pass-through service. The service owns no direct-output pool: each
// published Tensor keeps its original backing, and Channel retains that backing through the same
// temporal, activation, and reuse claims exercised for scalar payloads.
struct TensorSessionFixture {
  SessionFixture base;
  TensorSourceOperator source;
  TensorSinkOperator sink;
  TensorSinkOperator second_sink;
  OperatorSpec source_spec;
  OperatorSpec sink_spec;
  OperatorSpec second_sink_spec;
  std::array<RuntimeChannelPortReplayRecord, 3> replay;

  TensorSessionFixture();
  void enable_latest_within();
  void enable_reuse_last();
  void enable_drain_all(std::uint32_t max_selected_samples);
  void enable_bracketed_window(std::uint32_t max_selected_samples);
  void use_direct_retained_delivery();
  void add_blocking_fanout();
  void use_one_slot_blocking_fanout();
  void use_one_slot_blocking_transport();
  void use_one_slot_drop_oldest_queue();
  void use_cross_process_edge();
};

struct OwnedTensorSample {
  Tensor tensor;
  const void* data{};
  std::shared_ptr<std::atomic<std::uint32_t>> releases;
};

[[nodiscard]] expected<OwnedTensorSample, Error> make_owned_tensor_sample(std::uint64_t value);
[[nodiscard]] expected<Tensor, Error> make_contract_test_tensor(
    std::span<const std::int64_t> shape, DLDataType dtype,
    std::span<const std::int64_t> byte_strides = {},
    MemoryStorageType storage_type = MemoryStorageType::kSystem);

struct PublishedTensorScript {
  std::vector<RuntimeEvent> events;
  std::vector<TemporalSampleRecord> records;
  std::vector<const void*> data;
  std::vector<std::shared_ptr<std::atomic<std::uint32_t>>> releases;
};

struct PublishedImageSample {
  RuntimeEvent event;
  TemporalSampleRecord record;
  const void* data{};
  std::shared_ptr<std::atomic<std::uint32_t>> releases;
};

struct PublishedImageTensor {
  RuntimeEvent event;
  TemporalSampleRecord record;
};

struct PublishedImageScript {
  std::vector<RuntimeEvent> events;
  std::vector<TemporalSampleRecord> records;
  std::vector<const void*> data;
  std::vector<std::shared_ptr<std::atomic<std::uint32_t>>> releases;
};

[[nodiscard]] expected<PublishedImageScript, Error> publish_image_script(
    RuntimeChannelSession& session, PortBinding& binding, ImageSourceOperator& source,
    std::span<const std::uint64_t> values, std::span<const std::int64_t> capture_times);
[[nodiscard]] expected<PublishedImageSample, Error> publish_image_sample(
    RuntimeChannelSession& session, PortBinding& binding, ImageSourceOperator& source,
    std::uint64_t value);
[[nodiscard]] expected<PublishedImageTensor, Error> publish_image_tensor(
    RuntimeChannelSession& session, PortBinding& binding, ImageSourceOperator& source,
    Tensor tensor);
[[nodiscard]] expected<PublishedImageTensor, Error> publish_tensor_packet_tensor(
    RuntimeChannelSession& session, PortBinding& binding, TensorPacketSourceOperator& source,
    Tensor tensor);

struct StreamTrackingState {
  std::atomic<std::uint32_t> live_handles{0U};
  std::atomic<std::uint32_t> consumer_syncs{0U};
  std::atomic<std::uint32_t> deallocation_streams{0U};
  std::atomic<std::uint32_t> destroyed{0U};
  cudaStream_t last_consumer_stream{};
  cudaStream_t last_deallocation_stream{};
};

class StreamTrackingTensorStorage final : public TensorMemoryReference {
 public:
  explicit StreamTrackingTensorStorage(std::shared_ptr<StreamTrackingState> state);
  ~StreamTrackingTensorStorage() override;

  void retain_tensor_handle() noexcept override;
  void release_tensor_handle() noexcept override;
  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override;
  [[nodiscard]] expected<void, Error> synchronize_for_consumer(cudaStream_t stream) override;
  bool set_deallocation_stream(cudaStream_t stream) override;

  alignas(std::uint64_t) std::array<std::byte, sizeof(std::uint64_t)> data{};

 private:
  std::shared_ptr<StreamTrackingState> state_;
};

struct HostCopyTrackingState {
  std::atomic<std::uint32_t> host_synchronizations{};
  std::array<std::byte, 16U> data{};
};

class HostCopyTrackingTensorStorage final : public TensorMemoryReference {
 public:
  explicit HostCopyTrackingTensorStorage(std::shared_ptr<HostCopyTrackingState> state);
  [[nodiscard]] expected<void, Error> synchronize_for_host() override;
  [[nodiscard]] void* data() const noexcept;

 private:
  std::shared_ptr<HostCopyTrackingState> state_;
};

struct HostCopyTrackingTensor {
  Tensor tensor;
  std::shared_ptr<HostCopyTrackingState> state;
};

[[nodiscard]] HostCopyTrackingTensor make_host_copy_tracking_tensor(std::size_t byte_offset = 0U);
[[nodiscard]] expected<Sample<Tensor>, Error> receive_catalog_tensor(
    ChannelSubscriber<Tensor>& subscriber);

template <PublishedPayload T>
[[nodiscard]] expected<void, Error> wait_for_catalog_subscriber(
    ChannelPublisher<T>& publisher, const std::chrono::milliseconds timeout = 2s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    auto refreshed = publisher.update_connections();
    if (!refreshed) {
      return make_unexpected(std::move(refreshed).error());
    }
    if (publisher.has_subscribers()) {
      return {};
    }
    std::this_thread::sleep_for(1ms);
  }
  return make_unexpected(Error{ErrorCode::kTimeout});
}

[[nodiscard]] expected<void, Error> wait_for_catalog_input(RuntimeChannelSession& session,
                                                           PortId input,
                                                           std::chrono::milliseconds timeout = 2s);
[[nodiscard]] expected<void, Error> publish_catalog_host_tensor(ChannelPublisher<Tensor>& publisher,
                                                                std::byte value);

struct StreamTrackingTensor {
  Tensor tensor;
  const void* data{};
  std::shared_ptr<StreamTrackingState> state;
};

[[nodiscard]] StreamTrackingTensor make_stream_tracking_tensor();
[[nodiscard]] expected<PublishedTensorScript, Error> publish_tensor_script(
    RuntimeChannelSession& session, PortBinding& binding, TensorSourceOperator& source,
    std::span<const std::uint64_t> values, std::span<const std::int64_t> capture_times);

}  // namespace holoscan::detail::runtime_channel_binding_schema_tensor_test
