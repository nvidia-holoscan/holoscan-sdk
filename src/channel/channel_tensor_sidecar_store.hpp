/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <holomq/context_builder.hpp>
#include <holomq/delivery_attribution.hpp>
#include <holomq/errors.hpp>
#include <holomq/inbound_message.hpp>
#include <holomq/outbound_message.hpp>
#include <holomq/publisher.hpp>
#include <holomq/pubsub_capabilities.hpp>
#include <holomq/pubsub_options.hpp>
#include <holomq/subscriber.hpp>
#include <holoscan/channel/channel.hpp>

#include "../core/domain/tensor_access.hpp"
#include "channel_common.hpp"
#include "process_identity.hpp"
namespace holoscan {

using channel_internal::make_channel_error;

class TensorSidecarStore {
 public:
  TensorSidecarStore(std::size_t capacity, std::uint32_t expected_recipient_count)
      : expected_recipient_count_(expected_recipient_count),
        full_claim_mask_(claim_mask_for(expected_recipient_count)),
        local_process_id_(channel_internal::current_process_identity()),
        records_(std::max<std::size_t>(capacity, 1U)),
        recipient_ids_(expected_recipient_count) {
    initialize_free_stack();
  }

  [[nodiscard]] std::uint32_t expected_recipient_count() const noexcept {
    return expected_recipient_count_;
  }

  [[nodiscard]] expected<std::size_t, Error> register_recipient(
      const holomq::DeliveryRecipientId& recipient_id) {
    const std::lock_guard<std::mutex> lock(setup_mutex_);
    std::optional<std::size_t> available_index;
    for (std::size_t index = 0U; index < recipient_ids_.size(); ++index) {
      if (recipient_ids_[index] == recipient_id) {
        return index;
      }
      if (!available_index && recipient_ids_[index] == holomq::DeliveryRecipientId{}) {
        available_index = index;
      }
    }
    if (registered_recipient_count_ >= recipient_ids_.size() || !available_index) {
      return make_unexpected(make_channel_error(
          ErrorCode::kResourceExhausted,
          "Tensor channel has more subscribers than its expected recipient count"));
    }
    const std::size_t index = *available_index;
    recipient_ids_[index] = recipient_id;
    ++registered_recipient_count_;
    registered_recipient_count_atomic_.store(registered_recipient_count_,
                                             std::memory_order_release);
    return index;
  }

  /// Release one registered recipient after its endpoint is destroyed.
  ///
  /// Recipient indices are claim-bit positions and therefore are never compacted while other
  /// recipients remain registered. Releasing one endpoint makes its exact hole available to a
  /// replacement without changing any surviving claim identity.
  void unregister_recipient(const std::size_t recipient_index,
                            const holomq::DeliveryRecipientId& recipient_id) noexcept {
    {
      const std::lock_guard<std::mutex> lock(setup_mutex_);
      if (recipient_index >= recipient_ids_.size() || registered_recipient_count_ == 0U ||
          recipient_ids_[recipient_index] != recipient_id) {
        return;
      }
      recipient_ids_[recipient_index] = {};
      --registered_recipient_count_;
      registered_recipient_count_atomic_.store(registered_recipient_count_,
                                               std::memory_order_release);
    }
    // Endpoint teardown is a bounded cold lifecycle path. Clear this recipient's unread records
    // after making new inserts fail closed, and do not retain the setup lock while Tensor backing
    // and its application-defined memory reference are released.
    release_unread_claims_for_recipient(recipient_index);
  }

  [[nodiscard]] expected<ChannelSampleDescriptor, Error> insert(Tensor tensor,
                                                                SampleMetadata metadata = {}) {
    const detail::TensorAccessSnapshot tensor_snapshot = detail::TensorAccess::snapshot(tensor);
    if (!tensor_snapshot.initialized || !tensor_snapshot.valid) {
      return make_unexpected(
          make_channel_error(ErrorCode::kInvalidArgument, "cannot emit an invalid Tensor"));
    }
    // Tensor sidecar records carry per-recipient claim bits. HoloMQ send currently reports a
    // delivered count, not the delivered-recipient set, so this first slice requires static
    // recipient registration before publish and leaves dynamic fanout changes to the runtime
    // lifecycle contract.
    const std::size_t registered_recipient_count =
        registered_recipient_count_atomic_.load(std::memory_order_acquire);
    if (registered_recipient_count != expected_recipient_count_) {
      return make_unexpected(make_channel_error(
          ErrorCode::kNotReady,
          "Tensor channel has not registered all expected recipients before publish"));
    }
    const auto maybe_index = pop_free_index();
    if (!maybe_index) {
      return make_unexpected(
          make_channel_error(ErrorCode::kResourceExhausted, "Tensor sidecar store is full"));
    }
    const std::size_t index = maybe_index.value();
    Record& record = records_[index];
    RecordState expected = RecordState::kFree;
    if (!record.state.compare_exchange_strong(expected,
                                              RecordState::kReserved,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
      return make_unexpected(make_channel_error(
          ErrorCode::kFailure, "Tensor sidecar free list contained a non-free record"));
    }

    std::uint64_t generation = next_generation_.fetch_add(1U, std::memory_order_relaxed);
    if (generation == 0U) {
      generation = next_generation_.fetch_add(1U, std::memory_order_relaxed);
    }
    const std::uint64_t sequence = metadata.sequence != 0U
                                       ? metadata.sequence
                                       : next_sequence_.fetch_add(1U, std::memory_order_relaxed);
    const std::uint64_t token = static_cast<std::uint64_t>(index + 1U);
    const auto descriptor = ChannelSampleDescriptor{
        .magic = kChannelSampleDescriptorMagic,
        .abi_version = kChannelSampleDescriptorAbiVersion,
        .payload_kind = ChannelPayloadKind::kLocalBufferDescriptor,
        .flags = 0U,
        .expected_recipient_count = expected_recipient_count_,
        .process_id = local_process_id_,
        .sidecar_token = token,
        .generation = generation,
        .schema_id = SchemaId{},
        .sequence = sequence,
        .byte_size = tensor_snapshot.byte_size,
    };

    metadata.sequence = descriptor.sequence;
    metadata.byte_size = descriptor.byte_size;
    metadata.payload_kind = PayloadKind::kDescriptorBacked;
    record.tensor = std::move(tensor);
    record.metadata = metadata;
    record.remaining_claim_mask.store(full_claim_mask_, std::memory_order_release);
    record.unread_claim_mask.store(full_claim_mask_, std::memory_order_release);
    record.generation.store(generation, std::memory_order_release);
    record.state.store(RecordState::kPublished, std::memory_order_release);
    return descriptor;
  }

  [[nodiscard]] expected<Tensor, Error> acquire(const ChannelSampleDescriptor& descriptor,
                                                std::size_t recipient_index) {
    auto sample = acquire_sample(descriptor, recipient_index);
    if (!sample) {
      return make_unexpected(std::move(sample).error());
    }
    return std::move(sample).value().data;
  }

  [[nodiscard]] expected<Sample<Tensor>, Error> acquire_sample(
      const ChannelSampleDescriptor& descriptor, std::size_t recipient_index) {
    if (auto valid = validate_descriptor(descriptor); !valid) {
      return make_unexpected(std::move(valid).error());
    }
    if (recipient_index >= expected_recipient_count_) {
      return make_unexpected(make_channel_error(ErrorCode::kInvalidArgument,
                                                "Tensor channel recipient index is out of range"));
    }

    Record* record = find_record(descriptor);
    if (record == nullptr) {
      return make_unexpected(
          make_channel_error(ErrorCode::kReceiveError, "Tensor sidecar descriptor is stale"));
    }
    if (!try_enter_record(*record, descriptor)) {
      return make_unexpected(
          make_channel_error(ErrorCode::kReceiveError, "Tensor sidecar descriptor is stale"));
    }
    const std::uint64_t recipient_bit = recipient_bit_mask(recipient_index);
    if (recipient_bit == 0U) {
      return make_unexpected(
          make_channel_error(ErrorCode::kInvalidArgument, "recipient index exceeds mask width"));
    }
    const std::uint64_t previous_unread =
        record->unread_claim_mask.fetch_and(~recipient_bit, std::memory_order_acq_rel);
    if ((previous_unread & recipient_bit) == 0U) {
      leave_record(*record);
      return make_unexpected(make_channel_error(
          ErrorCode::kReceiveError, "Tensor sidecar descriptor has no unread recipient claim"));
    }

    // The recipient's remaining claim is still set while this lightweight Tensor handle is copied.
    // That prevents drop-oldest cleanup for other recipients from clearing the record before the
    // receiver owns its independent DLPack context reference.
    Tensor tensor = record->tensor;
    const SampleMetadata metadata = record->metadata;
    release_remaining_claim(*record, descriptor, recipient_bit);
    leave_record(*record);
    return Sample<Tensor>(std::move(tensor), metadata);
  }

  void release_replaced(const ChannelSampleDescriptor& descriptor,
                        const holomq::DeliveryRecipientId& recipient_id) noexcept {
    try {
      const auto recipient_index = find_recipient_index(recipient_id);
      if (!recipient_index) {
        return;
      }
      Record* record = find_record(descriptor);
      if (record != nullptr && try_enter_record(*record, descriptor)) {
        release_unread_claim_for_recipient(*record, descriptor, recipient_index.value());
        leave_record(*record);
      }
    } catch (...) {
      return;
    }
  }

  void release_all(const ChannelSampleDescriptor& descriptor) noexcept {
    try {
      Record* record = find_record(descriptor);
      if (record != nullptr && try_enter_record(*record, descriptor)) {
        force_release(*record, descriptor);
        leave_record(*record);
      }
    } catch (...) {
      return;
    }
  }

  void release_recipient_claim(const ChannelSampleDescriptor& descriptor,
                               std::size_t recipient_index) noexcept {
    try {
      if (recipient_index >= expected_recipient_count_) {
        return;
      }
      Record* record = find_record(descriptor);
      if (record != nullptr && try_enter_record(*record, descriptor)) {
        release_unread_claim_for_recipient(*record, descriptor, recipient_index);
        leave_record(*record);
      }
    } catch (...) {
      return;
    }
  }

 private:
  enum class RecordState : std::uint8_t {
    kFree,
    kReserved,
    kPublished,
    kClearing,
    kFinalizing,
  };

  static constexpr std::uint32_t kInvalidFreeIndex = std::numeric_limits<std::uint32_t>::max();

  struct Record {
    std::atomic<RecordState> state{RecordState::kFree};
    std::atomic<std::uint64_t> generation{0U};
    std::atomic<std::uint32_t> active_readers{0U};
    std::atomic<std::uint32_t> next_free{kInvalidFreeIndex};
    Tensor tensor{};
    SampleMetadata metadata{};
    std::atomic<std::uint64_t> remaining_claim_mask{0U};
    std::atomic<std::uint64_t> unread_claim_mask{0U};
  };

  static_assert(sizeof(std::uint64_t) >= 2U * sizeof(std::uint32_t));

  [[nodiscard]] static constexpr std::uint64_t pack_free_head(std::uint32_t index,
                                                              std::uint32_t tag) noexcept {
    return (static_cast<std::uint64_t>(tag) << 32U) | static_cast<std::uint64_t>(index);
  }

  [[nodiscard]] static constexpr std::uint32_t free_head_index(std::uint64_t head) noexcept {
    return static_cast<std::uint32_t>(head & 0xffffffffULL);
  }

  [[nodiscard]] static constexpr std::uint32_t free_head_tag(std::uint64_t head) noexcept {
    return static_cast<std::uint32_t>(head >> 32U);
  }

  void initialize_free_stack() noexcept {
    for (std::size_t index = 0U; index < records_.size(); ++index) {
      const std::uint32_t next =
          index + 1U < records_.size() ? static_cast<std::uint32_t>(index + 1U) : kInvalidFreeIndex;
      records_[index].next_free.store(next, std::memory_order_relaxed);
    }
    free_head_.store(pack_free_head(0U, 0U), std::memory_order_release);
  }

  [[nodiscard]] std::optional<std::size_t> pop_free_index() noexcept {
    std::uint64_t head = free_head_.load(std::memory_order_acquire);
    while (true) {
      const std::uint32_t index = free_head_index(head);
      if (index == kInvalidFreeIndex) {
        return std::nullopt;
      }
      const std::uint32_t next = records_[index].next_free.load(std::memory_order_acquire);
      const std::uint64_t desired = pack_free_head(next, free_head_tag(head) + 1U);
      if (free_head_.compare_exchange_weak(
              head, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
        records_[index].next_free.store(kInvalidFreeIndex, std::memory_order_release);
        return static_cast<std::size_t>(index);
      }
    }
  }

  void push_free_index(std::size_t index) noexcept {
    const auto free_index = static_cast<std::uint32_t>(index);
    std::uint64_t head = free_head_.load(std::memory_order_acquire);
    while (true) {
      records_[index].next_free.store(free_head_index(head), std::memory_order_release);
      const std::uint64_t desired = pack_free_head(free_index, free_head_tag(head) + 1U);
      if (free_head_.compare_exchange_weak(
              head, desired, std::memory_order_acq_rel, std::memory_order_acquire)) {
        return;
      }
    }
  }

  [[nodiscard]] std::size_t record_index(const Record& record) const noexcept {
    return static_cast<std::size_t>(&record - records_.data());
  }

  [[nodiscard]] expected<void, Error> validate_descriptor(
      const ChannelSampleDescriptor& descriptor) const {
    if (!descriptor.has_supported_abi()) {
      return make_unexpected(make_channel_error(ErrorCode::kReceiveError,
                                                "Tensor sidecar descriptor ABI is not supported"));
    }
    if (descriptor.payload_kind != ChannelPayloadKind::kLocalBufferDescriptor) {
      return make_unexpected(make_channel_error(
          ErrorCode::kReceiveError, "Tensor sidecar descriptor has unexpected payload kind"));
    }
    if (descriptor.process_id != local_process_id_) {
      return make_unexpected(make_channel_error(
          ErrorCode::kReceiveError, "Tensor sidecar descriptor is not from this process"));
    }
    if (descriptor.sidecar_token == 0U) {
      return make_unexpected(
          make_channel_error(ErrorCode::kReceiveError, "Tensor sidecar descriptor is empty"));
    }
    return {};
  }

  [[nodiscard]] Record* find_record(const ChannelSampleDescriptor& descriptor) noexcept {
    if (descriptor.sidecar_token == 0U || descriptor.sidecar_token > records_.size()) {
      return nullptr;
    }
    Record& record = records_[static_cast<std::size_t>(descriptor.sidecar_token - 1U)];
    if (record.state.load(std::memory_order_acquire) != RecordState::kPublished ||
        record.generation.load(std::memory_order_acquire) != descriptor.generation) {
      return nullptr;
    }
    return &record;
  }

  [[nodiscard]] bool try_enter_record(Record& record,
                                      const ChannelSampleDescriptor& descriptor) noexcept {
    if (record.state.load(std::memory_order_acquire) != RecordState::kPublished ||
        record.generation.load(std::memory_order_acquire) != descriptor.generation) {
      return false;
    }
    record.active_readers.fetch_add(1U, std::memory_order_acq_rel);
    if (record.state.load(std::memory_order_acquire) == RecordState::kPublished &&
        record.generation.load(std::memory_order_acquire) == descriptor.generation) {
      return true;
    }
    leave_record(record);
    return false;
  }

  void leave_record(Record& record) noexcept {
    const std::uint32_t previous_readers =
        record.active_readers.fetch_sub(1U, std::memory_order_acq_rel);
    if (previous_readers == 1U) {
      finish_clearing_if_idle(record);
    }
  }

  void force_release(Record& record, const ChannelSampleDescriptor& descriptor) noexcept {
    if (record.generation.load(std::memory_order_acquire) != descriptor.generation) {
      return;
    }
    record.remaining_claim_mask.store(0U, std::memory_order_release);
    record.unread_claim_mask.store(0U, std::memory_order_release);
    try_begin_clearing(record, descriptor);
  }

  void try_begin_clearing(Record& record, const ChannelSampleDescriptor& descriptor) noexcept {
    if (record.generation.load(std::memory_order_acquire) != descriptor.generation ||
        record.remaining_claim_mask.load(std::memory_order_acquire) != 0U) {
      return;
    }
    RecordState expected = RecordState::kPublished;
    if (!record.state.compare_exchange_strong(expected,
                                              RecordState::kClearing,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
      return;
    }
    finish_clearing_if_idle(record);
  }

  void finish_clearing_if_idle(Record& record) noexcept {
    if (record.active_readers.load(std::memory_order_acquire) != 0U) {
      return;
    }
    RecordState expected = RecordState::kClearing;
    if (!record.state.compare_exchange_strong(expected,
                                              RecordState::kFinalizing,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) {
      return;
    }
    record.tensor = Tensor{};
    record.remaining_claim_mask.store(0U, std::memory_order_release);
    record.unread_claim_mask.store(0U, std::memory_order_release);
    record.generation.store(0U, std::memory_order_release);
    record.state.store(RecordState::kFree, std::memory_order_release);
    push_free_index(record_index(record));
  }

  void release_unread_claim_for_recipient(Record& record, const ChannelSampleDescriptor& descriptor,
                                          std::size_t recipient_index) noexcept {
    const std::uint64_t recipient_bit = recipient_bit_mask(recipient_index);
    if (recipient_bit == 0U) {
      return;
    }
    const std::uint64_t previous_unread =
        record.unread_claim_mask.fetch_and(~recipient_bit, std::memory_order_acq_rel);
    if ((previous_unread & recipient_bit) == 0U) {
      return;
    }
    release_remaining_claim(record, descriptor, recipient_bit);
  }

  void release_unread_claims_for_recipient(const std::size_t recipient_index) noexcept {
    for (std::size_t index = 0U; index < records_.size(); ++index) {
      Record& record = records_[index];
      const std::uint64_t generation = record.generation.load(std::memory_order_acquire);
      if (generation == 0U) {
        continue;
      }
      const ChannelSampleDescriptor descriptor{
          .sidecar_token = static_cast<std::uint64_t>(index + 1U),
          .generation = generation,
      };
      if (try_enter_record(record, descriptor)) {
        release_unread_claim_for_recipient(record, descriptor, recipient_index);
        leave_record(record);
      }
    }
  }

  void release_remaining_claim(Record& record, const ChannelSampleDescriptor& descriptor,
                               std::uint64_t recipient_bit) noexcept {
    const std::uint64_t previous_remaining =
        record.remaining_claim_mask.fetch_and(~recipient_bit, std::memory_order_acq_rel);
    if ((previous_remaining & recipient_bit) == 0U || (previous_remaining & ~recipient_bit) != 0U) {
      return;
    }
    try_begin_clearing(record, descriptor);
  }

  [[nodiscard]] std::optional<std::size_t> find_recipient_index(
      const holomq::DeliveryRecipientId& recipient_id) const {
    const std::lock_guard<std::mutex> lock(setup_mutex_);
    for (std::size_t index = 0U; index < recipient_ids_.size(); ++index) {
      if (recipient_ids_[index] == recipient_id) {
        return index;
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] static constexpr std::uint64_t claim_mask_for(
      std::uint32_t recipient_count) noexcept {
    return recipient_count >= 64U ? ~std::uint64_t{0U}
                                  : ((std::uint64_t{1U} << recipient_count) - 1U);
  }

  [[nodiscard]] static constexpr std::uint64_t recipient_bit_mask(
      std::size_t recipient_index) noexcept {
    constexpr std::size_t kMaskWidth = sizeof(std::uint64_t) * 8U;
    return recipient_index < kMaskWidth ? std::uint64_t{1U} << recipient_index : std::uint64_t{0U};
  }

  std::uint32_t expected_recipient_count_{};
  std::uint64_t full_claim_mask_{};
  std::uint64_t local_process_id_{};
  mutable std::mutex setup_mutex_;
  std::vector<Record> records_;
  std::vector<holomq::DeliveryRecipientId> recipient_ids_;
  std::size_t registered_recipient_count_{0U};
  std::atomic<std::size_t> registered_recipient_count_atomic_{0U};
  std::atomic<std::uint64_t> free_head_{pack_free_head(kInvalidFreeIndex, 0U)};
  std::atomic<std::uint64_t> next_generation_{1U};
  std::atomic<std::uint64_t> next_sequence_{1U};
};
}  // namespace holoscan
