/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include <cuda/stream>

#include <holoscan/channel/manual_tensor_publisher.hpp>
#include <holoscan/channel/tensor_endpoints.hpp>
#include <holoscan/channel/tensor_loan.hpp>

namespace holoscan::channel_internal {

enum class ForeignTensorDisposition : std::uint8_t {
  kExportable,
  kCopyRequired,
};

struct StagedTensorCopy {
  explicit StagedTensorCopy(Tensor staged_tensor) : tensor(std::move(staged_tensor)) {}

  StagedTensorCopy(const StagedTensorCopy&) = delete;
  StagedTensorCopy& operator=(const StagedTensorCopy&) = delete;
  StagedTensorCopy(StagedTensorCopy&&) noexcept = default;
  StagedTensorCopy& operator=(StagedTensorCopy&&) noexcept = default;

  Tensor tensor;
};

// Source-private operation tables keep the installed facades independent of Runtime, pool, CUDA
// event, and HoloMQ control types. The admitted endpoint state binds through this seam without
// changing the public facade layout established by HSDK5-013.
struct TensorLoanWriteGuardOps {
  void (*abandon)(void*) noexcept;
  void* (*data)(const void*) noexcept;
  std::size_t (*byte_size)(const void*) noexcept;
  std::optional<::cuda::stream_ref> (*stream)(const void*) noexcept;
  expected<void, Error> (*commit)(void*) noexcept;
};

struct TensorPublishLoanOps {
  void (*abandon)(void*) noexcept;
  TensorShapeView (*shape)(const void*) noexcept;
  TensorByteStridesView (*byte_strides)(const void*) noexcept;
  DLDataType (*dtype)(const void*) noexcept;
  DeviceId (*device_id)(const void*) noexcept;
  MemoryKind (*memory_kind)(const void*) noexcept;
  std::size_t (*byte_size)(const void*) noexcept;
  expected<TensorLoanWriteGuard, Error> (*write)(void*, ::cuda::stream_ref) noexcept;
  expected<TensorLoanWriteGuard, Error> (*write_host)(void*) noexcept;
  expected<void, Error> (*cancel)(void*) noexcept;
};

struct TensorPublisherOps {
  void (*destroy)(void*) noexcept;
  expected<void, Error> (*close)(void*) noexcept;
  expected<TensorPublishLoan, Error> (*loan_tensor)(void*, TensorLoanRequest) noexcept;
  expected<std::uint32_t, Error> (*emit)(void*, TensorPublishLoan, EmitOptions) noexcept;
  expected<std::uint32_t, Error> (*emit_captured)(void*, TensorPublishLoan, std::int64_t,
                                                  EmitOptions) noexcept;
  expected<std::uint32_t, Error> (*emit_sample)(void*, TensorPublishLoan, SampleMetadata) noexcept;
  expected<std::uint32_t, Error> (*emit_copy)(void*, const Tensor&, ::cuda::stream_ref,
                                              EmitOptions) noexcept;
  expected<ForeignTensorDisposition, Error> (*classify_foreign_tensor)(const void*,
                                                                       const Tensor&) noexcept;
  expected<StagedTensorCopy, Error> (*stage_foreign_tensor)(void*, const Tensor&, cudaStream_t,
                                                            cudaStream_t) noexcept;
  expected<std::uint32_t, Error> (*emit_prevalidated_exportable_tensor)(void*, const Tensor&,
                                                                        SampleMetadata) noexcept;
  expected<TimePoint, Error> (*make_capture_time)(const void*, std::int64_t) noexcept;
  expected<void, Error> (*report_clock_health)(void*, ClockHealthReport) noexcept;
  expected<void, Error> (*update_connections)(void*) noexcept;
  bool (*has_subscribers)(const void*) noexcept;
  bool (*exhaust_operation_sequence_for_test)(void*) noexcept;
  bool (*fail_next_terminal_pre_submit_for_test)(void*) noexcept;
  void (*fail_next_post_submit_for_test)(void*, ErrorCode) noexcept;
  void (*fail_next_readiness_registration_for_test)(void*) noexcept;
  void (*corrupt_next_free_slot_for_test)(void*) noexcept;
  std::size_t (*preallocated_copy_event_count_for_test)(const void*) noexcept;
};

/// Construct source-private facades without exporting their control-state representation.
class TensorPublishLoanAccess {
 public:
  struct TakenLoan {
    void* state{};
    const TensorPublishLoanOps* ops{};
  };

  [[nodiscard]] static TensorLoanWriteGuard write_guard(
      void* state, const TensorLoanWriteGuardOps* ops) noexcept {
    return TensorLoanWriteGuard{state, ops};
  }

  [[nodiscard]] static TensorPublishLoan loan(void* state,
                                              const TensorPublishLoanOps* ops) noexcept {
    return TensorPublishLoan{state, ops};
  }

  [[nodiscard]] static ChannelPublisher<Tensor> publisher(void* state,
                                                          const TensorPublisherOps* ops) noexcept {
    return ChannelPublisher<Tensor>{state, ops};
  }

  [[nodiscard]] static expected<std::uint32_t, Error> emit_end_of_stream(
      ManualTensorPublisher& publisher, SampleMetadata metadata) noexcept {
    return publisher.emit_end_of_stream(metadata);
  }

  static void set_health_report_sequence(ManualTensorPublisher& publisher,
                                         const std::uint64_t sequence) noexcept {
    publisher.set_health_report_sequence(sequence);
  }

  [[nodiscard]] static bool loan_bound_to(const TensorPublishLoan& loan,
                                          const void* state) noexcept {
    return loan.state_ == state;
  }

  [[nodiscard]] static expected<ForeignTensorDisposition, Error> classify_foreign_tensor(
      const ChannelPublisher<Tensor>& publisher, const Tensor& tensor) noexcept {
    if (publisher.state_ == nullptr || publisher.ops_ == nullptr ||
        publisher.ops_->classify_foreign_tensor == nullptr) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return publisher.ops_->classify_foreign_tensor(publisher.state_, tensor);
  }

  [[nodiscard]] static expected<StagedTensorCopy, Error> stage_foreign_tensor(
      ChannelPublisher<Tensor>& publisher, const Tensor& tensor,
      cudaStream_t producer_stream = nullptr, cudaStream_t copy_stream = nullptr) noexcept {
    if (publisher.state_ == nullptr || publisher.ops_ == nullptr ||
        publisher.ops_->stage_foreign_tensor == nullptr) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return publisher.ops_->stage_foreign_tensor(
        publisher.state_, tensor, producer_stream, copy_stream);
  }

  [[nodiscard]] static expected<std::uint32_t, Error> emit_prevalidated_exportable_tensor(
      ChannelPublisher<Tensor>& publisher, const Tensor& tensor, SampleMetadata metadata) noexcept {
    if (publisher.state_ == nullptr || publisher.ops_ == nullptr ||
        publisher.ops_->emit_prevalidated_exportable_tensor == nullptr) {
      return make_unexpected(Error{ErrorCode::kClosed});
    }
    return publisher.ops_->emit_prevalidated_exportable_tensor(
        publisher.state_, tensor, std::move(metadata));
  }

  /// Inject one source-tree-only acknowledgement loss after a completed catalog Tensor submit.
  ///
  /// This bounded seam does not bypass the transport send. It changes only the result returned to
  /// the caller after one successful backend submission so tests can prove
  /// publication-indeterminate authority and retention without exporting fault controls in the
  /// production API.
  [[nodiscard]] static bool fail_next_catalog_post_submit_acknowledgement_for_test(
      ChannelPublisher<Tensor>& publisher) noexcept {
    return fail_next_catalog_post_submit_for_test(publisher, ErrorCode::kPublicationIndeterminate);
  }

  /// Inject one source-tree-only terminal-loan exhaustion before a catalog Tensor submit.
  [[nodiscard]] static bool fail_next_catalog_terminal_pre_submit_for_test(
      ChannelPublisher<Tensor>& publisher) noexcept {
    if (publisher.state_ == nullptr || publisher.ops_ == nullptr ||
        publisher.ops_->fail_next_terminal_pre_submit_for_test == nullptr) {
      return false;
    }
    return publisher.ops_->fail_next_terminal_pre_submit_for_test(publisher.state_);
  }

  /// Set one source-tree-only catalog Tensor publisher to permanent sequence exhaustion.
  [[nodiscard]] static bool exhaust_catalog_operation_sequence_for_test(
      ChannelPublisher<Tensor>& publisher) noexcept {
    if (publisher.state_ == nullptr || publisher.ops_ == nullptr ||
        publisher.ops_->exhaust_operation_sequence_for_test == nullptr) {
      return false;
    }
    return publisher.ops_->exhaust_operation_sequence_for_test(publisher.state_);
  }

  /// Inject one source-tree-only post-submit result after a completed catalog Tensor submit.
  [[nodiscard]] static bool fail_next_catalog_post_submit_for_test(
      ChannelPublisher<Tensor>& publisher, const ErrorCode code) noexcept {
    if (publisher.state_ == nullptr || publisher.ops_ == nullptr ||
        publisher.ops_->fail_next_post_submit_for_test == nullptr) {
      return false;
    }
    publisher.ops_->fail_next_post_submit_for_test(publisher.state_, code);
    return true;
  }

  /// Inject one source-tree-only readiness-registration failure during writer commit.
  [[nodiscard]] static bool fail_next_catalog_readiness_registration_for_test(
      ChannelPublisher<Tensor>& publisher) noexcept {
    if (publisher.state_ == nullptr || publisher.ops_ == nullptr ||
        publisher.ops_->fail_next_readiness_registration_for_test == nullptr) {
      return false;
    }
    publisher.ops_->fail_next_readiness_registration_for_test(publisher.state_);
    return true;
  }

  /// Corrupt one source-tree-only free slot so tests can prove fail-closed removal.
  [[nodiscard]] static bool corrupt_next_catalog_free_slot_for_test(
      ChannelPublisher<Tensor>& publisher) noexcept {
    if (publisher.state_ == nullptr || publisher.ops_ == nullptr ||
        publisher.ops_->corrupt_next_free_slot_for_test == nullptr) {
      return false;
    }
    publisher.ops_->corrupt_next_free_slot_for_test(publisher.state_);
    return true;
  }

  /// Return the number of cold-bound producer-to-copy-stream ordering events.
  [[nodiscard]] static std::size_t preallocated_copy_event_count_for_test(
      const ChannelPublisher<Tensor>& publisher) noexcept {
    if (publisher.state_ == nullptr || publisher.ops_ == nullptr ||
        publisher.ops_->preallocated_copy_event_count_for_test == nullptr) {
      return 0U;
    }
    return publisher.ops_->preallocated_copy_event_count_for_test(publisher.state_);
  }

  /// Detach one public loan so a consuming publisher operation can validate and retire it.
  [[nodiscard]] static TakenLoan take(TensorPublishLoan& loan) noexcept {
    return TakenLoan{.state = std::exchange(loan.state_, nullptr),
                     .ops = std::exchange(loan.ops_, nullptr)};
  }

  [[nodiscard]] static void* state(const TensorPublishLoan& loan) noexcept { return loan.state_; }
};

}  // namespace holoscan::channel_internal
