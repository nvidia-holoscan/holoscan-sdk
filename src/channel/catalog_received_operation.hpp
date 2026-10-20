/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/channel/endpoint_interfaces.hpp>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace holoscan::channel_internal {

/// @brief Snapshot receiver-applied clock-health evidence for Runtime integration.
struct CatalogClockHealthState {
  bool has_report{};  ///< True after at least one ordered report was receiver-applied.
  ClockHealthAssessment assessment{ClockHealthAssessment::kHealthy};  ///< Last assessment.
  ClockHealthReason reason{ClockHealthReason::kNone};                 ///< Last primary reason.
  std::uint64_t producer_epoch{};  ///< Wire producer-incarnation token of the reporting producer.
  std::uint64_t publisher_operation_sequence{};  ///< Operation sequence of the last report.
  std::uint64_t report_sequence{};               ///< Contiguous accepted-report sequence.
};

/// @brief Classify one receiver-side operation in a catalog publisher's total order.
enum class CatalogReceivedOperationKind : std::uint8_t {
  kNone = 0,         ///< No next admissible operation is currently observable.
  kClockHealth = 1,  ///< One private clock-health control operation was applied.
  kData = 2,         ///< One application data operation was decoded.
  kDataPending = 3,  ///< Current admissible data remains staged until the caller can retain it.
  kEndOfStream = 4,  ///< Explicit ordered endpoint close; carries no application payload.
  kPeerIncarnationChanged = 5,  ///< Staged traffic belongs to another producer incarnation.
};

/// @brief Preserve one private ordered receive result for Runtime integration.
struct CatalogReceivedOperation {
  CatalogReceivedOperationKind kind{CatalogReceivedOperationKind::kNone};
  SampleMetadata metadata{};  ///< Valid for data and sequence-only end-of-stream operations.
  /// Exact application bytes decoded into the caller's destination; valid only for a copied data
  /// operation. Zero for a borrowed schema payload, which reports its extent through
  /// \ref schema_payload instead.
  std::size_t payload_size{};
  CatalogClockHealthState clock_health{};  ///< Valid only for a clock-health operation.
  /// Borrowed serialized schema payload, valid until the specialized subscriber releases it.
  std::span<const std::byte> schema_payload{};
  /// Materialized raw Tensor data, valid only for a raw-Tensor data operation.
  std::optional<Sample<Tensor>> tensor_sample{};
};

/// @brief Attribute retained catalog traffic to the accepted producer incarnation.
struct CatalogImportQuiescence {
  bool pending_current{};  ///< A retained head belongs to the accepted producer incarnation.
  bool pending_foreign{};  ///< A retained head belongs to another producer incarnation.
  std::uint64_t latched_producer_generation{};  ///< Zero until one operation is accepted.
};

/// @brief Source-private catalog lifecycle extension for Tensor-schema subscribers.
class CatalogTensorSchemaPayloadSubscriber : public TensorSchemaPayloadSubscriber {
 public:
  /// @brief Consume at most one ordered catalog data, health, or terminal operation.
  [[nodiscard]] virtual expected<CatalogReceivedOperation, Error> receive_catalog_operation(
      bool accept_data) noexcept = 0;
  /// @brief Report whether any catalog data, health, or terminal operation is pending.
  [[nodiscard]] virtual expected<bool, Error> has_pending_catalog_operation() const noexcept = 0;
  /// @brief Detach this subscriber from its current backend wait set.
  [[nodiscard]] virtual expected<void, Error> detach_catalog_wait_set() noexcept = 0;
  /// @brief Return the producer incarnation accepted by this catalog attachment.
  [[nodiscard]] virtual std::uint64_t latched_producer_generation() const noexcept = 0;
  /// @brief Fence one producer incarnation on a fresh replacement attachment.
  [[nodiscard]] virtual expected<void, Error> supersede_producer_generation(
      std::uint64_t generation) noexcept = 0;
  /// @brief Classify bounded pending traffic without accepting a payload.
  [[nodiscard]] virtual expected<CatalogImportQuiescence, Error> import_quiescence() noexcept = 0;
};

}  // namespace holoscan::channel_internal
