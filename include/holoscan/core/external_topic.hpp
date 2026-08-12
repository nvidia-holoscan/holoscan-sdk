/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>

#include <holoscan/channel/clock_health.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/schema_identity.hpp>
#include <holoscan/time/clock_registry.hpp>

namespace holoscan {

class Graph;

namespace detail {
class GraphState;
}  // namespace detail

/// @brief Require capture timestamps in one declared clock domain.
struct RequireCaptureTime {
  ClockKey clock;  ///< Cold named/default clock key copied into Graph state.
};

/// @brief Select whether absence of an external peer is admissible.
enum class PeerRequirement : std::uint8_t {
  kOptional = 0,  ///< Graph start never waits for or fails on an absent peer.
  kRequired = 1,  ///< Explicit attach/wait may report timeout at the accepted bound.
};

/// @brief Carry the endpoint peer requirement and finite attach/wait bound.
///
/// The default optional peer never makes Graph composition or Graph start wait
/// for an external process. A positive \ref max_wait is retained only as a bound
/// for later explicit attach/service verification and required-peer waiting.
struct TopicAttachPolicy {
  PeerRequirement peer{PeerRequirement::kOptional};  ///< Whether an absent external peer is valid.
  std::optional<std::chrono::nanoseconds> max_wait;  ///< Positive finite attach bound when present.
};

/// @brief Classify logical Tensor stride contracts at an external boundary.
enum class TensorStridePolicy : std::uint8_t {
  kContiguousOnly = 0,  ///< Only canonical contiguous logical layout is admitted.
  kBoundedStrided = 1,  ///< Noncontiguous layout is admitted within explicit axis bounds.
};

/// @brief Bound one logical Tensor axis at an external boundary.
struct TensorAxisBounds {
  std::uint64_t max_extent{};            ///< Maximum logical extent; zero permits zero extent.
  std::uint64_t max_abs_stride_bytes{};  ///< Maximum absolute byte stride.
};

/// @brief Bound logical Tensor rank, byte span, and per-axis layout evidence.
///
/// An exact rank, when present, must not exceed \ref max_rank. Rank zero and
/// zero extents are valid logical values. Contiguous layouts may retain partial
/// axis evidence; bounded-strided layouts require exactly \ref max_rank axis
/// records and a positive maximum absolute stride for every axis.
struct LogicalTensorBounds {
  std::uint32_t max_rank{};       ///< Positive maximum rank; rank-zero payloads remain valid.
  std::uint64_t max_byte_span{};  ///< Positive maximum validated logical byte span.
  std::span<const TensorAxisBounds> axes;  ///< Call-lifetime axis bounds copied by Graph.
};

/// @brief Carry complete logical Tensor requirements for an external topic.
///
/// Omitting this carrier supplies no explicit boundary-side emission policy.
/// When present, its policy/cap participates in normalization with producer
/// \ref PayloadOptions: one explicit side wins, identical sides agree, and a
/// mismatch rejects the edge. An inline cap is required only for the inline
/// policy and is rejected for both non-inline policies.
struct LogicalTensorRequirements {
  std::optional<MemoryKind> memory_kind;  ///< Exact logical storage kind when constrained.
  std::optional<DLDataType> dtype;        ///< Exact DLPack element type when constrained.
  std::optional<std::uint32_t> rank;      ///< Exact rank when constrained; zero denotes a scalar.
  std::optional<LogicalTensorBounds> logical_bounds;  ///< Finite rank/span/axis bounds.
  TensorStridePolicy strides{TensorStridePolicy::kContiguousOnly};  ///< Admitted stride family.
  InterfaceTensorEmitPolicy foreign_emit_policy{
      InterfaceTensorEmitPolicy::kRequireExportable};  ///< Non-exportable storage handling.
  std::optional<std::uint32_t> max_inline_bytes;  ///< Positive cap required only by inline policy.
};

/// @brief Bound publisher-local external endpoint claims.
struct PublisherEndpointBounds {
  std::optional<std::uint32_t> max_loaned_samples;  ///< Positive finite external publisher claims.
};

/// @brief Bound subscriber-local external endpoint queues and claims.
struct SubscriberEndpointBounds {
  std::optional<std::uint32_t> queue_depth;  ///< Positive finite external subscriber queue depth.
  std::optional<std::uint32_t> max_borrowed_samples;  ///< Positive finite read claims.
};

/// @brief Carry the direction-specific external endpoint bounds.
struct EndpointBounds {
  std::optional<PublisherEndpointBounds> publisher;    ///< External publisher bounds for imports.
  std::optional<SubscriberEndpointBounds> subscriber;  ///< External subscriber bounds for exports.
};

/// @brief Carry the complete backend-neutral external topic authoring contract.
///
/// Every string/span is a call-lifetime input copied during Graph normalization.
/// This type intentionally exposes no backend option bag, raw handle, queue
/// policy, device ordinal, or live health state. Graph topic declarations admit
/// only an exact nonempty `holomq://` URI. The URI remains descriptive and does
/// not grant runtime attachment authority.
struct TopicOptions {
  std::optional<std::string_view> endpoint_name;  ///< Explicit deployment-stable name to copy.
  std::optional<MemoryConversionPolicy> memory_conversion;  ///< Tensor conversion request.
  std::optional<SchemaIdentity> schema;  ///< Complete schema override for Tensor or schema root.
  std::optional<RequireCaptureTime> capture_time;  ///< Required capture-time domain when present.
  std::optional<ClockHealthOptions> clock_health;  ///< Closed health capability/requirement mode.
  TopicAttachPolicy attach{};  ///< Peer presence and optional finite attach wait policy.
  std::optional<LogicalTensorRequirements> tensor;  ///< Tensor-only logical boundary contract.
  EndpointBounds bounds{};  ///< External peer bounds, interpreted by import/export direction.
};

/// @brief Identify an intrinsic external-to-Graph typed boundary.
/// @tparam T Exact public payload type entering the Graph.
template <PublishedPayload T>
class ImportedTopic {
 public:
  /// @brief Return the Graph-side output used by typed `add_flow`.
  /// @return Dynamically resolved boundary declaration owned by the source Graph.
  /// @throws RuntimeError if the Graph was destroyed or failed composition, or
  /// the declaration was otherwise retired. The handle never returns a pointer
  /// to discarded draft storage.
  [[nodiscard]] Output<T>& output() const;

  /// @brief Return whether the source Graph still owns this boundary.
  /// @return \c true only while the exact Graph declaration can be re-resolved.
  [[nodiscard]] bool graph_valid() const noexcept;

  /// @brief Return whether this handle names a live Graph declaration.
  /// @return The same declaration-liveness result as \ref graph_valid.
  [[nodiscard]] explicit operator bool() const noexcept { return graph_valid(); }

 private:
  ImportedTopic(std::weak_ptr<detail::GraphState> state, std::uint64_t graph_port) noexcept
      : state_(std::move(state)), graph_port_(graph_port) {}

  std::weak_ptr<detail::GraphState> state_;
  std::uint64_t graph_port_{};

  friend class Graph;
};

/// @brief Identify an intrinsic Graph-to-external typed boundary.
/// @tparam T Exact public payload type leaving the Graph.
template <PublishedPayload T>
class ExportedTopic {
 public:
  /// @brief Return the Graph-side input used by typed `add_flow`.
  /// @return Dynamically resolved boundary declaration owned by the source Graph.
  /// @throws RuntimeError if the Graph was destroyed or failed composition, or
  /// the declaration was otherwise retired. The handle never returns a pointer
  /// to discarded draft storage.
  [[nodiscard]] Input<T>& input() const;

  /// @brief Return whether the source Graph still owns this boundary.
  /// @return \c true only while the exact Graph declaration can be re-resolved.
  [[nodiscard]] bool graph_valid() const noexcept;

  /// @brief Return whether this handle names a live Graph declaration.
  /// @return The same declaration-liveness result as \ref graph_valid.
  [[nodiscard]] explicit operator bool() const noexcept { return graph_valid(); }

 private:
  ExportedTopic(std::weak_ptr<detail::GraphState> state, std::uint64_t graph_port) noexcept
      : state_(std::move(state)), graph_port_(graph_port) {}

  std::weak_ptr<detail::GraphState> state_;
  std::uint64_t graph_port_{};

  friend class Graph;
};

}  // namespace holoscan
