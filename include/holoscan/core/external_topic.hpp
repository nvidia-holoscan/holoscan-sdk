/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/channel/clock_health.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/payload_options.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/schema_identity.hpp>
#include <holoscan/core/tensor_field.hpp>
#include <holoscan/time/clock_registry.hpp>

namespace holoscan {

class Graph;

namespace detail {
class GraphState;

/// Preserve one call-lifetime descriptor-Tensor topic field before Graph normalization.
struct AuthoredDescriptorTensorTopicField {
  std::uint16_t field_id{};
  std::uint16_t tensor_index{};
  std::optional<TensorBoundaryContract> contract;
  bool selector_valid{};
};
}  // namespace detail

/// @brief Declare Tensor field contracts for one descriptor-backed schema external topic.
///
/// The schema type fixes the selector domain at compile time. Each selected field must receive
/// exactly one finite boundary contract, and a complete inventory must be supplied when the
/// companion is passed to \ref Graph::import_topic or \ref Graph::export_topic. Borrowed shape and
/// stride views are copied during that Graph authoring call.
///
/// @tparam SchemaT Owning descriptor-backed generated schema type.
template <typename SchemaT>
  requires detail::HasDescriptorTensorSchemaNativeTraits<SchemaT>
class DescriptorTensorTopicContract {
 public:
  /// @brief Configure the contract for one selected descriptor Tensor field.
  class FieldBuilder {
   public:
    /// @brief Set the finite external-boundary contract for this field.
    /// @param contract Representation, bounds, transfer, and conversion requirements.
    /// @return The owning topic contract for continued authoring.
    DescriptorTensorTopicContract& contract(TensorBoundaryContract contract) {
      parent_->fields_[index_].contract = std::move(contract);
      return *parent_;
    }

   private:
    FieldBuilder(DescriptorTensorTopicContract* parent, std::size_t index) noexcept
        : parent_(parent), index_(index) {}

    DescriptorTensorTopicContract* parent_{};
    std::size_t index_{};

    friend class DescriptorTensorTopicContract;
  };

  /// @brief Select one Tensor member for external-boundary authoring.
  /// @param selector Generated schema-typed field token.
  /// @return Field-scoped builder used to provide its finite contract.
  [[nodiscard]] FieldBuilder field(TensorFieldSelector<SchemaT> selector) {
    fields_.push_back(detail::AuthoredDescriptorTensorTopicField{
        .field_id = selector.field_id,
        .tensor_index = selector.tensor_index,
        .contract = std::nullopt,
        .selector_valid = detail::descriptor_tensor_schema_field_selector_matches<SchemaT>(
            selector.field_id, selector.tensor_index),
    });
    return FieldBuilder{this, fields_.size() - 1U};
  }

  /// @brief Declare one selected Tensor member and its finite boundary contract.
  /// @param selector Generated schema-typed field token.
  /// @param contract Representation, bounds, transfer, and conversion requirements.
  /// @return This topic contract for continued authoring.
  DescriptorTensorTopicContract& field(TensorFieldSelector<SchemaT> selector,
                                       TensorBoundaryContract contract) {
    return field(selector).contract(std::move(contract));
  }

 private:
  std::vector<detail::AuthoredDescriptorTensorTopicField> fields_;

  friend class Graph;
};

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
  std::optional<TensorBoundaryContract> tensor;  ///< Tensor-only logical boundary contract.
  EndpointBounds bounds{};  ///< External peer bounds, interpreted by import/export direction.
};

/// @brief Identify an intrinsic external-to-Graph typed boundary.
/// @tparam T Exact public payload type entering the Graph.
template <PublishedPayload T>
class ImportedTopic {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

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
  ImportedTopic(std::weak_ptr<detail::GraphState> state, std::uint64_t graph_port,
                std::shared_ptr<const void> handle_identity) noexcept
      : state_(std::move(state)),
        graph_port_(graph_port),
        handle_identity_(std::move(handle_identity)) {}

  std::weak_ptr<detail::GraphState> state_;
  std::uint64_t graph_port_{};
  std::shared_ptr<const void> handle_identity_;

  friend class Graph;
};

/// @brief Identify an intrinsic Graph-to-external typed boundary.
/// @tparam T Exact public payload type leaving the Graph.
template <PublishedPayload T>
class ExportedTopic {
 public:
  /// @brief Opt this authoring handle out of replayable Graph captures.
  using graph_authoring_handle_tag = void;

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
  ExportedTopic(std::weak_ptr<detail::GraphState> state, std::uint64_t graph_port,
                std::shared_ptr<const void> handle_identity) noexcept
      : state_(std::move(state)),
        graph_port_(graph_port),
        handle_identity_(std::move(handle_identity)) {}

  std::weak_ptr<detail::GraphState> state_;
  std::uint64_t graph_port_{};
  std::shared_ptr<const void> handle_identity_;

  friend class Graph;
};

}  // namespace holoscan
