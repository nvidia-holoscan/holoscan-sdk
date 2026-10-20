/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/external_topic.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/payload_identity.hpp>
#include <holoscan/core/sample.hpp>

namespace holoscan {

/// @brief Build or edit one immutable external endpoint catalog.
///
/// The builder owns all copied authoring state. It creates no transport
/// service, performs no attachment, and cannot mutate a catalog that has
/// already been built. Build-time validation is identical for graph-derived,
/// parsed, and independently authored catalogs.
///
/// @note Mutations validate record-local facts immediately. Cross-record
/// identity and canonical ordering checks run once when \ref build consumes the
/// completed draft.
class HOLOSCAN_CORE_EXPORT ChannelEndpointCatalog::Builder {
 public:
  /// @brief Destroy the builder and its uncommitted draft.
  ~Builder();

  /// @brief Move one independent catalog draft.
  /// @param other Builder whose draft is transferred.
  Builder(Builder&& other) noexcept;

  /// @brief Replace this draft by moving another builder.
  /// @param other Builder whose draft is transferred.
  /// @return This builder.
  Builder& operator=(Builder&& other) noexcept;

  Builder(const Builder&) = delete;
  Builder& operator=(const Builder&) = delete;

  /// @brief Add one typed endpoint that publishes into a Graph.
  /// @tparam T Exact public payload type.
  /// @param endpoint_name Explicit deployment-stable endpoint identity.
  /// @param uri Exact nonempty \c holomq:// service URI.
  /// @param options Complete logical endpoint options copied by this call.
  /// @return Success, or a typed validation/duplicate error.
  template <PublishedPayload T>
  [[nodiscard]] expected<void, Error> import_topic(std::string_view endpoint_name,
                                                   std::string_view uri,
                                                   const TopicOptions& options = {}) {
    return add_topic(TopicDirection::kImport,
                     endpoint_name,
                     uri,
                     options,
                     detail::operator_spec_payload_identity<T>(),
                     detail::serialized_payload_capacity_bytes<T>());
  }

  /// @brief Add one typed endpoint that subscribes to Graph output.
  /// @tparam T Exact public payload type.
  /// @param endpoint_name Explicit deployment-stable endpoint identity.
  /// @param uri Exact nonempty \c holomq:// service URI.
  /// @param options Complete logical endpoint options copied by this call.
  /// @return Success, or a typed validation/duplicate error.
  template <PublishedPayload T>
  [[nodiscard]] expected<void, Error> export_topic(std::string_view endpoint_name,
                                                   std::string_view uri,
                                                   const TopicOptions& options = {}) {
    return add_topic(TopicDirection::kExport,
                     endpoint_name,
                     uri,
                     options,
                     detail::operator_spec_payload_identity<T>(),
                     detail::serialized_payload_capacity_bytes<T>());
  }

  /// @brief Replace logical options for one existing endpoint.
  /// @param endpoint_name Exact endpoint identity in this draft.
  /// @param options Complete replacement options copied by this call.
  /// @return Success, or a typed lookup/validation error.
  ///
  /// The endpoint name, URI, direction, and payload type remain unchanged.
  /// Supplying \ref TopicOptions::endpoint_name is accepted only when it is
  /// byte-identical to \p endpoint_name.
  [[nodiscard]] expected<void, Error> set_topic_options(std::string_view endpoint_name,
                                                        const TopicOptions& options);

  /// @brief Validate, canonicalize, and freeze this draft.
  /// @return A new immutable catalog, or a typed validation error.
  ///
  /// This operation consumes the builder. No partially valid catalog is
  /// returned on failure.
  [[nodiscard]] expected<ChannelEndpointCatalog, Error> build() &&;

 private:
  struct State;

  explicit Builder(std::unique_ptr<State> state) noexcept;

  [[nodiscard]] expected<void, Error> add_topic(TopicDirection direction,
                                                std::string_view endpoint_name,
                                                std::string_view uri, const TopicOptions& options,
                                                PayloadIdentityRecord payload_identity,
                                                std::uint64_t serialized_payload_capacity_bytes);

  std::unique_ptr<State> state_;

  friend class ChannelEndpointCatalog;
};

}  // namespace holoscan
