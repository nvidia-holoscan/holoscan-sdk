/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/core/external_topic_types.hpp>
#include <holoscan/core/graph_records.hpp>
#include <holoscan/core/port_types.hpp>

namespace holoscan::detail {

/// Maximum endpoint records admitted by endpoint catalog schema version one.
inline constexpr std::size_t kMaximumCatalogEndpoints = 4'096U;

/// @brief Own one complete portable clock declaration used by catalog endpoints.
struct CatalogClockRecord {
  std::uint64_t id{};               ///< Dense nonzero catalog-local identity.
  ClockKey key;                     ///< Stable clock name.
  std::string kind;                 ///< Closed clock-kind token.
  std::string unit;                 ///< Closed coordinate unit token.
  std::string epoch;                ///< Closed epoch token.
  std::string monotonicity;         ///< Closed monotonicity token.
  std::string discipline_kind;      ///< Closed deployment-discipline kind.
  std::string discipline_identity;  ///< Stable deployment-discipline identity.
};

/// @brief Own one exact directed affine conversion between catalog clocks.
struct CatalogClockConversionRecord {
  ClockKey from;                        ///< Source clock key.
  ClockKey to;                          ///< Destination clock key.
  std::int64_t scale_numerator{};       ///< Positive reduced numerator.
  std::uint64_t scale_denominator{1U};  ///< Positive reduced denominator.
  std::int64_t offset_ns{};             ///< Signed destination-domain offset.
  std::int64_t admitted_min{};          ///< Smallest admitted source coordinate.
  std::int64_t admitted_max{};          ///< Largest admitted source coordinate.
  std::string provenance;               ///< Stable conversion-evidence identity.
};

/// @brief Own versioned derivation evidence for one generated endpoint name.
struct GeneratedEndpointNameRecord {
  std::string algorithm;    ///< Stable generation algorithm identifier.
  std::uint32_t version{};  ///< Observable algorithm version; zero is invalid.
};

/// @brief Own normalized POD identity and its precomputed immutable views.
struct PodCatalogIdentityRecord {
  PodIdentityRecord identity;                          ///< Complete copied raw-POD identity.
  std::array<std::uint8_t, 32> digest_bytes{};         ///< Canonical type-identity SHA-256.
  TypeKey key{};                                       ///< First 16 identity digest bytes.
  std::array<std::uint8_t, 32> layout_digest_bytes{};  ///< Canonical layout SHA-256.
  std::vector<PodFieldLayoutView> field_views;         ///< Catalog-owned ordered field views.
  std::vector<CapabilityEvidenceView> compatibility_evidence;  ///< Build/toolchain evidence.

  void rebuild_field_views() {
    field_views.clear();
    field_views.reserve(identity.fields.size());
    for (const auto& field : identity.fields) {
      field_views.push_back(PodFieldLayoutView{&field});
    }
  }
};

/// @brief Own the immutable service-wide facts every peer must create or verify identically.
///
/// This record is deliberately distinct from endpoint-local publisher and
/// subscriber requests. Runtime adapters may consume it as one symmetric
/// create-or-verify value without allowing the first attached peer to redefine
/// service geometry from its local request.
struct ResolvedServiceStaticRequirements {
  std::string id;  ///< Stable service identity referenced by endpoint records.
  PayloadIdentityRecord payload_identity;  ///< Complete portable payload/wire identity.
  /// Positive service writer maximum for a serialized schema payload.
  std::uint64_t schema_writer_max_serialized_bytes{};
  /// Complete data-frame bytes required by the service writer maximum; the service's exact
  /// inline-payload capacity.
  std::uint64_t required_data_frame_bytes{};
  std::optional<PodCatalogIdentityRecord> pod_identity;  ///< Derived POD identity when applicable.
  std::optional<SchemaIdentityRecord>
      schema_identity;  ///< Complete schema identity when applicable.
  std::optional<TensorBoundaryContractRecord> tensor;  ///< Logical Tensor wire policy.
  /// Complete descriptor-schema field contracts in canonical field-id order.
  std::vector<DescriptorTensorTopicFieldRecord> tensor_fields;
  QueuePolicy queue_policy{QueuePolicy::kBlocking};  ///< Frozen service queue family.
  std::uint32_t max_subscriber_depth{1U};            ///< Positive service queue maximum.
  std::uint32_t max_subscriber_borrowed{1U};         ///< Positive service borrow maximum.
  std::uint32_t publisher_history_depth{};  ///< Frozen baseline history depth; exactly zero.
  bool event_wake{true};                    ///< Frozen readiness/event mode.
};

/// @brief Own one endpoint's publisher-local admitted request.
struct ResolvedPublisherEndpointRequest {
  std::optional<std::uint32_t>
      max_loaned_samples;  ///< Positive endpoint-local outstanding-loan bound when applicable.
  std::optional<ClockKey> capture_time_clock;  ///< Required capture clock when present.
  ClockHealthOptions clock_health{};           ///< Explicit ordinary or monitored health mode.
};

/// @brief Own one endpoint's subscriber-local admitted request.
struct ResolvedSubscriberEndpointRequest {
  std::uint32_t queue_depth{1U};           ///< Positive local requested queue depth.
  std::uint32_t max_borrowed_samples{1U};  ///< Positive local simultaneous-borrow bound.
};

/// @brief Own one normalized endpoint catalog record.
struct ChannelEndpointRecord {
  std::string name;  ///< Exact stable client lookup identity.
  std::string uri;   ///< Deployment service URI copied into the catalog.
  TopicDirection direction{TopicDirection::kImport};              ///< Boundary direction.
  EndpointNameOrigin name_origin{EndpointNameOrigin::kExplicit};  ///< Name provenance.
  std::optional<GeneratedEndpointNameRecord> generated_name;      ///< Generated-name evidence.
  PortId boundary_port{};  ///< Dense plan-local boundary port, never serialized.
  std::string service;     ///< Stable reference into the canonical service array.
  QueuePolicy queue_policy{QueuePolicy::kBlocking};  ///< Frozen external service queue family.
  std::uint32_t service_max_subscriber_depth{};      ///< Symmetric service queue maximum.
  std::uint32_t service_max_subscriber_borrowed{};   ///< Symmetric service borrow maximum.
  PayloadIdentityRecord payload_identity;            ///< Complete logical payload identity.
  /// Positive catalog service writer maximum for a serialized schema payload.
  std::uint64_t schema_writer_max_serialized_bytes{};
  std::optional<PodCatalogIdentityRecord> pod_identity;     ///< Derived complete POD catalog facts.
  std::optional<SchemaIdentityRecord> schema_identity;      ///< Effective complete schema identity.
  std::optional<MemoryConversionPolicy> memory_conversion;  ///< Normalized memory policy.
  std::optional<ClockKey> capture_time_clock;  ///< Required capture-time domain when present.
  ClockHealthOptions clock_health{};           ///< Explicit normalized ordinary or monitored mode.
  TopicAttachPolicy attach{};                  ///< Peer requirement and finite wait bound.
  std::optional<TensorBoundaryContractRecord> tensor;  ///< Bounded Tensor wire policy.
  /// Complete descriptor-schema field contracts in canonical field-id order.
  std::vector<DescriptorTensorTopicFieldRecord> tensor_fields;
  EndpointBounds bounds{};  ///< Direction-local finite endpoint bounds.

  // These three records are the normalized ownership seam. The fields above
  // remain the compiler/Builder input carrier; normalization derives the
  // records below once and every canonical JSON, compatibility, and public
  // inspection path consumes these mechanically separated values.
  ResolvedServiceStaticRequirements service_static;  ///< Symmetric service-wide requirements.
  std::optional<ResolvedPublisherEndpointRequest> publisher_request;    ///< Publisher-local facts.
  std::optional<ResolvedSubscriberEndpointRequest> subscriber_request;  ///< Subscriber-local facts.
};

/// @brief Deep-own one normalized immutable external endpoint catalog.
///
/// This channel-private state is the single ownership seam shared by the
/// compiler producer and standalone catalog parser/builder. It intentionally
/// contains no transport, queue, runtime adapter, authoring-object pointer, or
/// compiler-owned execution-plan state.
struct ChannelEndpointCatalogState {
  std::string scope;                             ///< Canonical deployment scope.
  std::string json;                              ///< Canonical normalized JSON bytes.
  std::array<std::byte, 32> digest_bytes{};      ///< Canonical digest bytes for internal joins.
  std::string digest;                            ///< Catalog-owned `sha256:` digest spelling.
  std::optional<std::string> raw_source_sha256;  ///< Exact-source digest when parsed or loaded.
  std::optional<std::uint64_t> raw_source_size_bytes;  ///< Exact source byte count when parsed.
  std::vector<CatalogClockRecord> clocks;  ///< Canonically sorted portable clock records.
  std::vector<CatalogClockConversionRecord>
      clock_conversions;                         ///< Canonically sorted directed conversions.
  std::vector<ChannelEndpointRecord> endpoints;  ///< Canonically sorted endpoint records.
};

/// @brief Deep-own one deterministic catalog compatibility report.
struct ChannelEndpointCompatibilityReportState {
  bool compatible{true};                ///< True only when every required record is satisfied.
  std::vector<Diagnostic> diagnostics;  ///< Stable ordered cold-path diagnostics.
  std::string json;                     ///< Canonical report JSON.
};

/// Provide deterministic source-tree fault injection for catalog-save tests.
///
/// This private header is not installed. The default path is inert apart from
/// one relaxed atomic exchange in the cold save operation.
class EndpointCatalogTestAccess {
 public:
  /// Append one byte to the next staged catalog before verification.
  HOLOSCAN_CORE_EXPORT static void oversize_next_staged_save() noexcept;

  /// Clear any pending catalog-save fault injection.
  HOLOSCAN_CORE_EXPORT static void reset_save_faults() noexcept;
};

/// @brief Construct and inspect the private state behind the public catalog facade.
class ChannelEndpointCatalogAccess {
 public:
  /// @brief Publish one complete immutable catalog state.
  /// @param state Deep-owning normalized state, or null for an empty catalog.
  /// @return Public copyable immutable catalog facade.
  // The shared state is an ownership sink; by-value admission preserves move-only caller intent.
  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  [[nodiscard]] static ChannelEndpointCatalog make(
      std::shared_ptr<const ChannelEndpointCatalogState> state) noexcept {
    return ChannelEndpointCatalog{std::move(state)};
  }

  /// @brief Validate, canonicalize, and publish program-built endpoint records.
  /// @param scope Nonempty deployment scope copied into canonical JSON.
  /// @param endpoints Deep-owned endpoint records in arbitrary order.
  /// @return Canonically sorted immutable catalog or a typed validation error.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<ChannelEndpointCatalog, Error> normalize(
      std::string scope, std::vector<ChannelEndpointRecord> endpoints,
      std::vector<CatalogClockRecord> clocks = {},
      std::vector<CatalogClockConversionRecord> clock_conversions = {},
      bool infer_standalone_clocks = false);

  /// @brief Return the private immutable state behind a public catalog.
  /// @param catalog Catalog facade to inspect.
  /// @return State pointer, or null for an empty catalog.
  [[nodiscard]] static const ChannelEndpointCatalogState* state(
      const ChannelEndpointCatalog& catalog) noexcept {
    return catalog.state_.get();
  }

  /// @brief Return canonical digest bytes for compiler and artifact identity joins.
  /// @param catalog Catalog facade to inspect.
  /// @return State-owned bytes, or all-zero bytes for an empty catalog.
  [[nodiscard]] static std::array<std::byte, 32> digest_bytes(
      const ChannelEndpointCatalog& catalog) noexcept {
    const auto* catalog_state = state(catalog);
    return catalog_state == nullptr ? std::array<std::byte, 32>{} : catalog_state->digest_bytes;
  }

  /// @brief Return the immutable normalized record behind an endpoint view.
  /// @param endpoint Borrowed public endpoint view.
  /// @return Catalog-owned record, or null for an empty/invalid view.
  [[nodiscard]] static const ChannelEndpointRecord* record(
      ChannelEndpointCatalog::EndpointView endpoint) noexcept {
    return endpoint ? &endpoint.state_->endpoints[endpoint.index_] : nullptr;
  }

  /// @brief Derive the backend-private process-local service binding.
  /// @param endpoint Borrowed public endpoint view.
  /// @return Opaque binding, or an empty string for an invalid view.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static std::string backend_service_binding(
      ChannelEndpointCatalog::EndpointView endpoint);

  /// @brief Resolve an endpoint capture clock to its dense catalog-local identifier.
  /// @param endpoint Borrowed public endpoint view.
  /// @return Nonzero id when capture time is required, otherwise the invalid id.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static ClockId capture_clock_id(
      ChannelEndpointCatalog::EndpointView endpoint) noexcept;

  /// @brief Return how many clocks the catalog owning this endpoint declares.
  ///
  /// Catalog clock identities are dense and one-based, so this count is the upper bound of the
  /// resolvable identity range. A transport boundary uses it to reject an identity no peer reading
  /// this catalog could resolve.
  ///
  /// @param endpoint Borrowed public endpoint view.
  /// @return Declared clock count, or zero for an empty or invalid view.
  [[nodiscard]] static std::uint64_t clock_count(
      ChannelEndpointCatalog::EndpointView endpoint) noexcept {
    return endpoint ? static_cast<std::uint64_t>(endpoint.state_->clocks.size()) : 0U;
  }

  [[nodiscard]] static PodTypeIdentityView pod_view(
      const PodCatalogIdentityRecord* record) noexcept {
    return PodTypeIdentityView{record};
  }

  [[nodiscard]] static SchemaIdentityView schema_view(const SchemaIdentityRecord* record) noexcept {
    return SchemaIdentityView{record};
  }

  [[nodiscard]] static RequireCaptureTimeView capture_time_view(const ClockKey* clock) noexcept {
    return RequireCaptureTimeView{clock};
  }

  [[nodiscard]] static ClockHealthContractView clock_health_view(
      const ClockHealthOptions* options) noexcept {
    return ClockHealthContractView{options};
  }

  [[nodiscard]] static TensorBoundaryContractView tensor_requirements_view(
      const TensorBoundaryContractRecord* record) noexcept {
    return TensorBoundaryContractView{record};
  }
};

/// @brief Construct the public immutable compatibility-report facade.
class ChannelEndpointCompatibilityReportAccess {
 public:
  /// @brief Publish one complete report state.
  /// @param state Deep-owning immutable state.
  /// @return Public copyable report facade.
  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  [[nodiscard]] static ChannelEndpointCompatibilityReport make(
      std::shared_ptr<const ChannelEndpointCompatibilityReportState> state) noexcept {
    return ChannelEndpointCompatibilityReport{std::move(state)};
  }
};

}  // namespace holoscan::detail
