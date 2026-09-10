/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/channel/clock_health.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/payload_identity.hpp>
#include <holoscan/time/clock_registry.hpp>

namespace holoscan {

class ChannelEndpointCompatibilityReport;

namespace detail {
struct ChannelEndpointCatalogState;
struct ChannelEndpointCompatibilityReportState;
struct ChannelEndpointRecord;
struct PodCatalogIdentityRecord;
class ChannelEndpointCatalogAccess;
class ChannelEndpointCompatibilityReportAccess;
}  // namespace detail

/// @brief Identify whether one external topic enters or leaves the compiled Graph.
enum class TopicDirection : std::uint8_t {
  kImport = 0,  ///< Receive samples from a non-Operator client into the Graph.
  kExport = 1,  ///< Publish Graph samples to a non-Operator client.
};

/// @brief Identify how an immutable catalog endpoint received its final name.
enum class EndpointNameOrigin : std::uint8_t {
  kExplicit = 0,   ///< Copy an explicit authoring-time endpoint name.
  kGenerated = 1,  ///< Derive the name through the recorded versioned Core algorithm.
};

/// @brief View versioned derivation evidence for one generated endpoint name.
struct GeneratedEndpointNameInfo {
  std::string_view algorithm;  ///< Catalog-owned stable algorithm identifier.
  std::uint32_t version{};     ///< Observable algorithm version; zero is invalid.
};

/// @brief Carry the compact first-16-byte discriminator of a complete POD identity.
struct TypeKey {
  std::array<std::uint8_t, 16> bytes{};  ///< Digest-order bytes with no numeric interpretation.
  friend constexpr bool operator==(const TypeKey&, const TypeKey&) = default;
};

/// @brief Hash a \ref TypeKey for process-local associative containers.
struct HOLOSCAN_CORE_EXPORT TypeKeyHash {
  /// @brief Hash one compact type discriminator for process-local lookup.
  /// @param key Bytewise type discriminator.
  /// @return Process-local hash value with no portable identity meaning.
  [[nodiscard]] std::size_t operator()(const TypeKey& key) const noexcept;
};

/// @brief Inspect one catalog-owned raw-POD field layout record.
class HOLOSCAN_CORE_EXPORT PodFieldLayoutView {
 public:
  /// @brief Return the exact declared POD field name.
  /// @return Catalog-owned UTF-8 field name.
  [[nodiscard]] std::string_view name() const noexcept;
  /// @brief Return the field's byte offset in the complete POD object.
  /// @return Zero-based object byte offset.
  [[nodiscard]] std::uint32_t byte_offset() const noexcept;
  /// @brief Return the scalar element count represented by this field.
  /// @return Positive scalar or fixed-array element count.
  [[nodiscard]] std::uint32_t element_count() const noexcept;
  /// @brief Return the field's closed portable scalar representation.
  /// @return Exact fixed-width scalar kind.
  [[nodiscard]] PodScalarKind scalar_kind() const noexcept;

 private:
  explicit PodFieldLayoutView(const PodFieldIdentityRecord* record) noexcept : record_(record) {}
  const PodFieldIdentityRecord* record_{};
  friend struct detail::PodCatalogIdentityRecord;
};

/// @brief Inspect one catalog-owned compatibility-evidence reference.
class HOLOSCAN_CORE_EXPORT CapabilityEvidenceView {
 public:
  /// @brief Return the stable compatibility-evidence identity.
  /// @return Catalog-owned evidence identity.
  [[nodiscard]] std::string_view id() const noexcept { return id_; }
  /// @brief Return the closed compatibility-evidence kind.
  /// @return Catalog-owned evidence-kind token.
  [[nodiscard]] std::string_view kind() const noexcept { return kind_; }
  /// @brief Return the evidence schema version.
  /// @return Nonzero version for a populated view.
  [[nodiscard]] std::uint32_t version() const noexcept { return version_; }

 private:
  std::string_view id_;
  std::string_view kind_;
  std::uint32_t version_{};
};

/// @brief Inspect the complete immutable identity and layout of one raw-POD payload.
class HOLOSCAN_CORE_EXPORT PodTypeIdentityView {
 public:
  /// @brief Return the complete POD identity's canonical semantic name.
  /// @return Catalog-owned canonical name.
  [[nodiscard]] std::string_view name() const noexcept;
  /// @brief Return the exact POD semantic version.
  /// @return Nonzero semantic version.
  [[nodiscard]] std::uint32_t version() const noexcept;
  /// @brief Return the complete canonical POD identity digest bytes.
  /// @return Catalog-owned SHA-256 bytes in digest order.
  [[nodiscard]] const std::array<std::uint8_t, 32>& digest_bytes() const noexcept;
  /// @brief Return the compact first-16-byte POD discriminator.
  /// @return Bytewise compact key derived from the complete identity digest.
  [[nodiscard]] TypeKey key() const noexcept;
  /// @brief Return the validated POD object size.
  /// @return Positive object size in bytes.
  [[nodiscard]] std::uint32_t object_size() const noexcept;
  /// @brief Return the validated POD object alignment.
  /// @return Positive power-of-two alignment in bytes.
  [[nodiscard]] std::uint32_t object_alignment() const noexcept;
  /// @brief Return the POD layout-preimage format version.
  /// @return Version one for baseline catalogs.
  [[nodiscard]] std::uint32_t layout_format_version() const noexcept;
  /// @brief Return the complete canonical POD layout digest bytes.
  /// @return Catalog-owned SHA-256 bytes in digest order.
  [[nodiscard]] const std::array<std::uint8_t, 32>& layout_digest_bytes() const noexcept;
  /// @brief Return ordered POD field layout records.
  /// @return Catalog-owned immutable field views covering the complete object.
  [[nodiscard]] std::span<const PodFieldLayoutView> fields() const noexcept;
  /// @brief Return the portable integer wire-endianness token.
  /// @return Catalog-owned \c little token in baseline.
  [[nodiscard]] std::string_view integer_wire_endianness() const noexcept;
  /// @brief Return the portable floating-point wire-format token.
  /// @return Catalog-owned \c ieee754 token in baseline.
  [[nodiscard]] std::string_view floating_wire_format() const noexcept;
  /// @brief Return build or toolchain evidence required for POD compatibility.
  /// @return Catalog-owned immutable evidence views, possibly empty.
  [[nodiscard]] std::span<const CapabilityEvidenceView> compatibility_evidence() const noexcept;

 private:
  explicit PodTypeIdentityView(const detail::PodCatalogIdentityRecord* record) noexcept
      : record_(record) {}
  const detail::PodCatalogIdentityRecord* record_{};
  friend class detail::ChannelEndpointCatalogAccess;
};

/// @brief Inspect one catalog-owned complete generated-schema identity.
class HOLOSCAN_CORE_EXPORT SchemaIdentityView {
 public:
  /// @brief Return the complete schema identity's canonical name.
  /// @return Catalog-owned canonical name.
  [[nodiscard]] std::string_view canonical_name() const noexcept;
  /// @brief Return the exact schema semantic version.
  /// @return Nonzero semantic version.
  [[nodiscard]] std::uint32_t version() const noexcept;
  /// @brief Return the complete canonical schema digest.
  /// @return Catalog-owned 32-byte SHA-256 digest view.
  [[nodiscard]] std::span<const std::byte, 32> schema_sha256() const noexcept;
  /// @brief Return the compact schema reference derived from complete identity.
  /// @return Derived schema identifier; never sufficient as identity alone.
  [[nodiscard]] SchemaId id() const noexcept;

 private:
  explicit SchemaIdentityView(const SchemaIdentityRecord* record) noexcept : record_(record) {}
  const SchemaIdentityRecord* record_{};
  friend class detail::ChannelEndpointCatalogAccess;
};

/// @brief Inspect one catalog-owned required capture-time clock declaration.
class HOLOSCAN_CORE_EXPORT RequireCaptureTimeView {
 public:
  /// @brief Return the required capture-time clock key.
  /// @return Catalog-owned exact clock key.
  [[nodiscard]] const ClockKey& clock() const noexcept;

 private:
  explicit RequireCaptureTimeView(const ClockKey* clock) noexcept : clock_(clock) {}
  const ClockKey* clock_{};
  friend class detail::ChannelEndpointCatalogAccess;
};

/// @brief Inspect one immutable accepted endpoint clock-health contract.
class HOLOSCAN_CORE_EXPORT ClockHealthContractView {
 public:
  /// @brief Return the endpoint's declared clock-health reporting capability.
  /// @return Closed immutable capability.
  [[nodiscard]] ClockHealthCapability capability() const noexcept;
  /// @brief Return the endpoint's accepted clock-health requirement.
  /// @return Closed immutable requirement.
  [[nodiscard]] ClockHealthRequirement requirement() const noexcept;
  /// @brief Return the maximum accepted report age when monitored health is required.
  /// @return Positive finite age for monitored mode, otherwise no value.
  [[nodiscard]] std::optional<std::chrono::nanoseconds> max_report_age() const noexcept;

 private:
  explicit ClockHealthContractView(const ClockHealthOptions* options) noexcept
      : options_(options) {}
  const ClockHealthOptions* options_{};
  friend class detail::ChannelEndpointCatalogAccess;
};

/// @brief Inspect one catalog-owned logical Tensor boundary contract.
///
/// The view contains only immutable logical requirements. Physical pools,
/// allocator authority, CUDA objects, readiness tokens, and live claims remain
/// private execution-plan or Runtime state.
class HOLOSCAN_CORE_EXPORT TensorBoundaryContractView {
 public:
  /// @brief Return the required logical storage kind when constrained.
  /// @return Exact storage kind, or no value when the contract is storage-agnostic.
  [[nodiscard]] std::optional<MemoryKind> memory_kind() const noexcept;
  /// @brief Return the required DLPack element type when constrained.
  /// @return Exact element type, or no value when unconstrained.
  [[nodiscard]] std::optional<DLDataType> dtype() const noexcept;
  /// @brief Return the exact logical rank when constrained.
  /// @return Exact rank, including zero for a scalar Tensor, or no value.
  [[nodiscard]] std::optional<std::uint32_t> rank() const noexcept;
  /// @brief Return the admitted maximum logical rank.
  /// @return Positive finite bound, or no value when this optional logical contract omits it.
  [[nodiscard]] std::optional<std::uint32_t> max_rank() const noexcept;
  /// @brief Return the admitted maximum logical byte span.
  /// @return Positive finite byte bound, or no value when omitted.
  [[nodiscard]] std::optional<std::uint64_t> max_byte_span() const noexcept;
  /// @brief Return ordered immutable per-axis bounds.
  /// @return Catalog-owned axis records.
  [[nodiscard]] std::span<const TensorAxisBounds> axes() const noexcept;
  /// @brief Return the admitted logical stride family.
  /// @return Closed stride policy.
  [[nodiscard]] TensorStridePolicy stride_policy() const noexcept;
  /// @brief Return the foreign-emission handling policy.
  /// @return Closed boundary emission policy.
  [[nodiscard]] InterfaceTensorEmitPolicy foreign_emit_policy() const noexcept;
  /// @brief Return the positive inline-host cap when the inline policy is selected.
  /// @return Byte cap, or no value for non-inline policies.
  [[nodiscard]] std::optional<std::uint32_t> max_inline_bytes() const noexcept;
  /// @brief Return the positive foreign-storage staging cap when copying is admitted.
  /// @return Byte cap, or no value when copying is forbidden.
  [[nodiscard]] std::optional<std::uint64_t> max_staging_byte_span() const noexcept;

 private:
  explicit TensorBoundaryContractView(const TensorBoundaryContractRecord* record) noexcept
      : record_(record) {}
  const TensorBoundaryContractRecord* record_{};
  friend class detail::ChannelEndpointCatalogAccess;
};

/// @brief Own an immutable canonical catalog of external Graph endpoints.
///
/// Copies share immutable state. Views and spans remain valid while any catalog
/// sharing that state remains alive. The catalog contains no live attachment or
/// execution authority and is safe for concurrent reads.
class HOLOSCAN_CORE_EXPORT ChannelEndpointCatalog {
 public:
  /// @brief Inspect one catalog-owned endpoint without copying its strings.
  class EndpointView;

  /// @brief Build or edit one immutable catalog through the advanced header.
  class Builder;

  /// @brief Create an empty immutable catalog.
  ChannelEndpointCatalog() noexcept = default;

  /// @brief Parse and normalize one strict schema-v2 JSON catalog.
  /// @param source Exact source bytes; duplicate keys and invalid UTF-8 reject.
  /// @return Immutable normalized catalog with nonsemantic raw-source provenance.
  /// @retval ErrorCode::kResourceExhausted The source exceeds a fixed schema-v2 parser or
  /// normalized-collection safety limit. Version one accepts at most 16 MiB of source, nesting
  /// depth 64, 100000 JSON nodes, 4096 members per object, and 65536 elements per array. Normalized
  /// collections accept at most 4096 endpoints/services, 1024 clocks, 4096 clock conversions, 1024
  /// POD fields per identity, and 32 Tensor axes per contract.
  [[nodiscard]] static expected<ChannelEndpointCatalog, Error> parse_json(std::string_view source);

  /// @brief Load and normalize one strict schema-v2 JSON catalog.
  /// @param path File whose exact bytes are read before parsing.
  /// @return Immutable normalized catalog with nonsemantic raw-source provenance.
  /// @retval ErrorCode::kResourceExhausted The file exceeds the 16 MiB schema-v2 source limit or
  /// its decoded document exceeds another fixed schema-v2 safety limit. Loading reads at most the
  /// source limit plus one byte before rejecting an oversized file.
  [[nodiscard]] static expected<ChannelEndpointCatalog, Error> load_json(
      const std::filesystem::path& path);

  /// @brief Save canonical normalized JSON without mutating this value.
  /// @param path Destination replaced only when canonical bytes differ.
  /// @return Success or a typed filesystem error.
  /// @note Existing and staged files are compared with fixed-size reads capped
  /// at the canonical catalog size plus one byte. Oversized existing content
  /// is replaced; an oversized staged file fails verification and is removed.
  [[nodiscard]] expected<void, Error> save_json(const std::filesystem::path& path) const;

  /// @brief Return canonical normalized catalog JSON.
  /// @return Owned canonical bytes; an empty catalog returns an empty string.
  [[nodiscard]] std::string json() const;

  /// @brief Return the normalized catalog digest.
  /// @return Catalog-owned `sha256:` spelling, or an empty view for an empty catalog.
  [[nodiscard]] std::string_view digest() const noexcept;

  /// @brief Return the SHA-256 of exact parsed source bytes when available.
  /// @return Catalog-owned `sha256:` spelling, or no value for program-built catalogs.
  [[nodiscard]] std::optional<std::string_view> raw_source_sha256() const noexcept;

  /// @brief Return the exact parsed source byte count when available.
  /// @return Raw byte count, or no value for program-built catalogs.
  [[nodiscard]] std::optional<std::uint64_t> raw_source_size_bytes() const noexcept;

  /// @brief Return the exact deployment scope carried by this catalog.
  /// @return Catalog-owned scope, or an empty view for an empty catalog.
  [[nodiscard]] std::string_view scope() const noexcept;

  /// @brief Find one endpoint by its exact stable name.
  /// @param endpoint_name Exact case-sensitive endpoint identity.
  /// @return Immutable catalog-owned view, or \c ErrorCode::kNotFound.
  /// @note Failure returns a code-only error so lookup remains nonthrowing.
  [[nodiscard]] expected<EndpointView, Error> endpoint(
      std::string_view endpoint_name) const noexcept;

  /// @brief Compare this required catalog with one candidate catalog.
  /// @param candidate Candidate/provided immutable catalog.
  /// @return Deterministically ordered compatibility diagnostics.
  [[nodiscard]] ChannelEndpointCompatibilityReport compatibility_with(
      const ChannelEndpointCatalog& candidate) const;

  /// @brief Create an advanced builder initialized from this catalog.
  /// @return Independent builder; this immutable catalog is unchanged.
  [[nodiscard]] Builder edit() const;

  /// @brief Create an empty advanced builder for one deployment scope.
  /// @param deployment_scope Nonempty scope copied by the builder.
  /// @return Independent builder defined by `endpoint_catalog_builder.hpp`.
  [[nodiscard]] static Builder builder(std::string_view deployment_scope);

 private:
  explicit ChannelEndpointCatalog(
      std::shared_ptr<const detail::ChannelEndpointCatalogState> state) noexcept
      : state_(std::move(state)) {}

  std::shared_ptr<const detail::ChannelEndpointCatalogState> state_;
  friend class detail::ChannelEndpointCatalogAccess;
};

/// @brief Own deterministic cold-path compatibility evidence for two catalogs.
class HOLOSCAN_CORE_EXPORT ChannelEndpointCompatibilityReport {
 public:
  /// @brief Create an empty compatible report.
  ChannelEndpointCompatibilityReport() noexcept = default;

  /// @brief Return whether the candidate satisfies every required record.
  /// @return \c true only when no error diagnostic is present.
  [[nodiscard]] bool compatible() const noexcept;

  /// @brief Return ordered compatibility diagnostics.
  /// @return Report-owned immutable diagnostics.
  [[nodiscard]] std::span<const Diagnostic> diagnostics() const noexcept;

  /// @brief Return deterministic schema-v2 report JSON.
  /// @return Owned canonical report bytes.
  [[nodiscard]] std::string json() const;

 private:
  explicit ChannelEndpointCompatibilityReport(
      std::shared_ptr<const detail::ChannelEndpointCompatibilityReportState> state) noexcept
      : state_(std::move(state)) {}

  std::shared_ptr<const detail::ChannelEndpointCompatibilityReportState> state_;
  friend class detail::ChannelEndpointCompatibilityReportAccess;
};

/// @brief Inspect one immutable endpoint owned by a \ref ChannelEndpointCatalog.
///
/// The view borrows catalog state and becomes invalid when the final catalog
/// sharing that state is destroyed. It exposes no attachment or mutation
/// authority and is safe for concurrent reads while its catalog remains alive.
class HOLOSCAN_CORE_EXPORT ChannelEndpointCatalog::EndpointView {
 public:
  /// @brief Create an empty endpoint view.
  EndpointView() noexcept = default;

  /// @brief Return the exact stable endpoint name.
  /// @return Catalog-owned name, or an empty view.
  [[nodiscard]] std::string_view name() const noexcept;

  /// @brief Return the deployment service URI.
  /// @return Catalog-owned URI, or an empty view.
  [[nodiscard]] std::string_view uri() const noexcept;

  /// @brief Return the stable service identity referenced by this endpoint.
  /// @return Catalog-owned service identity, or an empty view.
  [[nodiscard]] std::string_view service() const noexcept;

  /// @brief Return whether the endpoint imports or exports samples.
  /// @return Catalog direction; an empty view returns \c TopicDirection::kImport.
  [[nodiscard]] TopicDirection direction() const noexcept;

  /// @brief Return whether the final name was explicit or generated.
  /// @return Catalog name provenance; an empty view returns explicit.
  [[nodiscard]] EndpointNameOrigin name_origin() const noexcept;

  /// @brief Return generated-name derivation evidence when applicable.
  /// @return Catalog-owned record for a generated name, otherwise no value.
  [[nodiscard]] std::optional<GeneratedEndpointNameInfo> generated_name() const noexcept;

  /// @brief Return the complete raw-POD identity when this endpoint is POD-backed.
  /// @return Catalog-owned complete POD view, or no value for another payload family.
  [[nodiscard]] std::optional<PodTypeIdentityView> pod_type_identity() const noexcept;

  /// @brief Return the complete generated-schema identity when schema-backed.
  /// @return Catalog-owned complete schema view, or no value for another payload family.
  [[nodiscard]] std::optional<SchemaIdentityView> schema_identity() const noexcept;

  /// @brief Return the finite serialized payload bound for a generated-schema endpoint.
  /// @return Positive maximum serialized payload bytes, or no value for another payload family.
  [[nodiscard]] std::optional<std::uint64_t> schema_max_serialized_size() const noexcept;

  /// @brief Return the required capture-time clock when present.
  /// @return Catalog-owned capture-time requirement, or no value when timestamps are optional.
  [[nodiscard]] std::optional<RequireCaptureTimeView> capture_time() const noexcept;

  /// @brief Return the normalized immutable clock-health contract.
  /// @return Accepted ordinary or monitored clock-health contract.
  [[nodiscard]] ClockHealthContractView clock_health() const noexcept;

  /// @brief Return the resolved logical memory-conversion policy.
  /// @return Explicit accepted policy; an empty view returns \c kDefault.
  [[nodiscard]] MemoryConversionPolicy memory_conversion() const noexcept;

  /// @brief Return the resolved external-peer attachment contract.
  /// @return Peer requirement and positive finite catalog bound.
  [[nodiscard]] TopicAttachPolicy attach_policy() const noexcept;

  /// @brief Return the resolved direction-local endpoint bounds.
  /// @return Positive finite bounds for the applicable endpoint role.
  [[nodiscard]] EndpointBounds bounds() const noexcept;

  /// @brief Return the logical Tensor contract when this endpoint is Tensor-backed.
  /// @return Catalog-owned immutable requirements, or no value for another payload family.
  [[nodiscard]] std::optional<TensorBoundaryContractView> tensor_requirements() const noexcept;

  /// @brief Return whether this view names a catalog endpoint.
  /// @return \c true when the borrowed state and record index are valid.
  [[nodiscard]] explicit operator bool() const noexcept;

 private:
  EndpointView(const detail::ChannelEndpointCatalogState* state, std::size_t index) noexcept
      : state_(state), index_(index) {}

  const detail::ChannelEndpointCatalogState* state_{};
  std::size_t index_{};
  friend class ChannelEndpointCatalog;
  friend class detail::ChannelEndpointCatalogAccess;
};

}  // namespace holoscan
