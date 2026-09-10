/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/core/schema_identity.hpp"

namespace holoscan {

class Tensor;

namespace schema {
struct ImageT;
}  // namespace schema

/// @brief Identify the only schema-bearing wrapper/envelope ABI emitted by baseline.
inline constexpr std::uint32_t kSchemaEnvelopeABIVersion = 2U;
/// @brief Identify the descriptor-array envelope ABI emitted by descriptor-backed Tensor schemas.
inline constexpr std::uint32_t kDescriptorTensorSchemaEnvelopeABIVersion = 3U;
/// @brief Identify the exact portable byte width of the schema envelope prefix.
inline constexpr std::uint32_t kSchemaEnvelopeHeaderSize = 40U;

/// @brief Carry the fixed prefix of one schema-bearing Channel message.
///
/// This prefix contains the complete compact identity and endpoint generation
/// even after attachment has compared the full catalog identity. It is a Core
/// value and contains no HoloMQ type, pointer, parser address, or backend state.
struct SchemaEnvelopeHeader {
  std::uint32_t abi_version{kSchemaEnvelopeABIVersion};  ///< Exact wrapper/envelope ABI.
  std::uint32_t header_size{kSchemaEnvelopeHeaderSize};  ///< Exact versioned prefix size.
  SchemaId schema_id{};                 ///< Complete 128-bit compact schema identity.
  std::uint32_t schema_version{};       ///< Exact nonzero semantic schema version.
  std::uint32_t payload_size{};         ///< Serialized payload bytes after this prefix.
  std::uint64_t endpoint_generation{};  ///< Exact attached endpoint generation.
};

static_assert(std::is_standard_layout_v<SchemaEnvelopeHeader>);
static_assert(std::is_trivially_copyable_v<SchemaEnvelopeHeader>);
static_assert(sizeof(SchemaEnvelopeHeader) == kSchemaEnvelopeHeaderSize);

/// @brief Record one source in a generated package's ordered include closure.
struct SchemaPackageSource {
  std::string_view logical_path{};             ///< Canonical package-relative logical path.
  std::array<std::byte, 32> content_sha256{};  ///< SHA-256 of the exact source bytes.
};

/// @brief Record the FlatBuffers Runtime ABI accepted by a generated package.
struct FlatBufferRuntimeABI {
  std::uint32_t major{};     ///< Required FlatBuffers Runtime major version.
  std::uint32_t minor{};     ///< Required FlatBuffers Runtime minor version.
  std::uint32_t revision{};  ///< Required FlatBuffers Runtime revision.

  /// @brief Compare all exact Runtime ABI components.
  friend constexpr bool operator==(FlatBufferRuntimeABI, FlatBufferRuntimeABI) noexcept = default;
};

/// @brief Describe reproducible generation evidence separate from semantic identity.
///
/// These views describe local package usability; none participates in catalog identity or
/// \ref derive_schema_id. Public package factories deep-copy them. Code that passes evidence
/// directly to a low-level descriptor constructor must provide generated-package-static storage.
struct SchemaPackageEvidence {
  std::string_view flatc_build{};                   ///< Exact `flatc` version/build identity.
  std::span<const std::string_view> ordered_flags;  ///< Complete ordered normalized flag vector.
  std::string_view root_logical_path{};             ///< Canonical package-relative root path.
  std::span<const SchemaPackageSource> includes;    ///< Ordered transitive include closure.
  std::string_view include_root_mapping{};          ///< Deterministic logical include-root map.
  bool host_paths_stripped{};                       ///< Proof that no host path remains.
  std::uint32_t generator_format_version{};         ///< Nonzero generator evidence format.
  std::uint32_t package_format_version{};           ///< Nonzero descriptor format version.
  FlatBufferRuntimeABI runtime_abi{};               ///< Exact compatible Runtime ABI.
};

namespace detail {

/// Deep-owned backing for evidence accepted by public package factories.
class SchemaPackageEvidenceStorage {
 public:
  explicit SchemaPackageEvidenceStorage(const SchemaPackageEvidence& source)
      : flatc_build_(source.flatc_build),
        root_logical_path_(source.root_logical_path),
        include_root_mapping_(source.include_root_mapping),
        host_paths_stripped_(source.host_paths_stripped),
        generator_format_version_(source.generator_format_version),
        package_format_version_(source.package_format_version),
        runtime_abi_(source.runtime_abi) {
    ordered_flag_storage_.reserve(source.ordered_flags.size());
    for (const std::string_view flag : source.ordered_flags) {
      ordered_flag_storage_.emplace_back(flag);
    }
    ordered_flags_.reserve(ordered_flag_storage_.size());
    for (const std::string& flag : ordered_flag_storage_) {
      ordered_flags_.emplace_back(flag);
    }

    include_path_storage_.reserve(source.includes.size());
    include_digest_storage_.reserve(source.includes.size());
    for (const SchemaPackageSource& include : source.includes) {
      include_path_storage_.emplace_back(include.logical_path);
      include_digest_storage_.push_back(include.content_sha256);
    }
    includes_.reserve(include_path_storage_.size());
    for (std::size_t index = 0U; index < include_path_storage_.size(); ++index) {
      includes_.push_back(SchemaPackageSource{
          .logical_path = include_path_storage_[index],
          .content_sha256 = include_digest_storage_[index],
      });
    }

    evidence_ = SchemaPackageEvidence{
        .flatc_build = flatc_build_,
        .ordered_flags = ordered_flags_,
        .root_logical_path = root_logical_path_,
        .includes = includes_,
        .include_root_mapping = include_root_mapping_,
        .host_paths_stripped = host_paths_stripped_,
        .generator_format_version = generator_format_version_,
        .package_format_version = package_format_version_,
        .runtime_abi = runtime_abi_,
    };
  }

  [[nodiscard]] const SchemaPackageEvidence& evidence() const noexcept { return evidence_; }

 private:
  std::string flatc_build_;
  std::vector<std::string> ordered_flag_storage_;
  std::vector<std::string_view> ordered_flags_;
  std::string root_logical_path_;
  std::vector<std::string> include_path_storage_;
  std::vector<std::array<std::byte, 32>> include_digest_storage_;
  std::vector<SchemaPackageSource> includes_;
  std::string include_root_mapping_;
  bool host_paths_stripped_{};
  std::uint32_t generator_format_version_{};
  std::uint32_t package_format_version_{};
  FlatBufferRuntimeABI runtime_abi_{};
  SchemaPackageEvidence evidence_{};
};

}  // namespace detail

/// @brief Preserve deep-owned generated-package provenance in an execution plan.
///
/// Records are keyed by \ref schema_id and contain only cold, pointer-free evidence. They are
/// serialized in the diagnostic plan artifact but deliberately excluded from Graph and execution-
/// plan semantic digests, schema identity, and wire messages.
struct SchemaPackagePlanRecord {
  SchemaId schema_id{};                    ///< Package identity used to join payload records.
  std::uint64_t binary_schema_size{};      ///< Exact pinned `.bfbs` byte size.
  std::string flatc_build;                 ///< Exact `flatc` version/build identity.
  std::vector<std::string> ordered_flags;  ///< Complete ordered normalized flag vector.
  std::array<std::byte, 32> include_closure_sha256{};  ///< Ordered include-closure digest.
  std::uint32_t generator_format_version{};            ///< Nonzero generator evidence format.
  std::uint32_t package_format_version{};              ///< Nonzero descriptor format version.
  FlatBufferRuntimeABI runtime_abi{};  ///< Exact compatible FlatBuffers Runtime ABI.

  friend bool operator==(const SchemaPackagePlanRecord&, const SchemaPackagePlanRecord&) = default;
};

/// @brief Digest one ordered generated-package include closure.
/// @param includes Ordered transitive include sources, excluding the root schema.
/// @return Domain-separated SHA-256 over source count, paths, and exact source digests.
///
/// The preimage is `"holoscan.schema-include-closure/v1\\0"`, the source count as unsigned
/// 32-bit big-endian, then for each source its logical-path byte count as unsigned 32-bit big-
/// endian, its unmodified path bytes, and its 32 raw content-digest bytes. Valid package evidence
/// bounds both counts to the unsigned 32-bit domain before this value is published.
[[nodiscard]] constexpr std::array<std::byte, 32> schema_include_closure_sha256(
    std::span<const SchemaPackageSource> includes) noexcept {
  detail::SchemaSha256Builder hash;
  hash.add("holoscan.schema-include-closure/v1");
  hash.add_byte(0U);
  hash.add_u32_be(static_cast<std::uint32_t>(includes.size()));
  for (const SchemaPackageSource& source : includes) {
    hash.add_u32_be(static_cast<std::uint32_t>(source.logical_path.size()));
    hash.add(source.logical_path);
    hash.add(source.content_sha256);
  }
  return hash.finish();
}

/// @brief Describe one exact generated schema package for semantic type `T`.
/// @tparam T Semantic generated FlatBuffers root/native type.
///
/// The descriptor borrows package-owned static binary-schema bytes and functions. Public factories
/// additionally retain shared ownership of deep-copied generator evidence. It does not own a
/// transport endpoint. Callable addresses never enter semantic identity, persisted artifacts, or
/// wire messages; deep-copied pointer-free provenance may enter the diagnostic plan artifact
/// without entering its digest.
template <typename T>
class SchemaPackageDescriptor {
 public:
  /// @brief Verify serialized bytes using the exact generated root verifier.
  using VerifyFunction = bool (*)(std::span<const std::byte>) noexcept;
  /// @brief Parse already verified bytes into a borrowed generated root view.
  using ParseFunction = const T* (*)(std::span<const std::byte>) noexcept;
  /// @brief Serialize one owning semantic value into caller-owned bounded storage.
  using EncodeFunction = expected<std::size_t, Error> (*)(const T&, std::span<std::byte>) noexcept;
  /// @brief Materialize one owning semantic value from already verified bytes.
  using MaterializeFunction = expected<T, Error> (*)(std::span<const std::byte>) noexcept;

  /// @brief Create a descriptor over generated package-static evidence.
  /// @param identity Complete semantic identity projected by `schema_identity<T>()`.
  /// @param binary_schema Exact pinned canonical `.bfbs` bytes.
  /// @param evidence Reproducible generator and Runtime compatibility evidence.
  /// @param verify Exact generated payload verifier.
  /// @param parse Parser for bytes already accepted by `verify`.
  constexpr SchemaPackageDescriptor(SchemaIdentity identity,
                                    std::span<const std::byte> binary_schema,
                                    SchemaPackageEvidence evidence, VerifyFunction verify,
                                    ParseFunction parse) noexcept
      : identity_(identity),
        binary_schema_(binary_schema),
        evidence_(evidence),
        verify_(verify),
        parse_(parse) {}

  /// @brief Create a descriptor with an owning-value codec and finite serialized bound.
  /// @param identity Complete semantic identity projected by `schema_identity<T>()`.
  /// @param binary_schema Exact pinned canonical `.bfbs` bytes.
  /// @param evidence Reproducible generator and Runtime compatibility evidence.
  /// @param verify Exact generated payload verifier.
  /// @param max_serialized_size Positive finite maximum serialized payload size.
  /// @param encode Bounded owning-value serializer.
  /// @param materialize Owning-value decoder for bytes already accepted by \p verify.
  /// @note The callback addresses are process-local package evidence and never enter schema
  /// identity, persisted artifacts, or wire messages. Only deep-copied pointer-free provenance
  /// may enter the diagnostic plan artifact.
  constexpr SchemaPackageDescriptor(SchemaIdentity identity,
                                    std::span<const std::byte> binary_schema,
                                    SchemaPackageEvidence evidence, VerifyFunction verify,
                                    std::size_t max_serialized_size, EncodeFunction encode,
                                    MaterializeFunction materialize) noexcept
      : identity_(identity),
        binary_schema_(binary_schema),
        evidence_(evidence),
        verify_(verify),
        max_serialized_size_(max_serialized_size),
        encode_(encode),
        materialize_(materialize) {}

  /// @brief Create a descriptor that deep-owns all supplied generator evidence.
  /// @param identity Complete semantic identity projected by `schema_identity<T>()`.
  /// @param binary_schema Exact pinned canonical `.bfbs` bytes with package-static lifetime.
  /// @param evidence Generator evidence whose borrowed inputs are copied before return.
  /// @param verify Exact generated payload verifier.
  /// @param parse Parser for bytes already accepted by @p verify.
  /// @return Descriptor retaining immutable shared ownership of copied evidence.
  [[nodiscard]] static SchemaPackageDescriptor with_owned_evidence(
      SchemaIdentity identity, std::span<const std::byte> binary_schema,
      const SchemaPackageEvidence& evidence, VerifyFunction verify, ParseFunction parse) {
    auto owner = std::make_shared<const detail::SchemaPackageEvidenceStorage>(evidence);
    return SchemaPackageDescriptor{
        identity, binary_schema, std::move(owner), verify, parse, 0U, nullptr, nullptr};
  }

  /// @brief Create an owning-codec descriptor that deep-owns generator evidence.
  /// @param identity Complete semantic identity projected by `schema_identity<T>()`.
  /// @param binary_schema Exact pinned canonical `.bfbs` bytes with package-static lifetime.
  /// @param evidence Generator evidence whose borrowed inputs are copied before return.
  /// @param verify Exact generated payload verifier.
  /// @param max_serialized_size Positive finite maximum serialized payload size.
  /// @param encode Bounded owning-value serializer.
  /// @param materialize Owning-value decoder for bytes already accepted by @p verify.
  /// @return Descriptor retaining immutable shared ownership of copied evidence.
  [[nodiscard]] static SchemaPackageDescriptor with_owned_evidence(
      SchemaIdentity identity, std::span<const std::byte> binary_schema,
      const SchemaPackageEvidence& evidence, VerifyFunction verify, std::size_t max_serialized_size,
      EncodeFunction encode, MaterializeFunction materialize) {
    auto owner = std::make_shared<const detail::SchemaPackageEvidenceStorage>(evidence);
    return SchemaPackageDescriptor{identity,
                                   binary_schema,
                                   std::move(owner),
                                   verify,
                                   nullptr,
                                   max_serialized_size,
                                   encode,
                                   materialize};
  }

  /// @brief Return the complete semantic identity supplied by this package.
  [[nodiscard]] constexpr const SchemaIdentity& identity() const noexcept { return identity_; }
  /// @brief Return the exact package-pinned canonical `.bfbs` bytes.
  [[nodiscard]] constexpr std::span<const std::byte> binary_schema() const noexcept {
    return binary_schema_;
  }
  /// @brief Return generator and Runtime compatibility evidence.
  [[nodiscard]] constexpr const SchemaPackageEvidence& evidence() const noexcept {
    return evidence_;
  }
  /// @brief Return whether the exact generated verifier is present.
  [[nodiscard]] constexpr bool has_verifier() const noexcept { return verify_ != nullptr; }
  /// @brief Return whether the generated parser is present.
  [[nodiscard]] constexpr bool has_parser() const noexcept { return parse_ != nullptr; }
  /// @brief Return the finite serialized-payload capacity for owning values.
  /// @return Positive byte capacity for an owning codec, or zero for a borrowed-root descriptor.
  [[nodiscard]] constexpr std::size_t max_serialized_size() const noexcept {
    return max_serialized_size_;
  }
  /// @brief Return whether this package can serialize owning semantic values.
  [[nodiscard]] constexpr bool has_encoder() const noexcept { return encode_ != nullptr; }
  /// @brief Return whether this package can materialize owning semantic values.
  [[nodiscard]] constexpr bool has_materializer() const noexcept { return materialize_ != nullptr; }

  /// @brief Invoke the exact generated verifier for serialized payload bytes.
  /// @param bytes Serialized payload bytes.
  /// @return `true` only when the generated root verifier accepts the complete buffer.
  [[nodiscard]] bool verify(std::span<const std::byte> bytes) const noexcept {
    return verify_ != nullptr && verify_(bytes);
  }

  /// @brief Parse bytes that the caller already verified with this descriptor.
  /// @param bytes Serialized payload bytes previously accepted by \ref verify.
  /// @return Borrowed generated root view, or null when no parser is installed.
  /// @warning The returned view is invalidated when `bytes` is destroyed or modified.
  [[nodiscard]] const T* parse_verified(std::span<const std::byte> bytes) const noexcept {
    return parse_ != nullptr ? parse_(bytes) : nullptr;
  }

  /// @brief Serialize an owning semantic value into bounded caller-owned storage.
  /// @param value Owning generated value to serialize.
  /// @param destination Preallocated destination whose size must be at least the package bound.
  /// @return Number of initialized serialized bytes, or a typed capacity or codec error.
  /// @note Success never returns zero or a count larger than \ref max_serialized_size or
  /// \p destination. The generated callback catches allocation and codec exceptions.
  [[nodiscard]] expected<std::size_t, Error> encode(
      const T& value, std::span<std::byte> destination) const noexcept {
    if (encode_ == nullptr || max_serialized_size_ == 0U) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "generated package has no owning encoder"});
    }
    if (destination.size() < max_serialized_size_) {
      return make_unexpected(
          Error{ErrorCode::kResourceExhausted, "schema serialization storage is below its bound"});
    }
    auto encoded = encode_(value, destination.first(max_serialized_size_));
    if (!encoded) {
      return encoded;
    }
    if (*encoded == 0U || *encoded > max_serialized_size_) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "generated encoder returned an invalid payload size"});
    }
    return encoded;
  }

  /// @brief Verify serialized bytes and materialize one owning semantic value.
  /// @param bytes Complete serialized payload bytes.
  /// @return Owning value, or a typed verifier, allocation, or codec error.
  /// @note The exact generated verifier runs before the materializer. No borrowed generated root
  /// is exposed as the owning semantic value.
  [[nodiscard]] expected<T, Error> materialize(std::span<const std::byte> bytes) const noexcept {
    if (materialize_ == nullptr || max_serialized_size_ == 0U) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "generated package has no owning materializer"});
    }
    if (bytes.empty() || bytes.size() > max_serialized_size_) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "schema payload size is outside its package bound"});
    }
    if (!verify(bytes)) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "generated schema payload verification failed"});
    }
    return materialize_(bytes);
  }

 private:
  SchemaPackageDescriptor(
      SchemaIdentity identity, std::span<const std::byte> binary_schema,
      std::shared_ptr<const detail::SchemaPackageEvidenceStorage> evidence_owner,
      VerifyFunction verify, ParseFunction parse, std::size_t max_serialized_size,
      EncodeFunction encode, MaterializeFunction materialize) noexcept
      : identity_(identity),
        binary_schema_(binary_schema),
        evidence_owner_(std::move(evidence_owner)),
        evidence_(evidence_owner_->evidence()),
        verify_(verify),
        parse_(parse),
        max_serialized_size_(max_serialized_size),
        encode_(encode),
        materialize_(materialize) {}

  SchemaIdentity identity_{};
  std::span<const std::byte> binary_schema_{};
  std::shared_ptr<const detail::SchemaPackageEvidenceStorage> evidence_owner_{};
  SchemaPackageEvidence evidence_{};
  VerifyFunction verify_{};
  ParseFunction parse_{};
  std::size_t max_serialized_size_{};
  EncodeFunction encode_{};
  MaterializeFunction materialize_{};
};

/// @brief Deep-copy one generated package's pointer-free plan provenance.
/// @tparam T Semantic generated root/native type.
/// @param package Process-static generated package descriptor.
/// @return Owning record safe after all borrowed evidence views expire.
template <typename T>
[[nodiscard]] SchemaPackagePlanRecord schema_package_plan_record(
    const SchemaPackageDescriptor<T>& package) {
  const SchemaPackageEvidence& evidence = package.evidence();
  SchemaPackagePlanRecord result{
      .schema_id = package.identity().id,
      .binary_schema_size = static_cast<std::uint64_t>(package.binary_schema().size()),
      .flatc_build = std::string(evidence.flatc_build),
      .include_closure_sha256 = schema_include_closure_sha256(evidence.includes),
      .generator_format_version = evidence.generator_format_version,
      .package_format_version = evidence.package_format_version,
      .runtime_abi = evidence.runtime_abi,
  };
  result.ordered_flags.reserve(evidence.ordered_flags.size());
  for (const std::string_view flag : evidence.ordered_flags) {
    result.ordered_flags.emplace_back(flag);
  }
  return result;
}

/// @brief Describe the generated table root and finite codec bound for owning semantic type `T`.
/// @tparam T Owning generated native type.
/// @note Generated packages explicitly specialize this trait. Arbitrary classes, table views,
/// transport wrappers, and \ref Tensor must not specialize it.
template <typename T>
struct GeneratedSchemaNativeTraits;

/// @brief Describe an owning schema whose object graph carries Core Tensor fields.
///
/// A descriptor-backed Tensor schema is not a self-contained serialized value. Its generated
/// metadata and Tensor storage travel through one bounded descriptor-backed sample lifetime, and
/// the Tensor bytes are never passed to the ordinary generated-schema encoder. Specializations
/// identify the generated table root, the finite metadata envelope, and the canonically indexed
/// embedded Tensor members used by Runtime synchronization and retention.
///
/// @tparam T Owning generated application-facing type.
/// @note An explicit specialization opts the type into descriptor-backed schema admission. It must
/// also provide the corresponding bounded codec and lifetime operations.
template <typename T>
struct DescriptorTensorSchemaNativeTraits;

namespace detail {

template <typename T>
concept HasGeneratedSchemaNativeTraits = requires {
  typename GeneratedSchemaNativeTraits<T>::root_type;
  { GeneratedSchemaNativeTraits<T>::max_serialized_size } -> std::convertible_to<std::size_t>;
} && (GeneratedSchemaNativeTraits<T>::max_serialized_size > 0U);

template <typename T>
concept HasDescriptorTensorSchemaMetadataTraits = requires {
  typename DescriptorTensorSchemaNativeTraits<T>::root_type;
  { DescriptorTensorSchemaNativeTraits<T>::metadata_capacity } -> std::convertible_to<std::size_t>;
};

template <typename T>
concept HasIndexedDescriptorTensorSchemaAccess = requires(const T& value, const std::size_t index) {
  { DescriptorTensorSchemaNativeTraits<T>::tensor_count } -> std::convertible_to<std::size_t>;
  { DescriptorTensorSchemaNativeTraits<T>::tensor_at(value, index) } -> std::same_as<const Tensor*>;
};

template <typename T>
concept HasSingularDescriptorTensorSchemaAccess = requires(const T& value) {
  { DescriptorTensorSchemaNativeTraits<T>::tensor(value) } -> std::same_as<const Tensor*>;
};

template <typename T, std::size_t Count>
[[nodiscard]] consteval bool valid_descriptor_tensor_field_ids() {
  if constexpr (Count <= 1U) {
    return true;
  } else if constexpr (!requires {
                         DescriptorTensorSchemaNativeTraits<T>::tensor_field_ids;
                         {
                           DescriptorTensorSchemaNativeTraits<T>::tensor_field_ids.size()
                         } -> std::convertible_to<std::size_t>;
                       }) {
    return false;
  } else {
    constexpr auto& field_ids = DescriptorTensorSchemaNativeTraits<T>::tensor_field_ids;
    using FieldId = std::remove_cvref_t<decltype(field_ids[0U])>;
    if constexpr (!std::is_integral_v<FieldId> || field_ids.size() != Count) {
      return false;
    } else {
      for (std::size_t index = 0U; index < Count; ++index) {
        if constexpr (std::is_signed_v<FieldId>) {
          if (field_ids[index] < 0) {
            return false;
          }
        }
        if (static_cast<std::uintmax_t>(field_ids[index]) >
                std::numeric_limits<std::uint16_t>::max() ||
            (index > 0U && field_ids[index - 1U] >= field_ids[index])) {
          return false;
        }
      }
      return true;
    }
  }
}

template <typename T>
[[nodiscard]] consteval bool valid_indexed_descriptor_tensor_schema_access() {
  constexpr std::size_t kCount =
      static_cast<std::size_t>(DescriptorTensorSchemaNativeTraits<T>::tensor_count);
  return kCount > 0U && kCount <= 64U && valid_descriptor_tensor_field_ids<T, kCount>();
}

template <typename T>
[[nodiscard]] consteval bool valid_singular_descriptor_tensor_schema_access() {
  if constexpr (requires {
                  {
                    DescriptorTensorSchemaNativeTraits<T>::tensor_count
                  } -> std::convertible_to<std::size_t>;
                }) {
    return static_cast<std::size_t>(DescriptorTensorSchemaNativeTraits<T>::tensor_count) == 1U;
  }
  return true;
}

template <typename T>
[[nodiscard]] consteval bool valid_descriptor_tensor_schema_native_traits() {
  // Invalid or incomplete specializations evaluate false so this remains usable by a constraining
  // concept; a hard static_assert here would break substitution-based API selection.
  if constexpr (HasDescriptorTensorSchemaMetadataTraits<T>) {
    if constexpr (HasIndexedDescriptorTensorSchemaAccess<T>) {
      return DescriptorTensorSchemaNativeTraits<T>::metadata_capacity > 0U &&
             valid_indexed_descriptor_tensor_schema_access<T>();
    }
    if constexpr (HasSingularDescriptorTensorSchemaAccess<T>) {
      return DescriptorTensorSchemaNativeTraits<T>::metadata_capacity > 0U &&
             valid_singular_descriptor_tensor_schema_access<T>();
    }
  }
  return false;
}

/// @brief Check whether an owning generated type has a descriptor-backed Tensor contract.
/// @tparam T Exact owning generated application-facing type.
/// @return True when the type names one generated root, a positive bounded metadata envelope, and
/// one to 64 indexed Core Tensor accessors. A legacy singular accessor represents field zero.
template <typename T>
concept HasDescriptorTensorSchemaNativeTraits = valid_descriptor_tensor_schema_native_traits<T>();

/// Maximum serialized metadata bytes for a descriptor-backed Tensor schema.
template <typename T>
  requires HasDescriptorTensorSchemaNativeTraits<T>
inline constexpr std::size_t descriptor_tensor_schema_metadata_capacity_v =
    DescriptorTensorSchemaNativeTraits<T>::metadata_capacity;

/// Number of direct Tensor fields exposed by one descriptor-backed owning schema type.
///
/// The fallback preserves source compatibility for existing third-party specializations authored
/// before multi-Tensor field inventories were public.
template <typename T>
  requires HasDescriptorTensorSchemaNativeTraits<T>
inline constexpr std::size_t descriptor_tensor_schema_tensor_count_v = [] {
  if constexpr (requires {
                  {
                    DescriptorTensorSchemaNativeTraits<T>::tensor_count
                  } -> std::convertible_to<std::size_t>;
                }) {
    return static_cast<std::size_t>(DescriptorTensorSchemaNativeTraits<T>::tensor_count);
  } else {
    return std::size_t{1U};
  }
}();

/// @brief Return one indexed Tensor from a descriptor-backed owning schema value.
/// @tparam T Owning generated descriptor-backed schema type.
/// @param value Borrowed owning value.
/// @param tensor_index Dense ascending-field-id Tensor index.
/// @return Borrowed Tensor, or null when the indexed field is absent or out of range.
template <typename T>
  requires HasDescriptorTensorSchemaNativeTraits<T>
[[nodiscard]] const Tensor* descriptor_tensor_schema_tensor_at(
    const T& value, const std::size_t tensor_index) noexcept {
  if (tensor_index >= descriptor_tensor_schema_tensor_count_v<T>) {
    return nullptr;
  }
  if constexpr (requires {
                  {
                    DescriptorTensorSchemaNativeTraits<T>::tensor_at(value, tensor_index)
                  } -> std::same_as<const Tensor*>;
                }) {
    return DescriptorTensorSchemaNativeTraits<T>::tensor_at(value, tensor_index);
  } else {
    return tensor_index == 0U ? DescriptorTensorSchemaNativeTraits<T>::tensor(value) : nullptr;
  }
}

/// @brief Return the canonical FlatBuffers field id for one descriptor Tensor index.
/// @tparam T Owning generated descriptor-backed schema type.
/// @param tensor_index Dense ascending-field-id Tensor index.
/// @return Canonical field id when the specialization publishes its inventory.
template <typename T>
  requires HasDescriptorTensorSchemaNativeTraits<T>
[[nodiscard]] constexpr std::optional<std::uint16_t> descriptor_tensor_schema_field_id_at(
    const std::size_t tensor_index) noexcept {
  if constexpr (requires { DescriptorTensorSchemaNativeTraits<T>::tensor_field_ids; }) {
    constexpr auto& field_ids = DescriptorTensorSchemaNativeTraits<T>::tensor_field_ids;
    if (tensor_index < field_ids.size()) {
      return field_ids[tensor_index];
    }
  }
  return std::nullopt;
}

/// @brief Check one typed field selector against a descriptor Tensor schema inventory.
/// @tparam T Owning generated descriptor-backed schema type.
/// @param field_id FlatBuffers field id carried by the typed selector.
/// @param tensor_index Dense Tensor index carried by the typed selector.
/// @return True for an exact published inventory match, or for field zero of a legacy singular
/// specialization that predates published field inventories.
template <typename T>
  requires HasDescriptorTensorSchemaNativeTraits<T>
[[nodiscard]] constexpr bool descriptor_tensor_schema_field_selector_matches(
    const std::uint16_t field_id, const std::size_t tensor_index) noexcept {
  const auto expected = descriptor_tensor_schema_field_id_at<T>(tensor_index);
  if (expected) {
    return *expected == field_id;
  }
  return descriptor_tensor_schema_tensor_count_v<T> == 1U && tensor_index == 0U && field_id == 0U;
}

}  // namespace detail

/// @brief Return the package-declared finite serialized capacity for owning generated type `T`.
/// @tparam T Type with an explicit \ref GeneratedSchemaNativeTraits specialization.
template <typename T>
  requires detail::HasGeneratedSchemaNativeTraits<T>
inline constexpr std::size_t generated_schema_max_serialized_size_v =
    GeneratedSchemaNativeTraits<T>::max_serialized_size;

/// @brief Return the generated package for semantic root/native type `T`.
/// @tparam T Semantic generated type with an explicit package specialization.
/// @return Process-static exact generated package descriptor.
template <typename T>
const SchemaPackageDescriptor<T>& schema_package() noexcept = delete;

namespace detail {

/// Check that the generated package specialization for `T` is visible at the use site.
///
/// Generated native traits describe only the semantic root and finite capacity. Keeping this
/// separate expression check prevents a trait-only declaration from entering a typed endpoint and
/// reaching the deleted primary package function during Runtime binding.
template <typename T>
concept HasSchemaPackage = requires {
  { schema_package<T>() } -> std::same_as<const SchemaPackageDescriptor<T>&>;
};

[[nodiscard]] constexpr bool schema_identity_equal(const SchemaIdentity& lhs,
                                                   const SchemaIdentity& rhs) noexcept {
  return lhs.canonical_name == rhs.canonical_name && lhs.version == rhs.version &&
         lhs.schema_sha256 == rhs.schema_sha256 && lhs.id == rhs.id;
}

// Package generation has already chosen the canonical spelling. Validation therefore rejects
// traversal and platform-specific separators instead of normalizing them and changing evidence.
[[nodiscard]] constexpr bool canonical_package_path(std::string_view value) noexcept {
  if (value.empty() || value.front() == '/' || value.back() == '/' ||
      value.find('\0') != std::string_view::npos || value.find('\\') != std::string_view::npos ||
      value.find(':') != std::string_view::npos) {
    return false;
  }
  std::size_t begin = 0U;
  while (begin < value.size()) {
    const std::size_t end = value.find('/', begin);
    const std::string_view segment = value.substr(begin, end - begin);
    if (segment.empty() || segment == "." || segment == "..") {
      return false;
    }
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1U;
  }
  return true;
}

// An ordered `flatc` argument can be either an option or a logical path. Preserve ordinary option
// values (including C++ names such as `std::shared_ptr`) while rejecting every host-path spelling.
[[nodiscard]] constexpr bool canonical_generator_flag(std::string_view value) noexcept {
  if (value.empty() || value.front() == '/' || value.back() == '/' ||
      value.find('\0') != std::string_view::npos || value.find('\\') != std::string_view::npos ||
      value.find("//") != std::string_view::npos || value.find("=/") != std::string_view::npos ||
      value.find(":/") != std::string_view::npos) {
    return false;
  }
  for (std::size_t index = 0U; index < value.size(); ++index) {
    if (value[index] == ':' && (index == 0U || value[index - 1U] != ':') &&
        (index + 1U == value.size() || value[index + 1U] != ':')) {
      return false;
    }
  }

  std::size_t begin = 0U;
  while (begin <= value.size()) {
    const std::size_t end = value.find_first_of("/=", begin);
    const std::string_view segment = value.substr(begin, end - begin);
    if (segment == "." || segment == "..") {
      return false;
    }
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1U;
  }

  const auto invalid_joined_path = [value](std::string_view prefix) constexpr noexcept {
    return value.starts_with(prefix) && value.size() > prefix.size() &&
           !canonical_package_path(value.substr(prefix.size()));
  };
  return !invalid_joined_path("-I") && !invalid_joined_path("-o");
}

[[nodiscard]] constexpr bool canonical_include_root_mapping_entry(
    std::string_view value, std::size_t expected_index) noexcept {
  const std::size_t separator = value.find('=');
  if (separator == std::string_view::npos || separator == 0U || separator + 1U >= value.size() ||
      value.find('=', separator + 1U) != std::string_view::npos) {
    return false;
  }
  const std::string_view logical_root = value.substr(0U, separator);
  const std::string_view mapped_path = value.substr(separator + 1U);
  constexpr std::string_view kPrefix{"schema-root-"};
  if (value.find('\0') != std::string_view::npos || !logical_root.starts_with(kPrefix) ||
      mapped_path != ".") {
    return false;
  }
  const std::string_view index_text = logical_root.substr(kPrefix.size());
  if (index_text.empty() || (index_text.size() > 1U && index_text.front() == '0')) {
    return false;
  }
  std::size_t parsed_index = 0U;
  for (const char digit : index_text) {
    if (digit < '0' || digit > '9') {
      return false;
    }
    const std::size_t value = static_cast<std::size_t>(digit - '0');
    if (parsed_index > (std::numeric_limits<std::size_t>::max() - value) / 10U) {
      return false;
    }
    parsed_index = parsed_index * 10U + value;
  }
  return parsed_index == expected_index;
}

// Package evidence format v1 uses one grammar for every root count. Indices begin at zero and are
// contiguous in the same order as normalized -I flags; even a singleton is `schema-root-0=.`.
[[nodiscard]] constexpr bool canonical_include_root_mapping(std::string_view value) noexcept {
  if (value.empty() || value.front() == ';' || value.back() == ';') {
    return false;
  }
  std::size_t begin = 0U;
  std::size_t expected_index = 0U;
  while (begin < value.size()) {
    const std::size_t end = value.find(';', begin);
    const std::string_view entry = value.substr(begin, end - begin);
    if (!canonical_include_root_mapping_entry(entry, expected_index)) {
      return false;
    }
    ++expected_index;
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1U;
  }
  return true;
}

// Keep the UTF-8 decoder in one bounded pass so package evidence and schema names share the same
// overlong, surrogate, and range checks before either can enter diagnostic JSON.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] constexpr bool valid_schema_utf8(std::string_view value,
                                               bool reject_ascii_controls) noexcept {
  std::size_t index = 0U;
  while (index < value.size()) {
    const auto first = static_cast<unsigned char>(value[index++]);
    if (first <= 0x7FU) {
      if (reject_ascii_controls && (first < 0x20U || first == 0x7FU)) {
        return false;
      }
      continue;
    }
    std::uint32_t code_point{};
    std::size_t continuation_count{};
    if (first >= 0xC2U && first <= 0xDFU) {
      code_point = first & 0x1FU;
      continuation_count = 1U;
    } else if (first >= 0xE0U && first <= 0xEFU) {
      code_point = first & 0x0FU;
      continuation_count = 2U;
    } else if (first >= 0xF0U && first <= 0xF4U) {
      code_point = first & 0x07U;
      continuation_count = 3U;
    } else {
      return false;
    }
    if (index + continuation_count > value.size()) {
      return false;
    }
    for (std::size_t count = 0U; count < continuation_count; ++count) {
      const auto next = static_cast<unsigned char>(value[index++]);
      if ((next & 0xC0U) != 0x80U) {
        return false;
      }
      code_point = (code_point << 6U) | (next & 0x3FU);
    }
    if ((continuation_count == 2U && code_point < 0x800U) ||
        (continuation_count == 3U && code_point < 0x10000U) ||
        (code_point >= 0xD800U && code_point <= 0xDFFFU) || code_point > 0x10FFFFU) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool valid_schema_name_utf8(std::string_view value) noexcept {
  return !value.empty() && value.size() <= std::numeric_limits<std::uint32_t>::max() &&
         value.front() != '.' && value.back() != '.' && value.find('.') != std::string_view::npos &&
         valid_schema_utf8(value, true);
}

inline void append_diagnostic_json_string(std::string& output, std::string_view value) {
  constexpr std::string_view kHex{"0123456789abcdef"};
  output.push_back('"');
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"':
        output += "\\\"";
        break;
      case '\\':
        output += "\\\\";
        break;
      case '\b':
        output += "\\b";
        break;
      case '\f':
        output += "\\f";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        if (byte < 0x20U) {
          output += "\\u00";
          output.push_back(kHex[byte >> 4U]);
          output.push_back(kHex[byte & 0x0FU]);
        } else {
          output.push_back(static_cast<char>(byte));
        }
        break;
    }
  }
  output.push_back('"');
}

inline void append_diagnostic_hex(std::string& output, std::span<const std::byte> bytes) {
  constexpr std::string_view kHex{"0123456789abcdef"};
  output.push_back('"');
  for (const std::byte byte : bytes) {
    const auto value = std::to_integer<std::uint8_t>(byte);
    output.push_back(kHex[value >> 4U]);
    output.push_back(kHex[value & 0x0FU]);
  }
  output.push_back('"');
}

inline void append_package_plan_record(std::string& output, const SchemaPackagePlanRecord& record) {
  output += "{\"binary_schema_size\":";
  output += std::to_string(record.binary_schema_size);
  output += ",\"flatc_build\":";
  append_diagnostic_json_string(output, record.flatc_build);
  output += ",\"generator_format_version\":";
  output += std::to_string(record.generator_format_version);
  output += ",\"include_closure_sha256\":";
  append_diagnostic_hex(output, record.include_closure_sha256);
  output += ",\"ordered_flags\":[";
  bool first = true;
  for (const std::string& flag : record.ordered_flags) {
    if (!first) {
      output.push_back(',');
    }
    first = false;
    append_diagnostic_json_string(output, flag);
  }
  output += "],\"package_format_version\":";
  output += std::to_string(record.package_format_version);
  output += ",\"runtime_abi\":{\"major\":";
  output += std::to_string(record.runtime_abi.major);
  output += ",\"minor\":";
  output += std::to_string(record.runtime_abi.minor);
  output += ",\"revision\":";
  output += std::to_string(record.runtime_abi.revision);
  output += "},\"schema_id\":";
  append_diagnostic_hex(output, record.schema_id.value);
  output.push_back('}');
}

inline void append_attach_identity(std::string& output, const SchemaIdentity& identity) {
  output += "{\"canonical_name\":";
  append_diagnostic_json_string(output, identity.canonical_name);
  output += ",\"schema_id\":";
  append_diagnostic_hex(output, identity.id.value);
  output += ",\"schema_sha256\":";
  append_diagnostic_hex(output, identity.schema_sha256);
  output += ",\"version\":";
  output += std::to_string(identity.version);
  output.push_back('}');
}

inline void append_local_attach_evidence(std::string& output, const SchemaIdentity& identity,
                                         const SchemaPackagePlanRecord& package) {
  output += "{\"canonical_name\":";
  append_diagnostic_json_string(output, identity.canonical_name);
  output += ",\"package\":";
  append_package_plan_record(output, package);
  output += ",\"schema_id\":";
  append_diagnostic_hex(output, identity.id.value);
  output += ",\"schema_sha256\":";
  append_diagnostic_hex(output, identity.schema_sha256);
  output += ",\"version\":";
  output += std::to_string(identity.version);
  output.push_back('}');
}

template <typename T>
[[nodiscard]] std::string schema_attach_mismatch_message(
    const SchemaIdentity& expected_identity, const SchemaPackageDescriptor<T>& local_package) {
  const SchemaPackagePlanRecord local_record = schema_package_plan_record(local_package);
  std::string output{"attached endpoint requires a different schema package; expected="};
  append_attach_identity(output, expected_identity);
  output += "; local=";
  append_local_attach_evidence(output, local_package.identity(), local_record);
  return output;
}

}  // namespace detail

/// @brief Validate one generated package's local authoring evidence.
/// @tparam T Semantic generated root/native type.
/// @param package Candidate package descriptor.
/// @return Empty success or a local invalid-argument/codec error.
///
/// Validation recomputes the artifact digest and compact identifier and checks
/// that every borrowed generator-evidence path is canonical. It never reruns
/// `flatc`, rewrites paths, negotiates a version, or invokes a payload parser.
template <typename T>
[[nodiscard]] expected<void, Error> validate_schema_package(
    const SchemaPackageDescriptor<T>& package) {
  const SchemaIdentity declared = schema_identity<T>();
  const SchemaIdentity& identity = package.identity();
  if (!detail::valid_schema_name_utf8(identity.canonical_name) || identity.version == 0U) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "generated schema identity name/version is invalid"});
  }
  if (!detail::schema_identity_equal(identity, declared)) {
    return make_unexpected(
        Error{ErrorCode::kCodecError, "generated package semantic identity mismatch"});
  }
  if (package.binary_schema().empty() ||
      schema_artifact_sha256(package.binary_schema()) != identity.schema_sha256) {
    return make_unexpected(
        Error{ErrorCode::kCodecError, "generated package binary schema digest mismatch"});
  }
  if (derive_schema_id(identity) != identity.id) {
    return make_unexpected(
        Error{ErrorCode::kCodecError, "generated package compact schema identity mismatch"});
  }

  const SchemaPackageEvidence& evidence = package.evidence();
  const bool has_borrowed_root_api = package.has_parser();
  const bool has_owning_value_api =
      package.max_serialized_size() > 0U && package.has_encoder() && package.has_materializer();
  const bool partial_owning_value_api =
      package.max_serialized_size() > 0U || package.has_encoder() || package.has_materializer();
  if (partial_owning_value_api != has_owning_value_api ||
      package.max_serialized_size() > std::numeric_limits<std::uint32_t>::max()) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "generated schema owning codec evidence is invalid"});
  }
  if (evidence.flatc_build.empty() || !detail::valid_schema_utf8(evidence.flatc_build, false) ||
      evidence.ordered_flags.empty() ||
      !detail::valid_schema_utf8(evidence.root_logical_path, false) ||
      !detail::canonical_package_path(evidence.root_logical_path) ||
      !detail::valid_schema_utf8(evidence.include_root_mapping, false) ||
      !detail::canonical_include_root_mapping(evidence.include_root_mapping) ||
      !evidence.host_paths_stripped || evidence.generator_format_version == 0U ||
      evidence.package_format_version == 0U || evidence.runtime_abi.major == 0U ||
      evidence.includes.size() > std::numeric_limits<std::uint32_t>::max() ||
      !package.has_verifier() || (!has_borrowed_root_api && !has_owning_value_api)) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "generated schema package evidence is incomplete"});
  }
  for (const std::string_view flag : evidence.ordered_flags) {
    if (!detail::valid_schema_utf8(flag, false) || !detail::canonical_generator_flag(flag)) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "generated schema flag evidence is invalid"});
    }
  }
  for (const SchemaPackageSource& source : evidence.includes) {
    if (source.logical_path.size() > std::numeric_limits<std::uint32_t>::max() ||
        !detail::valid_schema_utf8(source.logical_path, false) ||
        !detail::canonical_package_path(source.logical_path)) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "generated schema include evidence is invalid"});
    }
  }
  return {};
}

/// @brief Validate an exact generated package at endpoint attachment.
/// @tparam T Semantic generated root/native type.
/// @param expected_identity Complete identity admitted by the catalog/plan.
/// @param package Locally installed generated package.
/// @return Empty success, local package error, or `kIncompatible` for exact-identity mismatch.
template <typename T>
[[nodiscard]] expected<void, Error> validate_schema_package_for_attach(
    const SchemaIdentity& expected_identity, const SchemaPackageDescriptor<T>& package) {
  auto local = validate_schema_package(package);
  if (!local) {
    return make_unexpected(std::move(local).error());
  }
  if (!detail::schema_identity_equal(expected_identity, package.identity())) {
    return make_unexpected(
        Error{ErrorCode::kIncompatible,
              detail::schema_attach_mismatch_message(expected_identity, package)});
  }
  return {};
}

/// @brief Validate a peer schema envelope before payload verification.
/// @param header Received versioned envelope prefix.
/// @param expected_identity Exact identity bound during attachment.
/// @param expected_generation Exact nonzero endpoint generation bound during attachment.
/// @param received_payload_size Bytes present after the prefix.
/// @return Empty success or a protocol error; no payload bytes are exposed on failure.
[[nodiscard]] inline expected<void, Error> validate_received_schema_envelope(
    const SchemaEnvelopeHeader& header, const SchemaIdentity& expected_identity,
    std::uint64_t expected_generation, std::size_t received_payload_size,
    std::uint32_t expected_abi_version = kSchemaEnvelopeABIVersion) {
  if (expected_generation == 0U) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "expected schema endpoint generation is zero"});
  }
  if (header.abi_version != expected_abi_version ||
      header.header_size != kSchemaEnvelopeHeaderSize) {
    return make_unexpected(Error{ErrorCode::kProtocolError, "schema envelope ABI mismatch"});
  }
  if (header.schema_id != expected_identity.id ||
      header.schema_version != expected_identity.version) {
    return make_unexpected(Error{ErrorCode::kProtocolError, "schema envelope identity mismatch"});
  }
  if (header.endpoint_generation != expected_generation) {
    return make_unexpected(Error{ErrorCode::kProtocolError, "schema envelope generation mismatch"});
  }
  if (received_payload_size != header.payload_size) {
    return make_unexpected(
        Error{ErrorCode::kProtocolError, "schema envelope payload size mismatch"});
  }
  return {};
}

/// @brief Verify and parse one received generated-schema payload in protocol order.
/// @tparam T Semantic generated root/native type.
/// @param package Exact package bound at attachment.
/// @param header Received versioned envelope prefix.
/// @param expected_identity Exact catalog identity bound at attachment.
/// @param expected_generation Exact endpoint generation bound at attachment.
/// @param payload Received serialized payload bytes.
/// @return Borrowed generated root view, or a protocol/local error without typed exposure.
/// @warning A successful view remains valid only while `payload` is alive and unmodified.
template <typename T>
[[nodiscard]] expected<const T*, Error> verify_and_parse_received_schema(
    const SchemaPackageDescriptor<T>& package, const SchemaEnvelopeHeader& header,
    const SchemaIdentity& expected_identity, std::uint64_t expected_generation,
    std::span<const std::byte> payload,
    std::uint32_t expected_abi_version = kSchemaEnvelopeABIVersion) {
  auto envelope = validate_received_schema_envelope(
      header, expected_identity, expected_generation, payload.size(), expected_abi_version);
  if (!envelope) {
    return make_unexpected(std::move(envelope).error());
  }
  if (!detail::schema_identity_equal(package.identity(), expected_identity)) {
    return make_unexpected(
        Error{ErrorCode::kIncompatible, "bound generated schema package identity changed"});
  }
  if (!package.has_parser()) {
    return make_unexpected(
        Error{ErrorCode::kNotSupported, "generated schema package has no borrowed parser"});
  }
  if (!package.verify(payload)) {
    return make_unexpected(
        Error{ErrorCode::kProtocolError, "generated schema payload verification failed"});
  }
  const T* parsed = package.parse_verified(payload);
  if (parsed == nullptr) {
    return make_unexpected(Error{ErrorCode::kProtocolError, "generated schema parser failed"});
  }
  return parsed;
}

/// @brief Verify and materialize one received owning generated-schema value in protocol order.
/// @tparam T Owning semantic generated type.
/// @param package Exact package bound at attachment.
/// @param header Received versioned envelope prefix.
/// @param expected_identity Exact catalog identity bound at attachment.
/// @param expected_generation Exact endpoint generation bound at attachment.
/// @param payload Received serialized payload bytes.
/// @return Owning semantic value, or a protocol/local error without typed exposure.
template <typename T>
[[nodiscard]] expected<T, Error> verify_and_materialize_received_schema(
    const SchemaPackageDescriptor<T>& package, const SchemaEnvelopeHeader& header,
    const SchemaIdentity& expected_identity, std::uint64_t expected_generation,
    std::span<const std::byte> payload,
    std::uint32_t expected_abi_version = kSchemaEnvelopeABIVersion) {
  auto envelope = validate_received_schema_envelope(
      header, expected_identity, expected_generation, payload.size(), expected_abi_version);
  if (!envelope) {
    return make_unexpected(std::move(envelope).error());
  }
  if (!detail::schema_identity_equal(package.identity(), expected_identity)) {
    return make_unexpected(
        Error{ErrorCode::kIncompatible, "bound generated schema package identity changed"});
  }
  auto value = package.materialize(payload);
  if (!value) {
    Error error = std::move(value).error();
    if (error.code == ErrorCode::kCodecError) {
      error.code = ErrorCode::kProtocolError;
    }
    return make_unexpected(std::move(error));
  }
  return value;
}

}  // namespace holoscan
