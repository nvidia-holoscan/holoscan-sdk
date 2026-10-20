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
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <holoscan/holoscan_schema_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/schema_identity.hpp>

namespace holoscan {

class Tensor;

namespace schema {
struct ImageT;
}  // namespace schema

/// @brief Identify the only schema-bearing wrapper/envelope ABI emitted by this SDK.
inline constexpr std::uint32_t kSchemaEnvelopeABIVersion = 1U;
/// @brief Identify the descriptor-array envelope ABI emitted by descriptor-backed Tensor schemas.
inline constexpr std::uint32_t kDescriptorTensorSchemaEnvelopeABIVersion = 1U;
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
  SchemaId schema_family_id{};            ///< Stable 128-bit compatibility-family identity.
  std::uint32_t payload_size{};           ///< Serialized payload bytes after this prefix.
  std::uint32_t transport_profile_tag{};  ///< Zero inline or exact descriptor carriage profile.
  std::uint64_t endpoint_generation{};    ///< Exact attached endpoint generation.
};

static_assert(std::is_standard_layout_v<SchemaEnvelopeHeader>);
static_assert(std::is_trivially_copyable_v<SchemaEnvelopeHeader>);
static_assert(sizeof(SchemaEnvelopeHeader) == kSchemaEnvelopeHeaderSize);

/// @brief Classify the Holoscan carriage profile owned outside FlatBuffers identity.
enum class SchemaTransportProfileKind : std::uint8_t {
  kInlineFlatBuffer = 0,  ///< Self-contained serialized FlatBuffer bytes.
  kDescriptorTensor = 1,  ///< Metadata plus out-of-band Tensor descriptors.
};

/// @brief Carry the minimal semantic inputs of a transport profile.
///
/// Wrapper ABI, all-fields masks, carriage mode, and the compact profile tag are derived from
/// these inputs. They are not independently stored facts that can disagree.
struct SchemaTransportProfile {
  SchemaTransportProfileKind kind{SchemaTransportProfileKind::kInlineFlatBuffer};
  std::span<const std::uint16_t> tensor_field_ids;  ///< Canonical generated field inventory.
  std::uint32_t metadata_capacity_bytes{};          ///< Positive bounded metadata bytes.
};

/// @brief Derive the complete descriptor field mask for a generated Tensor-field count.
/// @param tensor_count Number of generated Tensor fields, in the supported range 1 through 64.
/// @return An all-fields mask, or zero when tensor_count is outside the supported range.
[[nodiscard]] constexpr std::uint64_t descriptor_tensor_all_fields_mask(
    const std::size_t tensor_count) noexcept {
  if (tensor_count == 0U || tensor_count > 64U) {
    return 0U;
  }
  return tensor_count == 64U ? std::numeric_limits<std::uint64_t>::max()
                             : (std::uint64_t{1U} << tensor_count) - 1U;
}

/// @brief Derive the closed descriptor-Tensor transport-profile tag.
/// @param tensor_field_ids Canonical ascending FlatBuffers Tensor field IDs.
/// @param metadata_capacity Positive bounded metadata capacity.
/// @return Nonzero deterministic tag, or zero when the supplied profile is invalid.
[[nodiscard]] constexpr std::uint32_t derive_descriptor_tensor_transport_profile_tag(
    const std::span<const std::uint16_t> tensor_field_ids,
    const std::size_t metadata_capacity) noexcept {
  if (tensor_field_ids.empty() || tensor_field_ids.size() > 64U || metadata_capacity == 0U ||
      metadata_capacity > std::numeric_limits<std::uint32_t>::max()) {
    return 0U;
  }
  for (std::size_t index = 1U; index < tensor_field_ids.size(); ++index) {
    if (tensor_field_ids[index - 1U] >= tensor_field_ids[index]) {
      return 0U;
    }
  }
  const std::uint64_t presence_mask = descriptor_tensor_all_fields_mask(tensor_field_ids.size());
  // Presence and carriage are separate compatibility coordinates even though the initial ABI
  // requires every generated Tensor field to be both present and descriptor-carried.
  const std::uint64_t descriptor_mask = presence_mask;
  detail::SchemaSha256Builder hash;
  hash.add("holoscan.schema.descriptor_tensor_profile.v1");
  hash.add_u32_be(kDescriptorTensorSchemaEnvelopeABIVersion);
  hash.add_u32_be(static_cast<std::uint32_t>(tensor_field_ids.size()));
  for (const std::uint16_t field_id : tensor_field_ids) {
    hash.add_u32_be(field_id);
  }
  hash.add_u32_be(static_cast<std::uint32_t>(presence_mask >> 32U));
  hash.add_u32_be(static_cast<std::uint32_t>(presence_mask));
  hash.add_u32_be(static_cast<std::uint32_t>(descriptor_mask >> 32U));
  hash.add_u32_be(static_cast<std::uint32_t>(descriptor_mask));
  hash.add_u32_be(static_cast<std::uint32_t>(metadata_capacity));
  hash.add_u32_be(1U);  // The current profile always uses out-of-band descriptor carriage.
  const auto digest = hash.finish();
  std::uint32_t tag{};
  for (std::size_t index = 0U; index < sizeof(tag); ++index) {
    tag = (tag << 8U) | std::to_integer<std::uint8_t>(digest[index]);
  }
  return tag == 0U ? std::numeric_limits<std::uint32_t>::max() : tag;
}

/// @brief Derive the complete all-fields mask for the current descriptor profile.
[[nodiscard]] constexpr std::uint64_t schema_transport_profile_field_mask(
    const SchemaTransportProfile& profile) noexcept {
  if (profile.kind != SchemaTransportProfileKind::kDescriptorTensor ||
      profile.tensor_field_ids.empty() || profile.tensor_field_ids.size() > 64U) {
    return 0U;
  }
  return descriptor_tensor_all_fields_mask(profile.tensor_field_ids.size());
}

/// @brief Derive the compact tag used by catalog and wire boundaries.
[[nodiscard]] constexpr std::uint32_t schema_transport_profile_tag(
    const SchemaTransportProfile& profile) noexcept {
  if (profile.kind != SchemaTransportProfileKind::kDescriptorTensor) {
    return 0U;
  }
  return derive_descriptor_tensor_transport_profile_tag(profile.tensor_field_ids,
                                                        profile.metadata_capacity_bytes);
}

/// @brief Build one exact descriptor-Tensor profile from generated and Holoscan carriage facts.
/// @param tensor_field_ids Canonical field inventory with storage that outlives every returned
/// profile and package that borrows it. Generated bindings provide static storage.
/// @param metadata_capacity Positive bounded descriptor metadata capacity.
/// @return Descriptor-Tensor profile borrowing p tensor_field_ids.
[[nodiscard]] constexpr SchemaTransportProfile make_descriptor_tensor_transport_profile(
    const std::span<const std::uint16_t> tensor_field_ids,
    const std::size_t metadata_capacity) noexcept {
  return SchemaTransportProfile{
      .kind = SchemaTransportProfileKind::kDescriptorTensor,
      .tensor_field_ids = tensor_field_ids,
      .metadata_capacity_bytes = static_cast<std::uint32_t>(metadata_capacity),
  };
}

/// @brief Carry finite local codec bounds separate from schema identity.
struct SchemaCodecCapacity {
  std::size_t max_serialized_size{};  ///< Positive owning bound, or zero for borrowed roots.
};

/// @brief Carry process-local generated codec callbacks for one semantic type.
/// @tparam T Borrowed root or owning object-API type.
template <typename T>
struct SchemaCodecDescriptor {
  /// @brief Verify serialized bytes using the exact generated root verifier.
  using VerifyFunction = bool (*)(std::span<const std::byte>) noexcept;
  /// @brief Parse already verified bytes into a borrowed generated root view.
  using ParseFunction = const T* (*)(std::span<const std::byte>) noexcept;
  /// @brief Serialize one owning semantic value into caller-owned bounded storage.
  using EncodeFunction = expected<std::size_t, Error> (*)(const T&, std::span<std::byte>) noexcept;
  /// @brief Materialize one owning semantic value from already verified bytes.
  using MaterializeFunction = expected<T, Error> (*)(std::span<const std::byte>) noexcept;

  VerifyFunction verify{};            ///< Generated verifier callback.
  ParseFunction parse{};              ///< Optional borrowed-root parser.
  EncodeFunction encode{};            ///< Optional bounded owning encoder.
  MaterializeFunction materialize{};  ///< Optional owning materializer.
};

/// @brief Describe one local generated schema package for semantic type `T`.
/// @tparam T Semantic generated FlatBuffers root/native type.
///
/// Identity, codec callbacks, finite capacity, and transport profile remain distinct. The family
/// view is projected from the one stored complete identity. The descriptor never embeds BFBS bytes
/// and does not own a transport endpoint.
/// Callback addresses never enter identity, persisted artifacts, or messages.
template <typename T>
class SchemaPackageDescriptor {
 public:
  using Codec = SchemaCodecDescriptor<T>;  ///< Process-local codec callback bundle.

  /// @brief Create a descriptor for one generated schema package.
  /// @param identity Stable family identity and independent buffer identifier.
  /// @param codec Generated verifier/parser/owning codec callbacks.
  /// @param capacity Finite local codec bound.
  /// @param transport_profile Holoscan carriage profile separate from identity.
  constexpr SchemaPackageDescriptor(SchemaIdentity identity, Codec codec,
                                    SchemaCodecCapacity capacity,
                                    SchemaTransportProfile transport_profile) noexcept
      : identity_(identity),
        codec_(codec),
        capacity_(capacity),
        transport_profile_(transport_profile),
        transport_profile_tag_(schema_transport_profile_tag(transport_profile)) {}

  /// @brief Return the stable family identity and independent buffer identifier.
  [[nodiscard]] constexpr const SchemaIdentity& identity() const noexcept { return identity_; }
  /// @brief Return the stable schema compatibility-family identity.
  [[nodiscard]] constexpr SchemaFamilyIdentity family_identity() const noexcept {
    return schema_family_identity_of(identity_);
  }
  /// @brief Return the process-local generated codec descriptor.
  [[nodiscard]] constexpr const Codec& codec() const noexcept { return codec_; }
  /// @brief Return the finite local codec capacity.
  [[nodiscard]] constexpr SchemaCodecCapacity capacity() const noexcept { return capacity_; }
  /// @brief Return the Holoscan transport profile.
  [[nodiscard]] constexpr SchemaTransportProfile transport_profile() const noexcept {
    return transport_profile_;
  }
  /// @brief Return the compact tag of \ref transport_profile, derived once at construction.
  ///
  /// Deriving the tag hashes the profile with SHA-256. Receive and send paths read this stored
  /// value instead of deriving it per message.
  [[nodiscard]] constexpr std::uint32_t transport_profile_tag() const noexcept {
    return transport_profile_tag_;
  }
  /// @brief Return whether the exact generated verifier is present.
  [[nodiscard]] constexpr bool has_verifier() const noexcept { return codec_.verify != nullptr; }
  /// @brief Return whether the generated parser is present.
  [[nodiscard]] constexpr bool has_parser() const noexcept { return codec_.parse != nullptr; }
  /// @brief Return the finite serialized-payload capacity for owning values.
  /// @return Positive byte capacity for an owning codec, or zero for a borrowed-root descriptor.
  [[nodiscard]] constexpr std::size_t max_serialized_size() const noexcept {
    return capacity_.max_serialized_size;
  }
  /// @brief Return whether this package can serialize owning semantic values.
  [[nodiscard]] constexpr bool has_encoder() const noexcept { return codec_.encode != nullptr; }
  /// @brief Return whether this package can materialize owning semantic values.
  [[nodiscard]] constexpr bool has_materializer() const noexcept {
    return codec_.materialize != nullptr;
  }

  /// @brief Invoke the exact generated verifier for serialized payload bytes.
  /// @param bytes Serialized payload bytes.
  /// @return `true` only when the generated root verifier accepts the complete buffer.
  [[nodiscard]] bool verify(std::span<const std::byte> bytes) const noexcept {
    return codec_.verify != nullptr && codec_.verify(bytes);
  }

  /// @brief Parse bytes that the caller already verified with this descriptor.
  /// @param bytes Serialized payload bytes previously accepted by \ref verify.
  /// @return Borrowed generated root view, or null when no parser is installed.
  /// @warning The returned view is invalidated when `bytes` is destroyed or modified.
  [[nodiscard]] const T* parse_verified(std::span<const std::byte> bytes) const noexcept {
    return codec_.parse != nullptr ? codec_.parse(bytes) : nullptr;
  }

  /// @brief Serialize an owning semantic value into bounded caller-owned storage.
  /// @param value Owning generated value to serialize.
  /// @param destination Preallocated destination whose size must be at least the package bound.
  /// @return Number of initialized serialized bytes, or a typed capacity or codec error.
  /// @note Success never returns zero or a count larger than \ref max_serialized_size or
  /// \p destination. The generated callback catches allocation and codec exceptions.
  [[nodiscard]] expected<std::size_t, Error> encode(
      const T& value, std::span<std::byte> destination) const noexcept {
    if (codec_.encode == nullptr || capacity_.max_serialized_size == 0U) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "generated package has no owning encoder"});
    }
    if (destination.size() < capacity_.max_serialized_size) {
      return make_unexpected(
          Error{ErrorCode::kResourceExhausted, "schema serialization storage is below its bound"});
    }
    auto encoded = codec_.encode(value, destination.first(capacity_.max_serialized_size));
    if (!encoded) {
      return encoded;
    }
    if (*encoded == 0U || *encoded > capacity_.max_serialized_size) {
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
    if (codec_.materialize == nullptr || capacity_.max_serialized_size == 0U) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "generated package has no owning materializer"});
    }
    if (bytes.empty() || bytes.size() > capacity_.max_serialized_size) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "schema payload size is outside its package bound"});
    }
    if (!verify(bytes)) {
      return make_unexpected(
          Error{ErrorCode::kCodecError, "generated schema payload verification failed"});
    }
    return codec_.materialize(bytes);
  }

 private:
  SchemaIdentity identity_{};
  Codec codec_{};
  SchemaCodecCapacity capacity_{};
  SchemaTransportProfile transport_profile_{};
  std::uint32_t transport_profile_tag_{};
};

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
  DescriptorTensorSchemaNativeTraits<T>::tensor_field_ids;
  { DescriptorTensorSchemaNativeTraits<T>::tensor_at(value, index) } -> std::same_as<const Tensor*>;
};

template <typename T, std::size_t Count>
[[nodiscard]] consteval bool valid_descriptor_tensor_field_ids() {
  if constexpr (!requires {
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
[[nodiscard]] consteval bool valid_descriptor_tensor_schema_native_traits() {
  // Invalid or incomplete specializations evaluate false so this remains usable by a constraining
  // concept; a hard static_assert here would break substitution-based API selection.
  if constexpr (HasDescriptorTensorSchemaMetadataTraits<T>) {
    if constexpr (HasIndexedDescriptorTensorSchemaAccess<T>) {
      return DescriptorTensorSchemaNativeTraits<T>::metadata_capacity > 0U &&
             valid_indexed_descriptor_tensor_schema_access<T>();
    }
  }
  return false;
}

/// @brief Check whether an owning generated type has a descriptor-backed Tensor contract.
/// @tparam T Exact owning generated application-facing type.
/// @return True when the type names one generated root, a positive bounded metadata envelope, and
/// one to 64 canonically identified indexed Core Tensor accessors.
template <typename T>
concept HasDescriptorTensorSchemaNativeTraits = valid_descriptor_tensor_schema_native_traits<T>();

/// Canonical generated Tensor field-id inventory used by the descriptor transport profile.
template <typename T>
  requires HasDescriptorTensorSchemaNativeTraits<T>
inline constexpr auto descriptor_tensor_schema_field_ids_v = [] {
  constexpr std::size_t kCount =
      static_cast<std::size_t>(DescriptorTensorSchemaNativeTraits<T>::tensor_count);
  std::array<std::uint16_t, kCount> result{};
  std::copy(DescriptorTensorSchemaNativeTraits<T>::tensor_field_ids.begin(),
            DescriptorTensorSchemaNativeTraits<T>::tensor_field_ids.end(),
            result.begin());
  return result;
}();

/// Exact descriptor profile generated from field IDs and Holoscan carriage policy.
template <typename T>
  requires HasDescriptorTensorSchemaNativeTraits<T>
inline constexpr SchemaTransportProfile descriptor_tensor_schema_transport_profile_v =
    make_descriptor_tensor_transport_profile(
        std::span{descriptor_tensor_schema_field_ids_v<T>},
        DescriptorTensorSchemaNativeTraits<T>::metadata_capacity);

/// Number of direct Tensor fields exposed by one descriptor-backed owning schema type.
template <typename T>
  requires HasDescriptorTensorSchemaNativeTraits<T>
inline constexpr std::size_t descriptor_tensor_schema_tensor_count_v =
    static_cast<std::size_t>(DescriptorTensorSchemaNativeTraits<T>::tensor_count);

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
  return DescriptorTensorSchemaNativeTraits<T>::tensor_at(value, tensor_index);
}

/// @brief Return the canonical FlatBuffers field id for one descriptor Tensor index.
/// @tparam T Owning generated descriptor-backed schema type.
/// @param tensor_index Dense ascending-field-id Tensor index.
/// @return Canonical field id, or no value when the index is out of range.
template <typename T>
  requires HasDescriptorTensorSchemaNativeTraits<T>
[[nodiscard]] constexpr std::optional<std::uint16_t> descriptor_tensor_schema_field_id_at(
    const std::size_t tensor_index) noexcept {
  constexpr auto& field_ids = DescriptorTensorSchemaNativeTraits<T>::tensor_field_ids;
  if (tensor_index < field_ids.size()) {
    return field_ids[tensor_index];
  }
  return std::nullopt;
}

/// @brief Check one typed field selector against a descriptor Tensor schema inventory.
/// @tparam T Owning generated descriptor-backed schema type.
/// @param field_id FlatBuffers field id carried by the typed selector.
/// @param tensor_index Dense Tensor index carried by the typed selector.
/// @return True for an exact published inventory match.
template <typename T>
  requires HasDescriptorTensorSchemaNativeTraits<T>
[[nodiscard]] constexpr bool descriptor_tensor_schema_field_selector_matches(
    const std::uint16_t field_id, const std::size_t tensor_index) noexcept {
  const auto expected = descriptor_tensor_schema_field_id_at<T>(tensor_index);
  return expected && *expected == field_id;
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

/// @brief Return a generated root package from its single schema binding.
template <typename Root>
  requires detail::HasFlatBufferSchemaRootBinding<Root>
inline const SchemaPackageDescriptor<Root>& schema_package() noexcept;

/// @brief Return a generated owning package through its root schema binding.
template <typename Native>
  requires detail::HasFlatBufferSchemaNativeBinding<Native>
inline const SchemaPackageDescriptor<Native>& schema_package() noexcept;

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

/// Format complete expected and local identities for a cold-path attachment diagnostic.
/// @note A target that instantiates attachment validation must link `holoscan::schema`; this
/// declaration is provided by that compiled library rather than FlatBuffers alone.
[[nodiscard]] HOLOSCAN_SCHEMA_EXPORT std::string schema_attach_mismatch_message(
    const SchemaIdentity& expected_identity, const SchemaIdentity& local_identity);

}  // namespace detail

/// @brief Validate one generated package's local contract.
/// @tparam T Semantic generated root/native type.
/// @param package Candidate package descriptor.
/// @return Empty success or a local invalid-argument/codec error.
///
/// Validation checks family identity, the independent buffer identifier, codec
/// completeness, finite bounds, and transport profile.
/// It never requires embedded BFBS bytes or invokes a payload parser.
template <typename T>
[[nodiscard]] expected<void, Error> validate_schema_package(
    const SchemaPackageDescriptor<T>& package) {
  const SchemaIdentity declared = schema_identity<T>();
  const SchemaIdentity& identity = package.identity();
  if (!valid_schema_identity(identity)) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "generated schema family identity is invalid"});
  }
  if (!schema_identity_equal(identity, declared)) {
    return make_unexpected(
        Error{ErrorCode::kCodecError, "generated package semantic identity mismatch"});
  }
  const bool has_borrowed_root_api = package.has_parser();
  const bool has_owning_value_api =
      package.max_serialized_size() > 0U && package.has_encoder() && package.has_materializer();
  const bool partial_owning_value_api =
      package.max_serialized_size() > 0U || package.has_encoder() || package.has_materializer();
  const SchemaTransportProfile transport_profile = package.transport_profile();
  const SchemaTransportProfileKind transport_kind = transport_profile.kind;
  const bool inline_profile = transport_kind == SchemaTransportProfileKind::kInlineFlatBuffer &&
                              transport_profile.tensor_field_ids.empty() &&
                              transport_profile.metadata_capacity_bytes == 0U;
  const bool descriptor_profile = transport_kind == SchemaTransportProfileKind::kDescriptorTensor &&
                                  schema_transport_profile_tag(transport_profile) != 0U;
  const bool descriptor_native_profile = descriptor_profile && !has_borrowed_root_api;
  if (partial_owning_value_api != has_owning_value_api ||
      package.max_serialized_size() > std::numeric_limits<std::uint32_t>::max() ||
      (!inline_profile && !descriptor_profile) || !package.has_verifier() ||
      (!has_borrowed_root_api && !has_owning_value_api && !descriptor_native_profile)) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "generated schema codec/profile contract is invalid"});
  }
  return {};
}

/// @brief Validate an exact generated package at endpoint attachment.
/// @tparam T Semantic generated root/native type.
/// @param expected_identity Complete identity admitted by the catalog/plan.
/// @param package Locally installed generated package.
/// @param expected_transport_profile_tag Exact endpoint carriage profile tag.
/// @return Empty success, local package error, or `kIncompatible` for exact-identity mismatch.
template <typename T>
[[nodiscard]] expected<void, Error> validate_schema_package_for_attach(
    const SchemaIdentity& expected_identity, const SchemaPackageDescriptor<T>& package,
    const std::uint32_t expected_transport_profile_tag = 0U) {
  auto local = validate_schema_package(package);
  if (!local) {
    return make_unexpected(std::move(local).error());
  }
  if (!schema_identity_equal(expected_identity, package.identity()) ||
      expected_transport_profile_tag != package.transport_profile_tag()) {
    return make_unexpected(
        Error{ErrorCode::kIncompatible,
              detail::schema_attach_mismatch_message(expected_identity, package.identity())});
  }
  return {};
}

/// @brief Validate a peer schema envelope before payload verification.
/// @param header Received versioned envelope prefix.
/// @param expected_identity Exact identity bound during attachment.
/// @param expected_generation Exact nonzero endpoint generation bound during attachment.
/// @param received_payload_size Bytes present after the prefix.
/// @param expected_abi_version Exact supported schema envelope ABI.
/// @param expected_transport_profile_tag Exact attached carriage profile tag.
/// @return Empty success or a protocol error; no payload bytes are exposed on failure.
[[nodiscard]] inline expected<void, Error> validate_received_schema_envelope(
    const SchemaEnvelopeHeader& header, const SchemaIdentity& expected_identity,
    std::uint64_t expected_generation, std::size_t received_payload_size,
    std::uint32_t expected_abi_version = kSchemaEnvelopeABIVersion,
    std::uint32_t expected_transport_profile_tag = 0U) {
  if (expected_generation == 0U) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "expected schema endpoint generation is zero"});
  }
  if (header.abi_version != expected_abi_version ||
      header.header_size != kSchemaEnvelopeHeaderSize) {
    return make_unexpected(Error{ErrorCode::kProtocolError, "schema envelope ABI mismatch"});
  }
  if (header.schema_family_id != expected_identity.family_id ||
      header.transport_profile_tag != expected_transport_profile_tag) {
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
  auto envelope = validate_received_schema_envelope(header,
                                                    expected_identity,
                                                    expected_generation,
                                                    payload.size(),
                                                    expected_abi_version,
                                                    package.transport_profile_tag());
  if (!envelope) {
    return make_unexpected(std::move(envelope).error());
  }
  if (!schema_identity_equal(package.identity(), expected_identity)) {
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
  auto envelope = validate_received_schema_envelope(header,
                                                    expected_identity,
                                                    expected_generation,
                                                    payload.size(),
                                                    expected_abi_version,
                                                    package.transport_profile_tag());
  if (!envelope) {
    return make_unexpected(std::move(envelope).error());
  }
  if (!schema_identity_equal(package.identity(), expected_identity)) {
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
