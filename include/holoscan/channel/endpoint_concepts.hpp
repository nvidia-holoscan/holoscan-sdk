/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/schema/schema_package.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace holoscan {
class ChannelRuntime;

/// @brief Check whether a type can use the manual host-object inline compatibility path.
/// @tparam T Type evaluated for standard layout, trivial copyability, and default construction.
/// @return True for an ordinary fixed-size host object with no owning generated-schema package.
/// @note A schema identity alone does not select the generated-schema transport path. Generated
/// native values with an admitted package and \ref Tensor use dedicated transport paths.
template <typename T>
concept ChannelInlinePayload =
    std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T> &&
    std::is_nothrow_default_constructible_v<T> && !std::is_same_v<T, Tensor> &&
    !detail::HasGeneratedSchemaNativeTraits<T>;

/// @brief Check whether a type is one of the portable catalog scalar identities.
/// @tparam T Exact scalar payload type.
/// @return True for the closed fixed-width catalog scalar set.
/// @note Local enums, platform-dependent arithmetic aliases, raw POD, generated schemas, and
/// Tensor are intentionally excluded from the scalar family.
template <typename T>
concept CatalogChannelScalar =
    std::same_as<T, bool> || std::same_as<T, std::uint8_t> || std::same_as<T, std::int8_t> ||
    std::same_as<T, std::uint16_t> || std::same_as<T, std::int16_t> ||
    std::same_as<T, std::uint32_t> || std::same_as<T, std::int32_t> ||
    std::same_as<T, std::uint64_t> || std::same_as<T, std::int64_t> ||
    (std::same_as<T, float> && std::numeric_limits<float>::is_iec559 && sizeof(float) == 4U) ||
    (std::same_as<T, double> && std::numeric_limits<double>::is_iec559 && sizeof(double) == 8U);

/// @brief Check whether a type has one complete portable raw-POD catalog contract.
/// @tparam T Exact unqualified payload type.
/// @return True when the explicit identity and layout specializations describe every object byte.
/// @note Admission requires a non-scalar, standard-layout, trivially-copyable,
/// copy-constructible type and the complete \ref ChannelTypeIdentity and
/// \ref ChannelPodLayout contract. Default construction and nothrow default construction are not
/// required.
/// @warning This is an experimental expert API. Prefer a generated FlatBuffers schema for normal
/// application-defined messages; raw-POD users own identity versioning, coordinated deployment,
/// and binary compatibility.
template <typename T>
concept CatalogChannelPod = std::same_as<T, std::remove_cvref_t<T>> && !CatalogChannelScalar<T> &&
                            std::copy_constructible<T> && detail::HasChannelPodContract<T> &&
                            detail::complete_channel_pod_contract<T>();

/// @brief Check whether a fixed-size value can attach through the catalog Channel facade.
/// @tparam T Exact unqualified payload type.
/// @return True for a portable catalog scalar or a complete raw-POD catalog contract.
/// @note Generated schema values and \ref Tensor use their dedicated catalog facade paths.
template <typename T>
concept CatalogChannelValue = CatalogChannelScalar<T> || CatalogChannelPod<T>;

/// @brief Check whether a type has an owning generated-schema catalog package.
/// @tparam T Exact unqualified semantic generated native type.
/// @return True when the generated package supplies explicit identity, owning copy semantics, and
/// a positive finite serialized capacity.
/// @note Generated table views, byte wrappers, arbitrary copyable classes, and the Core
/// \ref Tensor value are excluded. An owning generated type that contains a Core Tensor, such as
/// the current SDK `schema::ImageT`, is likewise excluded until EA1-014 supplies admitted Tensor
/// allocation and retirement bounds. Generated packages specialize
/// \ref GeneratedSchemaNativeTraits only for their owning application-facing native type.
/// The package specialization must be visible where this concept is evaluated. Owning values must
/// also be no-throw movable because Runtime materializes them behind checked, non-throwing type-
/// erased callbacks; serialization and allocation failures remain explicit \ref Error values.
template <typename T>
concept CatalogChannelSchema =
    std::same_as<T, std::remove_cvref_t<T>> && !std::same_as<T, Tensor> &&
    std::copy_constructible<T> && detail::HasSchemaIdentity<T> &&
    std::is_nothrow_move_constructible_v<T> && detail::HasGeneratedSchemaNativeTraits<T> &&
    detail::HasSchemaPackage<T>;

/// @brief Check whether a type can attach through the ordinary catalog Channel facade.
/// @tparam T Exact unqualified payload type.
/// @return True for portable scalar, raw-POD, or owning generated-schema payloads.
template <typename T>
concept CatalogChannelPayload = CatalogChannelValue<T> || CatalogChannelSchema<T>;

template <typename T>
class ChannelPublisher;

template <typename T>
class ChannelSubscriber;

namespace channel_internal {

struct TensorPublisherTestAccess;

/// Fixed generation of the immutable manual schema-compatibility endpoint generation.
/// Catalog-backed generated-schema endpoints carry their independently bound generation instead.
inline constexpr std::uint64_t kCompatibilitySchemaEndpointGeneration = 1U;

template <typename T>
struct IsInlineFlatBufferPayload : std::false_type {};

template <typename Root, std::size_t Capacity>
struct IsInlineFlatBufferPayload<schema::InlineFlatBufferPayload<Root, Capacity>> : std::true_type {
};

template <typename T>
struct IsDescriptorFlatBufferPayload : std::false_type {};

template <typename Root, std::size_t MetadataCapacity, std::size_t MaxTensors>
struct IsDescriptorFlatBufferPayload<
    schema::DescriptorFlatBufferPayload<Root, MetadataCapacity, MaxTensors>> : std::true_type {};

template <typename T>
inline constexpr bool is_descriptor_flatbuffer_payload_v =
    IsDescriptorFlatBufferPayload<std::remove_cvref_t<T>>::value;

template <typename T>
inline constexpr bool is_schema_envelope_payload_v =
    IsInlineFlatBufferPayload<std::remove_cvref_t<T>>::value ||
    IsDescriptorFlatBufferPayload<std::remove_cvref_t<T>>::value;

}  // namespace channel_internal

/** @brief Check whether a type is a descriptor-backed FlatBuffer payload wrapper.
 * @tparam T Type evaluated after cv/ref removal.
 * @return True when T wraps schema::DescriptorFlatBufferPayload. */
template <typename T>
concept ChannelDescriptorFlatBufferPayload =
    channel_internal::is_descriptor_flatbuffer_payload_v<T>;

/// @brief Check whether a manual compatibility payload carries a generated-schema envelope.
/// @tparam T Type evaluated after cv/ref removal.
/// @return True for the fixed-capacity inline or descriptor-backed FlatBuffer wrappers.
template <typename T>
concept ChannelSchemaEnvelopePayload = channel_internal::is_schema_envelope_payload_v<T>;
}  // namespace holoscan
