/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/channel/descriptor_endpoints.hpp>
#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/channel/endpoint_concepts.hpp>
#include <holoscan/channel/endpoint_interfaces.hpp>
#include <holoscan/channel/inline_endpoints.hpp>
#include <holoscan/channel/manual_tensor_publisher.hpp>
#include <holoscan/channel/tensor_endpoints.hpp>
#include <holoscan/channel/tensor_schema_endpoints.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/memory/tensor_storage.hpp>

namespace holoscan {
namespace channel_internal {
struct InlinePayloadTestAccess;
struct PublisherPressureOptions;
struct TensorSchemaStagingOptions;
class CatalogChannelBindingAccess;

/// @brief Build the complete cold payload identity for catalog attachment.
/// @tparam T Exact portable catalog scalar, raw-POD, or owning generated-schema type.
/// @return Deep-owned identity used for exact catalog comparison.
/// @throws std::bad_alloc if copying identity or field names cannot allocate.
template <CatalogChannelPayload T>
[[nodiscard]] PayloadIdentityRecord catalog_payload_identity() {
  PayloadIdentityRecord result{
      .kind = PayloadIdentityKind::kScalar,
      .payload_kind = PayloadKind::kInlineTyped,
      .scalar = {},
      .schema = {},
      .pod = {},
  };
  if constexpr (CatalogChannelSchema<T>) {
    const SchemaIdentity identity = schema_identity<T>();
    result.kind = PayloadIdentityKind::kSchema;
    result.payload_kind = PayloadKind::kSerializedBytes;
    result.schema = make_schema_identity_record(identity);
  } else if constexpr (CatalogChannelPod<T>) {
    result.kind = PayloadIdentityKind::kPod;
    PodIdentityRecord pod{
        .canonical_name = std::string(ChannelTypeIdentity<T>::name),
        .version = ChannelTypeIdentity<T>::version,
        .object_size = static_cast<std::uint32_t>(sizeof(T)),
        .object_alignment = static_cast<std::uint32_t>(alignof(T)),
        .fields = {},
    };
    std::apply(
        [&pod](const auto&... field) {
          (pod.fields.push_back(PodFieldIdentityRecord{
               .name = std::string(field.name),
               .byte_offset = field.byte_offset,
               .scalar_kind = field.scalar_kind,
               .element_count = field.element_count,
           }),
           ...);
        },
        ChannelPodLayout<T>::fields);
    result.pod = std::move(pod);
  } else if constexpr (std::same_as<T, bool>) {
    result.scalar = ScalarIdentityRecord{.kind = ScalarIdentityKind::kBoolean};
  } else if constexpr (std::same_as<T, std::uint8_t>) {
    result.scalar = ScalarIdentityRecord{.kind = ScalarIdentityKind::kUInt8};
  } else if constexpr (std::same_as<T, std::int8_t>) {
    result.scalar = ScalarIdentityRecord{.kind = ScalarIdentityKind::kInt8};
  } else if constexpr (std::same_as<T, std::uint16_t>) {
    result.scalar = ScalarIdentityRecord{.kind = ScalarIdentityKind::kUInt16};
  } else if constexpr (std::same_as<T, std::int16_t>) {
    result.scalar = ScalarIdentityRecord{.kind = ScalarIdentityKind::kInt16};
  } else if constexpr (std::same_as<T, std::uint32_t>) {
    result.scalar = ScalarIdentityRecord{.kind = ScalarIdentityKind::kUInt32};
  } else if constexpr (std::same_as<T, std::int32_t>) {
    result.scalar = ScalarIdentityRecord{.kind = ScalarIdentityKind::kInt32};
  } else if constexpr (std::same_as<T, std::uint64_t>) {
    result.scalar = ScalarIdentityRecord{.kind = ScalarIdentityKind::kUInt64};
  } else if constexpr (std::same_as<T, std::int64_t>) {
    result.scalar = ScalarIdentityRecord{.kind = ScalarIdentityKind::kInt64};
  } else if constexpr (std::same_as<T, float>) {
    result.scalar = ScalarIdentityRecord{.kind = ScalarIdentityKind::kFloat32};
  } else {
    result.scalar = ScalarIdentityRecord{.kind = ScalarIdentityKind::kFloat64};
  }
  return result;
}

/// @brief Validate one catalog schema endpoint against its installed native package.
/// @tparam T Owning generated-schema type.
/// @param endpoint Catalog endpoint view valid for this call.
/// @return Empty success or a typed absent, identity, artifact, or package-evidence error.
template <CatalogChannelSchema T>
[[nodiscard]] expected<void, Error> validate_catalog_schema_attachment(
    ChannelEndpointCatalog::EndpointView endpoint) {
  const auto catalog_identity = endpoint.schema_identity();
  if (!catalog_identity) {
    return make_unexpected(
        Error{ErrorCode::kIncompatible, "catalog endpoint is not generated-schema backed"});
  }
  SchemaIdentity expected{
      .canonical_name = catalog_identity->canonical_name(),
      .compatibility_epoch = catalog_identity->compatibility_epoch(),
      .family_id = catalog_identity->family_id(),
      .file_identifier = catalog_identity->file_identifier(),
  };
  const auto& package = schema_package<T>();
  if (package.max_serialized_size() != generated_schema_max_serialized_size_v<T>) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "generated schema package capacity evidence mismatch"});
  }
  const auto catalog_capacity = endpoint.schema_writer_max_serialized_bytes();
  if (!catalog_capacity) {
    return make_unexpected(
        Error{ErrorCode::kIncompatible, "catalog schema serialized capacity is absent"});
  }
  return validate_schema_package_for_attach(
      expected, package, endpoint.schema_transport_profile_tag().value_or(0U));
}
}  // namespace channel_internal
namespace detail {
class RuntimeChannelBindingAccess;
}  // namespace detail
/// @brief Own the private transport context and endpoint-scoped Channel resources.
///
/// Create the minimal common context with \ref create, then attach endpoints whose immutable
/// catalog or explicit compatibility contract supplies their finite resource requirements. The
/// runtime discovers backend capabilities privately; applications do not configure backend
/// services or branch on a public capability bag.
///
/// Endpoint factories are cold-path operations and may allocate bounded backend state or wait up
/// to an admitted attachment timeout. Concurrent mutation through one ChannelRuntime is not part
/// of the public contract; coordinate endpoint construction before steady-state use. Publisher and
/// subscriber operation contracts are documented by their respective facade types.
///
/// @warning A ChannelRuntime must outlive every ordinary publisher, subscriber, descriptor
/// endpoint, and Tensor-schema endpoint created from it. Before destroying the runtime, stop and
/// join those endpoint users and destroy the endpoints. A ManualTensorPublisher retains the runtime
/// authority consumed by its factory and is the explicit exception.
/// @warning The runtime must also outlive every descriptor-backed Tensor received through its
/// endpoints, all derived DLPack or language-binding views, and any asynchronous GPU work that
/// accesses those values.
class ChannelRuntime {
 public:
  HOLOSCAN_CORE_EXPORT ChannelRuntime(ChannelRuntime&& other) noexcept;
  HOLOSCAN_CORE_EXPORT ChannelRuntime& operator=(ChannelRuntime&& other) noexcept;
  HOLOSCAN_CORE_EXPORT ~ChannelRuntime();

  ChannelRuntime(const ChannelRuntime&) = delete;
  ChannelRuntime& operator=(const ChannelRuntime&) = delete;

  /// @brief Create the minimal runtime-owned Channel transport context.
  ///
  /// Construction performs only common-context setup and private capability discovery. It does
  /// not initialize optional descriptor/FD/VMM export infrastructure. An endpoint whose accepted
  /// contract requires an unavailable optional facility fails with a checked error; applications
  /// cannot enable backend facilities through a public runtime option bag.
  ///
  /// @return A movable runtime, or a checked error when the common transport context cannot be
  /// created.
  /// @note This cold-path operation may allocate and start common backend control state. It does
  /// not wait for an application publisher or subscriber peer.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT static expected<ChannelRuntime, Error> create();

  /** @brief Create a descriptor publisher.
   * @param channel_name Logical channel name.
   * @param options Channel queue and sidecar options.
   * @return Publisher or Error for invalid options or transport setup failure. */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ChannelDescriptorPublisher, Error>
  create_descriptor_publisher(std::string_view channel_name, const ChannelOptions& options = {});

  /** @brief Create a descriptor subscriber.
   * @param channel_name Logical channel name.
   * @param options Channel queue and sidecar options.
   * @return Subscriber or Error for invalid options or transport setup failure. */
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ChannelDescriptorSubscriber, Error>
  create_descriptor_subscriber(std::string_view channel_name, const ChannelOptions& options = {});

  /** @brief Create a typed inline publisher.
   * @tparam T Fixed-size inline payload type.
   * @param channel_name Logical channel name.
   * @param options Channel queue and sidecar options.
   * @return Publisher or Error for invalid options, unsupported transport, or setup failure. */
  template <typename T>
    requires ChannelInlinePayload<T>
  [[nodiscard]] expected<ChannelPublisher<T>, Error> create_publisher(
      std::string_view channel_name, const ChannelOptions& options = {});

  /// @brief Attach a standalone publisher to one exact catalog import endpoint.
  /// @tparam T Portable scalar, complete raw-POD, or owning generated-schema payload type.
  /// @param catalog Immutable catalog that owns the named endpoint.
  /// @param endpoint_name Exact stable endpoint name.
  /// @param options Caller-local timeout that may only shorten the catalog bound.
  /// @return Publisher or a checked lookup, role, identity, contract, timeout, or setup error.
  /// @retval ErrorCode::kTimeout The endpoint requires a peer and none became observable before the
  /// effective catalog/caller bound.
  /// @note The catalog is inspected only during this call. The returned publisher retains its
  /// private runtime attachment, not a reference to \p catalog. An optional peer never waits.
  template <CatalogChannelPayload T>
  [[nodiscard]] expected<ChannelPublisher<T>, Error> create_publisher(
      const ChannelEndpointCatalog& catalog, std::string_view endpoint_name,
      ChannelAttachOptions options = {});

  /// @brief Attach a standalone publisher through an immutable endpoint view.
  /// @tparam T Portable scalar, complete raw-POD, or owning generated-schema payload type.
  /// @param endpoint Catalog-owned import endpoint view valid for this call.
  /// @param options Caller-local timeout that may only shorten the catalog bound.
  /// @return Publisher or a checked role, identity, contract, timeout, or setup error.
  /// @retval ErrorCode::kTimeout The endpoint requires a peer and none became observable before the
  /// effective catalog/caller bound.
  /// @note The endpoint view need remain valid only for this call. The returned publisher retains
  /// its private runtime attachment, not the view. An optional peer never waits.
  template <CatalogChannelPayload T>
  [[nodiscard]] expected<ChannelPublisher<T>, Error> create_publisher(
      ChannelEndpointCatalog::EndpointView endpoint, ChannelAttachOptions options = {});

  /// @brief Attach a Tensor-loan publisher to one exact catalog import endpoint.
  /// @tparam T Exactly \ref Tensor; this overload does not widen \ref CatalogChannelPayload.
  /// @param catalog Immutable catalog that owns the named endpoint.
  /// @param endpoint_name Exact stable endpoint name.
  /// @param options Caller-local timeout that may only shorten the catalog bound.
  /// @return Endpoint-owned finite-pool publisher or a checked lookup, role, logical-layout,
  /// capability, timeout, or setup error.
  /// @note Catalog contracts admit ordinary host storage and same-device CUDA storage. Pinned,
  /// managed-destination, and direct graph-output zero-copy providers remain separately admitted.
  template <typename T>
    requires std::same_as<T, Tensor>
  [[nodiscard]] expected<ChannelPublisher<Tensor>, Error> create_publisher(
      const ChannelEndpointCatalog& catalog, std::string_view endpoint_name,
      ChannelAttachOptions options = {});

  /// @brief Attach a Tensor-loan publisher through an immutable endpoint view.
  /// @tparam T Exactly \ref Tensor; this overload remains distinct from inline payload factories.
  /// @param endpoint Catalog-owned import endpoint view valid for this call.
  /// @param options Caller-local timeout that may only shorten the catalog bound.
  /// @return Endpoint-owned finite-pool publisher or a checked role, logical-layout, capability,
  /// timeout, or setup error.
  template <typename T>
    requires std::same_as<T, Tensor>
  [[nodiscard]] expected<ChannelPublisher<Tensor>, Error> create_publisher(
      ChannelEndpointCatalog::EndpointView endpoint, ChannelAttachOptions options = {});

  /** @brief Create a typed subscriber.
   * @tparam T Inline payload type or Tensor.
   * @param channel_name Logical channel name.
   * @param options Channel queue and sidecar options.
   * @return Subscriber or Error for invalid options, unsupported transport, or setup failure. */
  template <typename T>
    requires(ChannelInlinePayload<T> || std::is_same_v<T, Tensor>)
  [[nodiscard]] expected<ChannelSubscriber<T>, Error> create_subscriber(
      std::string_view channel_name, const ChannelOptions& options = {});

  /// @brief Attach a standalone subscriber to one exact catalog export endpoint.
  /// @tparam T Portable scalar, complete raw-POD, or owning generated-schema payload type.
  /// @param catalog Immutable catalog that owns the named endpoint.
  /// @param endpoint_name Exact stable endpoint name.
  /// @param options Caller-local timeout that may only shorten the catalog bound.
  /// @return Subscriber or a checked lookup, role, identity, contract, timeout, or setup error.
  /// @retval ErrorCode::kTimeout The endpoint requires a peer and none became observable before the
  /// effective catalog/caller bound.
  /// @note The catalog is inspected only during this call. The returned subscriber retains its
  /// private runtime attachment, not a reference to \p catalog. An optional peer never waits.
  template <CatalogChannelPayload T>
  [[nodiscard]] expected<ChannelSubscriber<T>, Error> create_subscriber(
      const ChannelEndpointCatalog& catalog, std::string_view endpoint_name,
      ChannelAttachOptions options = {});

  /// @brief Attach a standalone subscriber through an immutable endpoint view.
  /// @tparam T Portable scalar, complete raw-POD, or owning generated-schema payload type.
  /// @param endpoint Catalog-owned export endpoint view valid for this call.
  /// @param options Caller-local timeout that may only shorten the catalog bound.
  /// @return Subscriber or a checked role, identity, contract, timeout, or setup error.
  /// @retval ErrorCode::kTimeout The endpoint requires a peer and none became observable before the
  /// effective catalog/caller bound.
  /// @note The endpoint view need remain valid only for this call. The returned subscriber retains
  /// its private runtime attachment, not the view. An optional peer never waits.
  template <CatalogChannelPayload T>
  [[nodiscard]] expected<ChannelSubscriber<T>, Error> create_subscriber(
      ChannelEndpointCatalog::EndpointView endpoint, ChannelAttachOptions options = {});

  /// @brief Attach a read-only Tensor subscriber to one exact catalog export endpoint.
  /// @tparam T Exactly \ref Tensor; this overload remains distinct from inline payload factories.
  /// @param catalog Immutable catalog that owns the named endpoint.
  /// @param endpoint_name Exact stable endpoint name.
  /// @param options Caller-local timeout that may only shorten the catalog bound.
  /// @return BufferHandle-backed subscriber or a checked lookup, role, logical-layout,
  /// capability, timeout, or setup error.
  /// @note Received Tensors retain read-only imported storage until the returned value and all
  /// derived lifetime-preserving views are destroyed.
  template <typename T>
    requires std::same_as<T, Tensor>
  [[nodiscard]] expected<ChannelSubscriber<Tensor>, Error> create_subscriber(
      const ChannelEndpointCatalog& catalog, std::string_view endpoint_name,
      ChannelAttachOptions options = {});

  /// @brief Attach a read-only Tensor subscriber through an immutable endpoint view.
  /// @tparam T Exactly \ref Tensor; this overload remains distinct from inline payload factories.
  /// @param endpoint Catalog-owned export endpoint view valid for this call.
  /// @param options Caller-local timeout that may only shorten the catalog bound.
  /// @return BufferHandle-backed subscriber or a checked role, logical-layout, capability,
  /// timeout, or setup error.
  /// @note The endpoint view need remain valid only for this call. The returned subscriber retains
  /// its private runtime attachment and imported storage claims, not the view.
  template <typename T>
    requires std::same_as<T, Tensor>
  [[nodiscard]] expected<ChannelSubscriber<Tensor>, Error> create_subscriber(
      ChannelEndpointCatalog::EndpointView endpoint, ChannelAttachOptions options = {});

  /** @brief Create a descriptor-backed Tensor schema publisher.
   * @tparam Payload Descriptor-backed FlatBuffer payload type.
   * @param channel_name Logical channel name.
   * @param options Channel queue and sidecar options.
   * @return Publisher or Error for invalid options or sidecar setup failure. */
  template <ChannelDescriptorFlatBufferPayload Payload>
  [[nodiscard]] expected<ChannelTensorSchemaPublisher<Payload>, Error>
  create_tensor_schema_publisher(std::string_view channel_name, const ChannelOptions& options = {});

  /** @brief Attach a descriptor-backed Tensor schema publisher to one catalog import endpoint.
   * @tparam Payload Fixed descriptor envelope for the exact generated schema root.
   * @param catalog Immutable catalog that owns the named endpoint.
   * @param endpoint_name Exact stable endpoint name.
   * @param options Caller-local timeout that may only shorten the catalog bound.
   * @return Publisher or a checked role, schema, field-contract, timeout, or setup error. */
  template <ChannelDescriptorFlatBufferPayload Payload>
  [[nodiscard]] expected<ChannelTensorSchemaPublisher<Payload>, Error>
  create_tensor_schema_publisher(const ChannelEndpointCatalog& catalog,
                                 std::string_view endpoint_name, ChannelAttachOptions options = {});

  /** @brief Attach a descriptor-backed Tensor schema publisher through a catalog endpoint view.
   * @tparam Payload Fixed descriptor envelope for the exact generated schema root.
   * @param endpoint Catalog-owned import endpoint view valid for this call.
   * @param options Caller-local timeout that may only shorten the catalog bound.
   * @return Publisher or a checked role, schema, field-contract, timeout, or setup error. */
  template <ChannelDescriptorFlatBufferPayload Payload>
  [[nodiscard]] expected<ChannelTensorSchemaPublisher<Payload>, Error>
  create_tensor_schema_publisher(ChannelEndpointCatalog::EndpointView endpoint,
                                 ChannelAttachOptions options = {});

  /** @brief Create a descriptor-backed Tensor schema subscriber.
   * @tparam Payload Descriptor-backed FlatBuffer payload type.
   * @param channel_name Logical channel name.
   * @param options Channel queue and sidecar options.
   * @return Subscriber or Error for invalid options or sidecar setup failure. */
  template <ChannelDescriptorFlatBufferPayload Payload>
  [[nodiscard]] expected<ChannelTensorSchemaSubscriber<Payload>, Error>
  create_tensor_schema_subscriber(std::string_view channel_name,
                                  const ChannelOptions& options = {});

  /** @brief Attach a descriptor-backed Tensor schema subscriber to one catalog export endpoint.
   * @tparam Payload Fixed descriptor envelope for the exact generated schema root.
   * @param catalog Immutable catalog that owns the named endpoint.
   * @param endpoint_name Exact stable endpoint name.
   * @param options Caller-local timeout that may only shorten the catalog bound.
   * @return Subscriber or a checked role, schema, field-contract, timeout, or setup error. */
  template <ChannelDescriptorFlatBufferPayload Payload>
  [[nodiscard]] expected<ChannelTensorSchemaSubscriber<Payload>, Error>
  create_tensor_schema_subscriber(const ChannelEndpointCatalog& catalog,
                                  std::string_view endpoint_name,
                                  ChannelAttachOptions options = {});

  /** @brief Attach a descriptor-backed Tensor schema subscriber through a catalog endpoint view.
   * @tparam Payload Fixed descriptor envelope for the exact generated schema root.
   * @param endpoint Catalog-owned export endpoint view valid for this call.
   * @param options Caller-local timeout that may only shorten the catalog bound.
   * @return Subscriber or a checked role, schema, field-contract, timeout, or setup error. */
  template <ChannelDescriptorFlatBufferPayload Payload>
  [[nodiscard]] expected<ChannelTensorSchemaSubscriber<Payload>, Error>
  create_tensor_schema_subscriber(ChannelEndpointCatalog::EndpointView endpoint,
                                  ChannelAttachOptions options = {});

  /// Mint authority for one manual compatibility endpoint.
  ///
  /// @param options Manual service identity and endpoint role.
  /// @return A move-only binding, or an error when the request is invalid, the runtime is closed,
  /// or cold service-identity allocation is exhausted.
  /// @note The returned authority is not derived from an endpoint catalog and can be consumed only
  /// by the matching manual compatibility factory.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ManualChannelBinding, Error> bind_manual_channel(
      ManualChannelBindingOptions options);

 private:
  class Impl;

  HOLOSCAN_CORE_EXPORT explicit ChannelRuntime(std::unique_ptr<Impl> impl) noexcept;
  explicit ChannelRuntime(std::shared_ptr<Impl> impl) noexcept;

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ManualTensorPublisher, Error> create_tensor_publisher(
      std::string_view channel_name, const ChannelOptions& options);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ManualTensorPublisher, Error> create_tensor_publisher(
      std::string_view channel_name, const ChannelOptions& options,
      const channel_internal::PublisherPressureOptions& pressure);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ChannelSubscriber<Tensor>, Error>
  create_tensor_subscriber(
      std::string_view channel_name, const ChannelOptions& options,
      std::optional<TensorBoundaryContractRecord> receive_contract = std::nullopt,
      std::uint32_t receive_control_capacity = 0U,
      std::unique_ptr<channel_internal::InlinePayloadSubscriber> health_subscriber = nullptr,
      std::uint64_t catalog_clock_count = 0U);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ChannelPublisher<Tensor>, Error>
  create_catalog_tensor_publisher(ChannelEndpointCatalog::EndpointView endpoint,
                                  ChannelAttachOptions options,
                                  std::optional<std::uint32_t> publisher_loan_bound = std::nullopt,
                                  TopicDirection publisher_direction = TopicDirection::kImport,
                                  bool enforce_peer_requirement = true,
                                  std::optional<DeviceId> planned_device = std::nullopt);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ChannelPublisher<Tensor>, Error>
  create_catalog_tensor_publisher(ChannelEndpointCatalog::EndpointView endpoint,
                                  ChannelAttachOptions options,
                                  std::optional<std::uint32_t> publisher_loan_bound,
                                  TopicDirection publisher_direction, bool enforce_peer_requirement,
                                  std::optional<DeviceId> planned_device,
                                  const channel_internal::PublisherPressureOptions& pressure);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ChannelSubscriber<Tensor>, Error>
  create_catalog_tensor_subscriber(ChannelEndpointCatalog::EndpointView endpoint,
                                   ChannelAttachOptions options,
                                   TopicDirection subscriber_direction = TopicDirection::kExport,
                                   bool enforce_peer_requirement = true);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ChannelSubscriber<Tensor>, Error>
  create_catalog_tensor_subscriber(ChannelEndpointCatalog::EndpointView endpoint,
                                   ChannelAttachOptions options,
                                   TopicDirection subscriber_direction,
                                   bool enforce_peer_requirement,
                                   std::uint32_t receive_control_capacity);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT
      expected<std::unique_ptr<channel_internal::InlinePayloadPublisher>, Error>
      create_inline_payload_publisher(std::string_view channel_name, std::size_t payload_size,
                                      const ChannelOptions& options);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT
      expected<std::unique_ptr<channel_internal::InlinePayloadPublisher>, Error>
      create_inline_payload_publisher(std::string_view channel_name, std::size_t payload_size,
                                      const ChannelOptions& options,
                                      const channel_internal::PublisherPressureOptions& pressure);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT
      expected<std::unique_ptr<channel_internal::InlinePayloadSubscriber>, Error>
      create_inline_payload_subscriber(std::string_view channel_name, std::size_t payload_size,
                                       const ChannelOptions& options);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT
      expected<std::unique_ptr<channel_internal::InlinePayloadPublisher>, Error>
      create_catalog_inline_payload_publisher(ChannelEndpointCatalog::EndpointView endpoint,
                                              PayloadIdentityRecord payload_identity,
                                              std::size_t payload_size,
                                              ChannelAttachOptions options,
                                              TopicDirection publisher_direction,
                                              bool enforce_peer_requirement);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT
      expected<std::unique_ptr<channel_internal::InlinePayloadPublisher>, Error>
      create_catalog_inline_payload_publisher(
          ChannelEndpointCatalog::EndpointView endpoint, PayloadIdentityRecord payload_identity,
          std::size_t payload_size, ChannelAttachOptions options,
          TopicDirection publisher_direction, bool enforce_peer_requirement,
          const channel_internal::PublisherPressureOptions& pressure);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT
      expected<std::unique_ptr<channel_internal::InlinePayloadSubscriber>, Error>
      create_catalog_inline_payload_subscriber(ChannelEndpointCatalog::EndpointView endpoint,
                                               PayloadIdentityRecord payload_identity,
                                               std::size_t payload_size,
                                               ChannelAttachOptions options,
                                               TopicDirection subscriber_direction,
                                               bool enforce_peer_requirement);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT
      expected<std::unique_ptr<channel_internal::InlinePayloadWaitSet>, Error>
      create_inline_payload_wait_set();

  [[nodiscard]] HOLOSCAN_CORE_EXPORT
      expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
      create_tensor_schema_payload_publisher(std::string_view channel_name,
                                             std::size_t payload_size,
                                             std::size_t descriptor_offset,
                                             const ChannelOptions& options);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT
      expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
      create_tensor_schema_payload_publisher(
          std::string_view channel_name, std::size_t payload_size, std::size_t descriptor_offset,
          const ChannelOptions& options,
          const channel_internal::PublisherPressureOptions& pressure);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT
      expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
      create_tensor_schema_payload_publisher(
          std::string_view channel_name, std::size_t payload_size, std::size_t descriptor_offset,
          const ChannelOptions& options, const channel_internal::PublisherPressureOptions& pressure,
          const channel_internal::TensorSchemaStagingOptions& staging);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT
      expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
      create_tensor_schema_payload_publisher(std::string_view channel_name,
                                             std::size_t payload_size,
                                             std::span<const std::size_t> descriptor_offsets,
                                             const ChannelOptions& options);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT
      expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
      create_tensor_schema_payload_publisher(
          std::string_view channel_name, std::size_t payload_size,
          std::span<const std::size_t> descriptor_offsets, const ChannelOptions& options,
          const channel_internal::PublisherPressureOptions& pressure,
          std::span<const channel_internal::TensorSchemaStagingOptions> staging,
          cudaStream_t staging_stream = nullptr, ClockId capture_clock = {},
          std::unique_ptr<channel_internal::InlinePayloadPublisher> health_publisher = nullptr,
          bool catalog_control = false);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT
      expected<std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber>, Error>
      create_tensor_schema_payload_subscriber(
          std::string_view channel_name, std::size_t payload_size, std::uint32_t tensor_field_count,
          const ChannelOptions& options,
          std::unique_ptr<channel_internal::InlinePayloadSubscriber> health_subscriber = nullptr,
          bool catalog_control = false,
          std::span<const channel_internal::TensorSchemaStagingOptions> field_contracts = {},
          std::uint64_t catalog_clock_count = 0U);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT
      expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
      create_catalog_tensor_schema_payload_publisher(
          ChannelEndpointCatalog::EndpointView endpoint, SchemaIdentity expected_schema,
          std::size_t payload_size, std::span<const std::size_t> descriptor_offsets,
          ChannelAttachOptions options);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT
      expected<std::unique_ptr<channel_internal::TensorSchemaPayloadPublisher>, Error>
      create_catalog_tensor_schema_payload_publisher(
          ChannelEndpointCatalog::EndpointView endpoint, SchemaIdentity expected_schema,
          std::size_t payload_size, std::span<const std::size_t> descriptor_offsets,
          ChannelAttachOptions options,
          std::span<const channel_internal::TensorSchemaStagingOptions> staging,
          cudaStream_t staging_stream, const channel_internal::PublisherPressureOptions& pressure,
          TopicDirection publisher_direction = TopicDirection::kImport,
          bool enforce_peer_requirement = true);

  [[nodiscard]] HOLOSCAN_CORE_EXPORT
      expected<std::unique_ptr<channel_internal::TensorSchemaPayloadSubscriber>, Error>
      create_catalog_tensor_schema_payload_subscriber(
          ChannelEndpointCatalog::EndpointView endpoint, SchemaIdentity expected_schema,
          std::size_t payload_size, std::uint32_t tensor_field_count, ChannelAttachOptions options,
          TopicDirection subscriber_direction = TopicDirection::kExport,
          bool enforce_peer_requirement = true);

  std::shared_ptr<Impl> impl_;

  friend class ManualChannelBinding;
  friend class ManualTensorPublisher;
  friend expected<ManualTensorPublisher, Error> make_manual_tensor_publisher(
      ManualChannelBinding binding, ManualTensorPublisherOptions options);
  friend struct channel_internal::TensorPublisherTestAccess;
  friend struct channel_internal::InlinePayloadTestAccess;
  friend class channel_internal::CatalogChannelBindingAccess;
  friend class detail::RuntimeChannelBindingAccess;
};

template <typename T>
  requires ChannelInlinePayload<T>
expected<ChannelPublisher<T>, Error> ChannelRuntime::create_publisher(
    std::string_view channel_name, const ChannelOptions& options) {
  constexpr std::size_t kPayloadSize = sizeof(T);
  auto publisher = create_inline_payload_publisher(channel_name, kPayloadSize, options);
  if (!publisher.has_value()) {
    return make_unexpected(std::move(publisher).error());
  }
  return ChannelPublisher<T>(std::move(publisher).value());
}

template <CatalogChannelPayload T>
expected<ChannelPublisher<T>, Error> ChannelRuntime::create_publisher(
    const ChannelEndpointCatalog& catalog, std::string_view endpoint_name,
    ChannelAttachOptions options) {
  auto endpoint = catalog.endpoint(endpoint_name);
  if (!endpoint) {
    return make_unexpected(std::move(endpoint).error());
  }
  return create_publisher<T>(*endpoint, std::move(options));
}

template <typename T>
  requires std::same_as<T, Tensor>
expected<ChannelPublisher<Tensor>, Error> ChannelRuntime::create_publisher(
    const ChannelEndpointCatalog& catalog, std::string_view endpoint_name,
    ChannelAttachOptions options) {
  auto endpoint = catalog.endpoint(endpoint_name);
  if (!endpoint) {
    return make_unexpected(std::move(endpoint).error());
  }
  return create_catalog_tensor_publisher(*endpoint, std::move(options));
}

template <typename T>
  requires std::same_as<T, Tensor>
expected<ChannelPublisher<Tensor>, Error> ChannelRuntime::create_publisher(
    ChannelEndpointCatalog::EndpointView endpoint, ChannelAttachOptions options) {
  return create_catalog_tensor_publisher(endpoint, std::move(options));
}

template <CatalogChannelPayload T>
expected<ChannelPublisher<T>, Error> ChannelRuntime::create_publisher(
    ChannelEndpointCatalog::EndpointView endpoint, ChannelAttachOptions options) {
  if constexpr (CatalogChannelSchema<T>) {
    auto package = channel_internal::validate_catalog_schema_attachment<T>(endpoint);
    if (!package) {
      return make_unexpected(std::move(package).error());
    }
  }
  constexpr std::size_t kPayloadSize = [] {
    if constexpr (CatalogChannelSchema<T>) {
      return generated_schema_max_serialized_size_v<T>;
    }
    return sizeof(T);
  }();
  auto publisher =
      create_catalog_inline_payload_publisher(endpoint,
                                              channel_internal::catalog_payload_identity<T>(),
                                              kPayloadSize,
                                              std::move(options),
                                              TopicDirection::kImport,
                                              true);
  if (!publisher) {
    return make_unexpected(std::move(publisher).error());
  }
  return ChannelPublisher<T>(std::move(publisher).value());
}

template <CatalogChannelPayload T>
expected<ChannelSubscriber<T>, Error> ChannelRuntime::create_subscriber(
    const ChannelEndpointCatalog& catalog, std::string_view endpoint_name,
    ChannelAttachOptions options) {
  auto endpoint = catalog.endpoint(endpoint_name);
  if (!endpoint) {
    return make_unexpected(std::move(endpoint).error());
  }
  return create_subscriber<T>(*endpoint, std::move(options));
}

template <typename T>
  requires std::same_as<T, Tensor>
expected<ChannelSubscriber<Tensor>, Error> ChannelRuntime::create_subscriber(
    const ChannelEndpointCatalog& catalog, std::string_view endpoint_name,
    ChannelAttachOptions options) {
  auto endpoint = catalog.endpoint(endpoint_name);
  if (!endpoint) {
    return make_unexpected(std::move(endpoint).error());
  }
  return create_catalog_tensor_subscriber(*endpoint, std::move(options));
}

template <typename T>
  requires std::same_as<T, Tensor>
expected<ChannelSubscriber<Tensor>, Error> ChannelRuntime::create_subscriber(
    ChannelEndpointCatalog::EndpointView endpoint, ChannelAttachOptions options) {
  return create_catalog_tensor_subscriber(endpoint, std::move(options));
}

template <CatalogChannelPayload T>
expected<ChannelSubscriber<T>, Error> ChannelRuntime::create_subscriber(
    ChannelEndpointCatalog::EndpointView endpoint, ChannelAttachOptions options) {
  if constexpr (CatalogChannelSchema<T>) {
    auto package = channel_internal::validate_catalog_schema_attachment<T>(endpoint);
    if (!package) {
      return make_unexpected(std::move(package).error());
    }
  }
  constexpr std::size_t kPayloadSize = [] {
    if constexpr (CatalogChannelSchema<T>) {
      return generated_schema_max_serialized_size_v<T>;
    }
    return sizeof(T);
  }();
  auto subscriber =
      create_catalog_inline_payload_subscriber(endpoint,
                                               channel_internal::catalog_payload_identity<T>(),
                                               kPayloadSize,
                                               std::move(options),
                                               TopicDirection::kExport,
                                               true);
  if (!subscriber) {
    return make_unexpected(std::move(subscriber).error());
  }
  return ChannelSubscriber<T>(std::move(subscriber).value());
}

template <typename T>
  requires(ChannelInlinePayload<T> || std::is_same_v<T, Tensor>)
expected<ChannelSubscriber<T>, Error> ChannelRuntime::create_subscriber(
    std::string_view channel_name, const ChannelOptions& options) {
  if constexpr (std::is_same_v<T, Tensor>) {
    return create_tensor_subscriber(channel_name, options);
  } else {
    constexpr std::size_t kPayloadSize = sizeof(T);
    auto subscriber = create_inline_payload_subscriber(channel_name, kPayloadSize, options);
    if (!subscriber.has_value()) {
      return make_unexpected(std::move(subscriber).error());
    }
    return ChannelSubscriber<T>(std::move(subscriber).value());
  }
}

template <ChannelDescriptorFlatBufferPayload Payload>
expected<ChannelTensorSchemaPublisher<Payload>, Error>
ChannelRuntime::create_tensor_schema_publisher(std::string_view channel_name,
                                               const ChannelOptions& options) {
  constexpr std::size_t kPayloadSize = sizeof(Payload);
  constexpr std::array<std::size_t, Payload::max_tensors> kDescriptorOffsets = [] {
    std::array<std::size_t, Payload::max_tensors> result{};
    constexpr std::size_t kFirstDescriptorOffset =
        offsetof(Payload, descriptor) +
        offsetof(typename Payload::descriptor_carriage_type, descriptors);
    for (std::size_t index = 0U; index < result.size(); ++index) {
      result[index] = kFirstDescriptorOffset + index * sizeof(PayloadDescriptor);
    }
    return result;
  }();
  auto publisher = create_tensor_schema_payload_publisher(
      channel_name, kPayloadSize, kDescriptorOffsets, options);
  if (!publisher) {
    return make_unexpected(std::move(publisher).error());
  }
  return ChannelTensorSchemaPublisher<Payload>(std::move(publisher).value());
}

template <ChannelDescriptorFlatBufferPayload Payload>
expected<ChannelTensorSchemaPublisher<Payload>, Error>
ChannelRuntime::create_tensor_schema_publisher(const ChannelEndpointCatalog& catalog,
                                               std::string_view endpoint_name,
                                               ChannelAttachOptions options) {
  auto endpoint = catalog.endpoint(endpoint_name);
  if (!endpoint) {
    return make_unexpected(std::move(endpoint).error());
  }
  return create_tensor_schema_publisher<Payload>(*endpoint, std::move(options));
}

template <ChannelDescriptorFlatBufferPayload Payload>
expected<ChannelTensorSchemaPublisher<Payload>, Error>
ChannelRuntime::create_tensor_schema_publisher(ChannelEndpointCatalog::EndpointView endpoint,
                                               ChannelAttachOptions options) {
  using Envelope = channel_internal::TensorSchemaRuntimeEnvelope<Payload>;
  constexpr std::array<std::size_t, Payload::max_tensors> kDescriptorOffsets = [] {
    std::array<std::size_t, Payload::max_tensors> result{};
    constexpr std::size_t kFirstDescriptorOffset =
        offsetof(Envelope, payload) + offsetof(Payload, descriptor) +
        offsetof(typename Payload::descriptor_carriage_type, descriptors);
    for (std::size_t index = 0U; index < result.size(); ++index) {
      result[index] = kFirstDescriptorOffset + index * sizeof(PayloadDescriptor);
    }
    return result;
  }();
  auto publisher = create_catalog_tensor_schema_payload_publisher(endpoint,
                                                                  Payload::expected_schema_identity,
                                                                  sizeof(Envelope),
                                                                  kDescriptorOffsets,
                                                                  std::move(options));
  if (!publisher) {
    return make_unexpected(std::move(publisher).error());
  }
  return ChannelTensorSchemaPublisher<Payload>(std::move(*publisher), true);
}

template <ChannelDescriptorFlatBufferPayload Payload>
expected<ChannelTensorSchemaSubscriber<Payload>, Error>
ChannelRuntime::create_tensor_schema_subscriber(std::string_view channel_name,
                                                const ChannelOptions& options) {
  auto subscriber = create_tensor_schema_payload_subscriber(
      channel_name, sizeof(Payload), Payload::max_tensors, options);
  if (!subscriber) {
    return make_unexpected(std::move(subscriber).error());
  }
  return ChannelTensorSchemaSubscriber<Payload>(std::move(subscriber).value());
}

template <ChannelDescriptorFlatBufferPayload Payload>
expected<ChannelTensorSchemaSubscriber<Payload>, Error>
ChannelRuntime::create_tensor_schema_subscriber(const ChannelEndpointCatalog& catalog,
                                                std::string_view endpoint_name,
                                                ChannelAttachOptions options) {
  auto endpoint = catalog.endpoint(endpoint_name);
  if (!endpoint) {
    return make_unexpected(std::move(endpoint).error());
  }
  return create_tensor_schema_subscriber<Payload>(*endpoint, std::move(options));
}

template <ChannelDescriptorFlatBufferPayload Payload>
expected<ChannelTensorSchemaSubscriber<Payload>, Error>
ChannelRuntime::create_tensor_schema_subscriber(ChannelEndpointCatalog::EndpointView endpoint,
                                                ChannelAttachOptions options) {
  using Envelope = channel_internal::TensorSchemaRuntimeEnvelope<Payload>;
  auto subscriber =
      create_catalog_tensor_schema_payload_subscriber(endpoint,
                                                      Payload::expected_schema_identity,
                                                      sizeof(Envelope),
                                                      Payload::max_tensors,
                                                      std::move(options));
  if (!subscriber) {
    return make_unexpected(std::move(subscriber).error());
  }
  return ChannelTensorSchemaSubscriber<Payload>(std::move(*subscriber), true);
}

}  // namespace holoscan
