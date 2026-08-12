/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/sample.hpp>

namespace holoscan {

class ChannelRuntime;
class ManualChannelBinding;
class ManualTensorPublisher;
struct ManualTensorPublisherOptions;

/// @brief Create one sidecar-only manual Tensor publisher from runtime-minted authority.
/// @param binding Move-only authority consumed by this call on success or failure.
/// @param options Required positive finite publisher-local compatibility bound.
/// @return A publisher or a checked setup error.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ManualTensorPublisher, Error>
make_manual_tensor_publisher(ManualChannelBinding binding, ManualTensorPublisherOptions options);

namespace channel_internal {
class TensorPublishLoanAccess;
struct TensorPublisherTestAccess;
}  // namespace channel_internal

/// @brief Identify the endpoint role authorized by a manual channel binding.
enum class ManualChannelRole : std::uint8_t {
  /// Authorize creation of one manual Tensor publisher.
  kPublisher,
};

/// @brief Describe the narrow authority that a runtime should mint for a manual channel endpoint.
///
/// Both fields are cold setup values borrowed only for the call to
/// `ChannelRuntime::bind_manual_channel`. The service name is copied into
/// runtime-owned storage before the call returns.
struct ManualChannelBindingOptions {
  /// Identify the transport service without conferring catalog authority.
  std::string_view service_name;
  /// Select the manual endpoint role. EA1 supports only \ref ManualChannelRole::kPublisher.
  ManualChannelRole role;
};

/// @brief Hold opaque runtime, service-identity, and endpoint-role authority for a manual endpoint.
///
/// The binding is minted only by \ref ChannelRuntime::bind_manual_channel and is consumed by a
/// matching manual endpoint factory. It is not catalog authority and cannot be copied, assigned,
/// or reconstructed from a service name.
class ManualChannelBinding {
 public:
  /// @brief Move authority from another binding.
  /// @param other Binding whose authority transfers to this object.
  /// @post `other` is empty and safe to destroy.
  HOLOSCAN_CORE_EXPORT ManualChannelBinding(ManualChannelBinding&& other) noexcept;

  /// @brief Reject move assignment so one binding has one stable authority owner.
  ManualChannelBinding& operator=(ManualChannelBinding&& other) = delete;

  /// @brief Reject copying of endpoint-creation authority.
  ManualChannelBinding(const ManualChannelBinding&) = delete;

  /// @brief Reject copy assignment of endpoint-creation authority.
  ManualChannelBinding& operator=(const ManualChannelBinding&) = delete;

  /// @brief Release unconsumed manual endpoint authority.
  HOLOSCAN_CORE_EXPORT ~ManualChannelBinding() noexcept;

 private:
  class Impl;

  explicit ManualChannelBinding(std::unique_ptr<Impl> impl) noexcept;

  std::unique_ptr<Impl> impl_;

  friend class ChannelRuntime;
  friend class ManualTensorPublisher;
  friend expected<ManualTensorPublisher, Error> make_manual_tensor_publisher(
      ManualChannelBinding binding, ManualTensorPublisherOptions options);
};

/// @brief Configure one publisher-local, finite manual Tensor compatibility resource domain.
///
/// The bound is setup-time metadata. Creating the publisher rejects zero and
/// `UINT32_MAX`; successful creation reserves no unbounded fallback path.
struct ManualTensorPublisherOptions {
  /// Bound outstanding compatibility records to \c 1..UINT32_MAX-1.
  std::uint32_t max_inflight_samples;
};

/// @brief Publish Tensor values through a same-runtime compatibility endpoint.
///
/// This facade owns the consumed \ref ManualChannelBinding and retains its runtime and service
/// identity. It has no catalog capture-clock authority, allocation-loan API, interprocess
/// descriptor-pool plan, or generic manual POD/schema implication. This provisional EA1 migration
/// admits exactly one same-runtime subscriber; creating another subscriber for the service fails
/// with \ref ErrorCode::kResourceExhausted. The class is not thread-safe; callers must serialize
/// operations on one facade.
class ManualTensorPublisher {
 public:
  /// @brief Move the publisher and retained binding from another object.
  /// @param other Publisher whose endpoint ownership transfers to this object.
  /// @post `other` is empty and safe to destroy.
  HOLOSCAN_CORE_EXPORT ManualTensorPublisher(ManualTensorPublisher&& other) noexcept;

  /// @brief Reject move assignment so endpoint ownership remains stable.
  ManualTensorPublisher& operator=(ManualTensorPublisher&& other) = delete;

  /// @brief Reject copying of publisher and binding authority.
  ManualTensorPublisher(const ManualTensorPublisher&) = delete;

  /// @brief Reject copy assignment of publisher and binding authority.
  ManualTensorPublisher& operator=(const ManualTensorPublisher&) = delete;

  /// @brief Destroy the endpoint after releasing its backend publisher state.
  HOLOSCAN_CORE_EXPORT ~ManualTensorPublisher() noexcept;

  /// @brief Emit a Tensor through this manual same-runtime sidecar endpoint.
  ///
  /// @param tensor Tensor whose ownership is consumed on every outcome. A retryable publication
  /// error preserves logical publication authority but requires a newly supplied equivalent
  /// Tensor; it does not restore this by-value argument.
  /// @param options Optional publication metadata overrides. A capture time is rejected because
  /// this compatibility facade has no catalog capture-clock authority.
  /// @return The number of subscribers reached, or a checked publication error.
  /// @retval ErrorCode::kResourceExhausted Release-callback registration allocation failed before
  /// transport accepted the sidecar descriptor; the inserted record was released.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::uint32_t, Error> emit(Tensor tensor,
                                                                         EmitOptions options = {});

  /// @brief Emit a Tensor sample while preserving admitted public metadata.
  ///
  /// @param sample Tensor and metadata consumed on every outcome. A retryable publication error
  /// preserves logical publication authority but requires a newly supplied equivalent sample; it
  /// does not restore this by-value argument. A valid capture time is rejected because this
  /// compatibility facade has no catalog capture-clock authority.
  /// @return The number of subscribers reached, or a checked publication error.
  /// @retval ErrorCode::kResourceExhausted Release-callback registration allocation failed before
  /// transport accepted the sidecar descriptor; the inserted record was released.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<std::uint32_t, Error> emit_sample(
      Sample<Tensor> sample);

  /// @brief Refresh backend connection state.
  /// @return Empty success, or an error when the endpoint is closed or refresh fails.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT expected<void, Error> update_connections();

  /// @brief Return whether at least one subscriber is currently connected.
  /// @return True when connected; false when closed or disconnected.
  [[nodiscard]] HOLOSCAN_CORE_EXPORT bool has_subscribers() const noexcept;

 private:
  class Impl;

  explicit ManualTensorPublisher(std::unique_ptr<Impl> impl) noexcept;

  [[nodiscard]] std::uint64_t producer_generation() const noexcept;
  [[nodiscard]] expected<std::uint32_t, Error> emit_end_of_stream(SampleMetadata metadata) noexcept;

  std::unique_ptr<Impl> impl_;

  friend class ChannelRuntime;
  friend expected<ManualTensorPublisher, Error> make_manual_tensor_publisher(
      ManualChannelBinding binding, ManualTensorPublisherOptions options);
  friend struct channel_internal::TensorPublisherTestAccess;
  friend class channel_internal::TensorPublishLoanAccess;
};

/// @brief Create one sidecar-only manual Tensor publisher from runtime-minted authority.
///
/// @param binding Move-only authority consumed by this call on success or failure.
/// @param options Required positive finite publisher-local compatibility bound.
/// @return A publisher retaining the binding, \ref ErrorCode::kInvalidArgument for an empty
/// service, unsupported role, zero bound, or reserved \c UINT32_MAX bound,
/// \ref ErrorCode::kResourceExhausted for a cold allocation or allocator-length failure, or a
/// duplicate-service error. Another backend setup exception maps to \ref ErrorCode::kFailure.
/// Every failure leaves existing endpoints unchanged, and a cold setup failure leaves no service
/// tombstone.
/// @note The binding is consumed on success and failure, so a caller cannot retry with the same
/// authority after observing an error.
/// @note A successfully created manual publisher reserves its service identity until the owning
/// `ChannelRuntime` is destroyed, even after the publisher itself is destroyed. This prevents an
/// older subscriber from reconnecting to a fresh sidecar generation under the same name.
/// @note Endpoint setup for the retained runtime is internally serialized through backend
/// construction, so another factory cannot observe provisional sidecar state that is later rolled
/// back.
[[nodiscard]] HOLOSCAN_CORE_EXPORT expected<ManualTensorPublisher, Error>
make_manual_tensor_publisher(ManualChannelBinding binding, ManualTensorPublisherOptions options);

}  // namespace holoscan
