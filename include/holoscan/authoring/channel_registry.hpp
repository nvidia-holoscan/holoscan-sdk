/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <stop_token>
#include <string_view>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/channel/channel.hpp>
#include <holoscan/channel/clock_health.hpp>
#include <holoscan/channel/endpoint_catalog.hpp>
#include <holoscan/channel/readiness_handle.hpp>
#include <holoscan/channel/runtime.hpp>
#include <holoscan/channel/tensor_loan.hpp>
#include <holoscan/channel/types.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/time/time_point.hpp>

namespace holoscan::authoring {

/// @file
/// @brief Prototype language-neutral standalone Channel registry adapters.

/// @brief Own one typed standalone Channel publisher behind a payload-registry boundary.
///
/// The matching registered payload row is the only authority that may recover the concrete
/// `ChannelPublisher<T>`. This Prototype adapter type intentionally exposes no transport or
/// payload-specific operations of its own.
class OpaqueChannelPublisher {
 public:
  /// @brief Create an empty invalid publisher holder.
  OpaqueChannelPublisher() = default;
  OpaqueChannelPublisher(OpaqueChannelPublisher&&) noexcept = default;
  OpaqueChannelPublisher& operator=(OpaqueChannelPublisher&&) noexcept = default;
  OpaqueChannelPublisher(const OpaqueChannelPublisher&) = delete;
  OpaqueChannelPublisher& operator=(const OpaqueChannelPublisher&) = delete;
  ~OpaqueChannelPublisher() = default;

  /// @brief Return whether this holder owns a typed publisher facade.
  [[nodiscard]] explicit operator bool() const noexcept { return holder_ != nullptr; }

 private:
  std::shared_ptr<void> holder_;
  const void* local_type_token_{};

  template <holoscan::detail::ExternalTopicPayload T>
  friend struct detail::PayloadRegistrationThunks;
};

/// @brief Own one typed standalone Channel subscriber behind a payload-registry boundary.
///
/// The matching registered payload row is the only authority that may recover the concrete
/// `ChannelSubscriber<T>`. The parent `ChannelRuntime` remains externally owned and must outlive
/// this holder, exactly as for the public typed C++ facade.
class OpaqueChannelSubscriber {
 public:
  /// @brief Create an empty invalid subscriber holder.
  OpaqueChannelSubscriber() = default;
  OpaqueChannelSubscriber(OpaqueChannelSubscriber&&) noexcept = default;
  OpaqueChannelSubscriber& operator=(OpaqueChannelSubscriber&&) noexcept = default;
  OpaqueChannelSubscriber(const OpaqueChannelSubscriber&) = delete;
  OpaqueChannelSubscriber& operator=(const OpaqueChannelSubscriber&) = delete;
  ~OpaqueChannelSubscriber() = default;

  /// @brief Return whether this holder owns a typed subscriber facade.
  [[nodiscard]] explicit operator bool() const noexcept { return holder_ != nullptr; }

 private:
  std::shared_ptr<void> holder_;
  const void* local_type_token_{};

  template <holoscan::detail::ExternalTopicPayload T>
  friend struct detail::PayloadRegistrationThunks;
};

/// @brief Describe standalone Channel operations for one registered payload row.
///
/// Keeping this cold callback family behind one pointer prevents the general authoring registry
/// header from importing Channel catalog and endpoint declarations into every authoring translation
/// unit. Every callback calls the matching public typed Channel facade.
struct RegisteredChannelThunks {
  /// Attach one matching standalone publisher through an immutable catalog endpoint view.
  expected<OpaqueChannelPublisher, Error> (*create_publisher)(
      ChannelRuntime&, ChannelEndpointCatalog::EndpointView,
      ChannelAttachOptions) noexcept = nullptr;
  /// Create one matching name-based standalone publisher.
  expected<OpaqueChannelPublisher, Error> (*create_name_publisher)(
      ChannelRuntime&, std::string_view, const ChannelOptions&) noexcept = nullptr;
  /// Attach one matching standalone subscriber through an immutable catalog endpoint view.
  expected<OpaqueChannelSubscriber, Error> (*create_subscriber)(
      ChannelRuntime&, ChannelEndpointCatalog::EndpointView,
      ChannelAttachOptions) noexcept = nullptr;
  /// Create one matching name-based standalone subscriber.
  expected<OpaqueChannelSubscriber, Error> (*create_name_subscriber)(
      ChannelRuntime&, std::string_view, const ChannelOptions&) noexcept = nullptr;
  /// Emit one exact fixed scalar through a standalone publisher.
  expected<std::uint32_t, Error> (*emit_scalar)(OpaqueChannelPublisher&, ParamScalarValue,
                                                EmitOptions) noexcept = nullptr;
  /// Emit one fixed scalar at a publisher-bound capture coordinate.
  expected<std::uint32_t, Error> (*emit_captured_scalar)(OpaqueChannelPublisher&, ParamScalarValue,
                                                         std::int64_t,
                                                         EmitOptions) noexcept = nullptr;
  /// Emit one exact process-local object through a standalone publisher.
  expected<std::uint32_t, Error> (*emit_object)(OpaqueChannelPublisher&, ForeignPayloadValue,
                                                EmitOptions) noexcept = nullptr;
  /// Emit one process-local object at a publisher-bound capture coordinate.
  expected<std::uint32_t, Error> (*emit_captured_object)(OpaqueChannelPublisher&,
                                                         ForeignPayloadValue, std::int64_t,
                                                         EmitOptions) noexcept = nullptr;
  /// Materialize and emit one generated-schema field tree through a standalone publisher.
  expected<std::uint32_t, Error> (*emit_schema)(OpaqueChannelPublisher&, ForeignSchemaValue,
                                                EmitOptions) noexcept = nullptr;
  /// Materialize and emit one schema value at a publisher-bound capture coordinate.
  expected<std::uint32_t, Error> (*emit_captured_schema)(OpaqueChannelPublisher&,
                                                         ForeignSchemaValue, std::int64_t,
                                                         EmitOptions) noexcept = nullptr;
  /// Acquire one endpoint-owned Tensor publication loan.
  expected<TensorPublishLoan, Error> (*loan_tensor)(OpaqueChannelPublisher&,
                                                    TensorLoanRequest) noexcept = nullptr;
  /// Consume and publish one ready endpoint-owned Tensor loan.
  expected<std::uint32_t, Error> (*emit_tensor_loan)(OpaqueChannelPublisher&, TensorPublishLoan,
                                                     EmitOptions) noexcept = nullptr;
  /// Copy and publish one admitted foreign Tensor.
  expected<std::uint32_t, Error> (*emit_tensor_copy)(OpaqueChannelPublisher&, const Tensor&,
                                                     ::cuda::stream_ref,
                                                     EmitOptions) noexcept = nullptr;
  /// Publish ordered end of stream and consume this publisher's authority on terminal outcomes.
  expected<void, Error> (*publisher_close)(OpaqueChannelPublisher&) noexcept = nullptr;
  /// Refresh this publisher's advisory peer snapshot.
  expected<void, Error> (*update_connections)(OpaqueChannelPublisher&) noexcept = nullptr;
  /// Read this publisher's latest advisory peer snapshot.
  bool (*has_subscribers)(OpaqueChannelPublisher&) noexcept = nullptr;
  /// Label one coordinate with this catalog publisher's capture clock.
  expected<TimePoint, Error> (*make_capture_time)(const OpaqueChannelPublisher&,
                                                  std::int64_t) noexcept = nullptr;
  /// Publish one clock-health control report.
  expected<void, Error> (*report_clock_health)(OpaqueChannelPublisher&,
                                               ClockHealthReport) noexcept = nullptr;
  /// Nonblocking receive of one fixed-scalar sample.
  expected<ForeignScalarSample, Error> (*try_receive_scalar)(OpaqueChannelSubscriber&) noexcept =
      nullptr;
  /// Blocking receive of one fixed-scalar sample against one absolute deadline.
  expected<ForeignScalarSample, Error> (*wait_and_receive_scalar)(
      OpaqueChannelSubscriber&, std::chrono::steady_clock::time_point,
      const std::stop_token&) noexcept = nullptr;
  /// Nonblocking receive of one process-local object sample.
  expected<ForeignObjectSample, Error> (*try_receive_object)(OpaqueChannelSubscriber&) noexcept =
      nullptr;
  /// Blocking receive of one process-local object sample against one absolute deadline.
  expected<ForeignObjectSample, Error> (*wait_and_receive_object)(
      OpaqueChannelSubscriber&, std::chrono::steady_clock::time_point,
      const std::stop_token&) noexcept = nullptr;
  /// Nonblocking receive of one generated-schema sample.
  expected<ForeignSchemaSample, Error> (*try_receive_schema)(OpaqueChannelSubscriber&) noexcept =
      nullptr;
  /// Blocking receive of one generated-schema sample against one absolute deadline.
  expected<ForeignSchemaSample, Error> (*wait_and_receive_schema)(
      OpaqueChannelSubscriber&, std::chrono::steady_clock::time_point,
      const std::stop_token&) noexcept = nullptr;
  /// Nonblocking receive of one read-only Tensor sample.
  expected<ForeignTensorSample, Error> (*try_receive_tensor)(OpaqueChannelSubscriber&) noexcept =
      nullptr;
  /// Blocking receive of one read-only Tensor sample against one absolute deadline.
  expected<ForeignTensorSample, Error> (*wait_and_receive_tensor)(
      OpaqueChannelSubscriber&, std::chrono::steady_clock::time_point,
      const std::stop_token&) noexcept = nullptr;
  /// Block for receive-relevant work without consuming a sample.
  expected<void, Error> (*wait)(OpaqueChannelSubscriber&, std::chrono::steady_clock::time_point,
                                const std::stop_token&) noexcept = nullptr;
  /// Claim one externally pollable readiness handle for this subscriber.
  expected<ChannelReadinessHandle, Error> (*subscriber_readiness)(
      OpaqueChannelSubscriber&) noexcept = nullptr;
  /// Retire this process's local subscriber endpoint without publishing EOS.
  expected<void, Error> (*subscriber_close)(OpaqueChannelSubscriber&) noexcept = nullptr;
  /// Refresh this subscriber's advisory peer snapshot without consuming a sample.
  expected<void, Error> (*refresh_connections)(OpaqueChannelSubscriber&) noexcept = nullptr;
  /// Query whether this subscriber may have a sample without consuming it.
  expected<bool, Error> (*has_samples)(const OpaqueChannelSubscriber&) noexcept = nullptr;
  /// Query whether this subscriber currently observes at least one publisher.
  bool (*has_publishers)(const OpaqueChannelSubscriber&) noexcept = nullptr;
};

}  // namespace holoscan::authoring
