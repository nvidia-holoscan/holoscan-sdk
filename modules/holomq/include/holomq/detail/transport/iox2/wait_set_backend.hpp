/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/detail/transport/iox2/setup_error.hpp>
#include <holomq/detail/transport/iox2/subscriber_port.hpp>

#include "iox2/callback_progression.hpp"
#include "iox2/node.hpp"
#include "iox2/service_type.hpp"
#include "iox2/waitset.hpp"

#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/logger.hpp>
#include <holomq/wait_set_handle.hpp>
#include <holomq/wait_strategy.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace holomq::detail::transport {

class Runtime;

class Iox2WaitSetBackend {
 public:
  Iox2WaitSetBackend(Iox2WaitSetBackend&&) noexcept = default;
  Iox2WaitSetBackend& operator=(Iox2WaitSetBackend&&) noexcept = default;
  ~Iox2WaitSetBackend();

  Iox2WaitSetBackend(const Iox2WaitSetBackend&) = delete;
  Iox2WaitSetBackend& operator=(const Iox2WaitSetBackend&) = delete;

  [[nodiscard]] static auto create(::iox2::Node<::iox2::ServiceType::Ipc>& node)
      -> holomq::expected<Iox2WaitSetBackend, holomq::Error>;

  [[nodiscard]] static auto create(Runtime& runtime)
      -> holomq::expected<Iox2WaitSetBackend, holomq::Error>;

  template <typename Payload>
  [[nodiscard]] auto attach(std::shared_ptr<SubscriberPort<Payload>> port)
      -> holomq::expected<WaitSetHandle, holomq::Error>;

  void detach(WaitSetHandle handle) noexcept;

  [[nodiscard]] auto wait(const WaitStrategy& strategy) noexcept
      -> holomq::expected<WaitResult, holomq::ErrorCode>;

  [[nodiscard]] std::span<const WaitSetHandle> ready_handles() const noexcept {
    return ready_handles_;
  }

  [[nodiscard]] std::size_t capacity() const noexcept;

  [[nodiscard]] std::size_t attachment_count() const noexcept { return attachments_.size(); }

  [[nodiscard]] std::uint64_t wait_set_id() const noexcept { return wait_set_id_; }

 private:
  struct AttachmentEntry {
    WaitSetHandle handle;
    bool kernel_attached{false};
    std::optional<::iox2::WaitSetGuard<::iox2::ServiceType::Ipc>> guard;
    std::function<holomq::expected<bool, holomq::ErrorCode>()> has_samples;
    std::function<holomq::expected<bool, holomq::ErrorCode>()> drain_wake_events;
    std::function<void(bool)> set_attached_to_wait_set;
  };

  explicit Iox2WaitSetBackend(::iox2::Node<::iox2::ServiceType::Ipc>& node,
                              ::iox2::WaitSet<::iox2::ServiceType::Ipc> waitset,
                              std::uint64_t wait_set_id) noexcept;

  [[nodiscard]] auto collect_ready_handles() noexcept -> holomq::expected<bool, holomq::ErrorCode>;

  [[nodiscard]] auto wait_poll(const WaitStrategy& strategy) noexcept
      -> holomq::expected<WaitResult, holomq::ErrorCode>;

  [[nodiscard]] auto wait_kernel(const WaitStrategy& strategy) noexcept
      -> holomq::expected<WaitResult, holomq::ErrorCode>;

  ::iox2::Node<::iox2::ServiceType::Ipc>* node_{nullptr};
  ::iox2::WaitSet<::iox2::ServiceType::Ipc> waitset_;
  std::uint64_t wait_set_id_{0};
  std::uint64_t next_handle_id_{1};
  std::vector<AttachmentEntry> attachments_;
  std::vector<WaitSetHandle> ready_handles_;
};

template <typename Payload>
auto Iox2WaitSetBackend::attach(std::shared_ptr<SubscriberPort<Payload>> port)
    -> holomq::expected<WaitSetHandle, holomq::Error> {
  if (port->is_attached_to_wait_set()) {
    HOLOMQ_LOG_ERROR("WaitSet attach failed: subscriber already attached");
    return holomq::make_unexpected(holomq::Error{holomq::ErrorCode::kInvalidArgument});
  }

  // A subscriber created before its publisher has not yet observed the publisher-side data
  // segment. Establish that inbound connection before registering readiness so the first
  // lossless publication after attachment cannot be reported as having zero recipients.
  auto ready = port->has_samples();
  if (!ready.has_value()) {
    return holomq::make_unexpected(holomq::Error{ready.error()});
  }

  AttachmentEntry entry{};
  entry.handle = WaitSetHandle{next_handle_id_++};
  entry.has_samples = [port]() { return port->has_samples(); };
  entry.drain_wake_events = [port]() { return port->drain_wake_events(); };
  entry.set_attached_to_wait_set = [port](const bool attached) {
    port->set_attached_to_wait_set(attached);
  };

  if (port->is_event_wake_enabled()) {
    auto* listener = port->get_listener_if_ready();
    if (listener == nullptr) {
      return holomq::make_unexpected(
          holomq::Error{holomq::ErrorCode::kFailure, "event-wake subscriber has no listener"});
    }
    auto guard_or = waitset_.attach_notification(*listener);
    if (!guard_or.has_value()) {
      auto error = map_wait_set_attachment_error(guard_or.error());
      HOLOMQ_LOG_ERROR("WaitSet attach failed: {}", holomq::error_message(error));
      return holomq::make_unexpected(std::move(error));
    }
    entry.kernel_attached = true;
    entry.guard.emplace(std::move(guard_or).value());
  }

  attachments_.push_back(std::move(entry));
  port->set_attached_to_wait_set(true);
  return attachments_.back().handle;
}

}  // namespace holomq::detail::transport
