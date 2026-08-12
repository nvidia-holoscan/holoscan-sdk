/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holomq/detail/transport/iox2/wait_set_backend.hpp>

#include <holomq/detail/transport/iox2/runtime.hpp>
#include <holomq/detail/wait_strategy_utils.hpp>
#include <holomq/logger.hpp>

#include "iox2/signal_handling_mode.hpp"

#include <algorithm>
#include <chrono>
#include <thread>
#include <utility>

namespace holomq::detail::transport {

namespace {

[[nodiscard]] std::uint64_t allocate_wait_set_id() noexcept {
  static std::uint64_t next_id{1};
  return next_id++;
}

[[nodiscard]] WaitResult readiness_or_timeout(const bool ready) noexcept {
  return ready ? WaitResult::ready : WaitResult::timeout;
}

}  // namespace

Iox2WaitSetBackend::~Iox2WaitSetBackend() {
  for (auto& attachment : attachments_) {
    attachment.set_attached_to_wait_set(false);
  }
}

auto Iox2WaitSetBackend::create(::iox2::Node<::iox2::ServiceType::Ipc>& node)
    -> holomq::expected<Iox2WaitSetBackend, holomq::Error> {
  auto waitset_or = std::move(::iox2::WaitSetBuilder())
                        .signal_handling_mode(::iox2::SignalHandlingMode::Disabled)
                        .create<::iox2::ServiceType::Ipc>();
  if (!waitset_or.has_value()) {
    return holomq::make_unexpected(map_wait_set_create_error(waitset_or.error()));
  }
  return Iox2WaitSetBackend(node, std::move(waitset_or).value(), allocate_wait_set_id());
}

auto Iox2WaitSetBackend::create(Runtime& runtime)
    -> holomq::expected<Iox2WaitSetBackend, holomq::Error> {
  return create(runtime.node_for_waitset());
}

Iox2WaitSetBackend::Iox2WaitSetBackend(::iox2::Node<::iox2::ServiceType::Ipc>& node,
                                       ::iox2::WaitSet<::iox2::ServiceType::Ipc> waitset,
                                       const std::uint64_t wait_set_id) noexcept
    : node_(&node), waitset_(std::move(waitset)), wait_set_id_(wait_set_id) {}

void Iox2WaitSetBackend::detach(const WaitSetHandle handle) noexcept {
  const auto it =
      std::find_if(attachments_.begin(), attachments_.end(), [&](const AttachmentEntry& entry) {
        return entry.handle == handle;
      });
  if (it == attachments_.end()) {
    return;
  }
  it->set_attached_to_wait_set(false);
  attachments_.erase(it);
}

std::size_t Iox2WaitSetBackend::capacity() const noexcept {
  return static_cast<std::size_t>(waitset_.capacity());
}

auto Iox2WaitSetBackend::collect_ready_handles() noexcept
    -> holomq::expected<bool, holomq::ErrorCode> {
  ready_handles_.clear();
  for (const auto& attachment : attachments_) {
    auto ready = attachment.has_samples();
    if (!ready.has_value()) {
      HOLOMQ_LOG_ERROR("WaitSet readiness query failed: {}",
                       holomq::error_code_name(ready.error()));
      return holomq::make_unexpected(ready.error());
    }
    if (ready.value()) {
      ready_handles_.push_back(attachment.handle);
    }
  }
  return !ready_handles_.empty();
}

auto Iox2WaitSetBackend::wait_poll(const WaitStrategy& strategy) noexcept
    -> holomq::expected<WaitResult, holomq::ErrorCode> {
  const auto spin_until = std::chrono::steady_clock::now() + strategy.spin_duration;

  while (std::chrono::steady_clock::now() < spin_until) {
    auto ready = collect_ready_handles();
    if (!ready.has_value()) {
      return holomq::make_unexpected(ready.error());
    }
    if (ready.value()) {
      return WaitResult::ready;
    }
    std::this_thread::yield();
  }

  if (strategy.mode == WaitMode::Return) {
    auto ready = collect_ready_handles();
    if (!ready.has_value()) {
      return holomq::make_unexpected(ready.error());
    }
    return readiness_or_timeout(ready.value());
  }

  std::this_thread::sleep_for(detail::timed_wait_sleep_duration(strategy));
  auto ready = collect_ready_handles();
  if (!ready.has_value()) {
    return holomq::make_unexpected(ready.error());
  }
  return readiness_or_timeout(ready.value());
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
auto Iox2WaitSetBackend::wait_kernel(const WaitStrategy& strategy) noexcept
    -> holomq::expected<WaitResult, holomq::ErrorCode> {
  const auto spin_until = std::chrono::steady_clock::now() + strategy.spin_duration;

  while (std::chrono::steady_clock::now() < spin_until) {
    auto ready = collect_ready_handles();
    if (!ready.has_value()) {
      return holomq::make_unexpected(ready.error());
    }
    if (ready.value()) {
      return WaitResult::ready;
    }
    std::this_thread::yield();
  }

  if (attachments_.empty()) {
    return holomq::make_unexpected(holomq::ErrorCode::kInvalidArgument);
  }

  bool saw_kernel_notification = false;
  holomq::ErrorCode drain_error = holomq::ErrorCode::kSuccess;
  const auto callback = ::iox2::bb::StaticFunction<::iox2::CallbackProgression(
      ::iox2::WaitSetAttachmentId<::iox2::ServiceType::Ipc>)>(
      [this, &saw_kernel_notification, &drain_error](
          const ::iox2::WaitSetAttachmentId<::iox2::ServiceType::Ipc>& attachment_id) {
        for (auto& attachment : attachments_) {
          if (attachment.kernel_attached && attachment.guard.has_value() &&
              attachment_id.has_event_from(attachment.guard.value())) {
            saw_kernel_notification = true;
            auto drained = attachment.drain_wake_events();
            if (!drained.has_value()) {
              HOLOMQ_LOG_ERROR("WaitSet wake drain failed: {}",
                               holomq::error_code_name(drained.error()));
              drain_error = drained.error();
              return ::iox2::CallbackProgression::Stop;
            }
          }
        }
        return ::iox2::CallbackProgression::Continue;
      });

  const bool timed_kernel_wait = strategy.timeout.has_value();

  if (timed_kernel_wait) {
    const ::iox2::bb::Duration timeout =
        ::iox2::bb::Duration::from_nanos(static_cast<std::uint64_t>(strategy.timeout->count()));
    const auto run_or = waitset_.wait_and_process_once_with_timeout(callback, timeout);
    if (!run_or.has_value()) {
      return holomq::make_unexpected(map_wait_set_run_error_code(run_or.error()));
    }
  } else {
    const auto run_or = waitset_.wait_and_process_once(callback);
    if (!run_or.has_value()) {
      return holomq::make_unexpected(map_wait_set_run_error_code(run_or.error()));
    }
  }

  if (drain_error != holomq::ErrorCode::kSuccess) {
    return holomq::make_unexpected(drain_error);
  }

  for (auto& attachment : attachments_) {
    if (attachment.kernel_attached) {
      auto drained = attachment.drain_wake_events();
      if (!drained.has_value()) {
        HOLOMQ_LOG_ERROR("WaitSet wake drain failed: {}", holomq::error_code_name(drained.error()));
        return holomq::make_unexpected(drained.error());
      }
    }
  }

  auto ready = collect_ready_handles();
  if (!ready.has_value()) {
    return holomq::make_unexpected(ready.error());
  }
  if (ready.value()) {
    return WaitResult::ready;
  }

  // Kernel wake without queued samples yet (spurious wake); callers loop on empty ready_handles().
  if (saw_kernel_notification || !timed_kernel_wait) {
    return WaitResult::ready;
  }

  return WaitResult::timeout;
}

auto Iox2WaitSetBackend::wait(const WaitStrategy& strategy) noexcept
    -> holomq::expected<WaitResult, holomq::ErrorCode> {
  ready_handles_.clear();
  detail::warn_if_ignored_timeout_on_return(strategy);

  if (attachments_.empty()) {
    HOLOMQ_LOG_ERROR("WaitSet wait failed: no attachments");
    return holomq::make_unexpected(holomq::ErrorCode::kInvalidArgument);
  }

  if (strategy.mode == WaitMode::KernelWait) {
    const bool has_kernel_attachment =
        std::any_of(attachments_.begin(), attachments_.end(), [](const AttachmentEntry& entry) {
          return entry.kernel_attached;
        });

    if (!has_kernel_attachment) {
      HOLOMQ_LOG_ERROR(
          "WaitSet wait failed: KernelWait requires event wake on at least one attachment");
      return holomq::make_unexpected(holomq::ErrorCode::kInvalidArgument);
    }

    return wait_kernel(strategy);
  }

  return wait_poll(strategy);
}

auto create_wait_set_backend(Runtime& runtime)
    -> holomq::expected<Iox2WaitSetBackend, holomq::Error> {
  return Iox2WaitSetBackend::create(runtime);
}

}  // namespace holomq::detail::transport
