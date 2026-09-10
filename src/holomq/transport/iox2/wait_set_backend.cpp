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
#include <memory>
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

auto Iox2WaitSetBackend::attach_trigger(std::shared_ptr<void> owner, const int descriptor,
                                        std::function<void()> drain)
    -> holomq::expected<WaitSetHandle, holomq::Error> {
  if (owner == nullptr || descriptor < 0 || !drain) {
    return holomq::make_unexpected(holomq::Error{holomq::ErrorCode::kInvalidArgument});
  }

  auto descriptor_or = ::iox2::FileDescriptor::create_non_owning(descriptor);
  if (!descriptor_or.has_value()) {
    HOLOMQ_LOG_ERROR("WaitSet attach_trigger failed: invalid wake descriptor");
    return holomq::make_unexpected(
        holomq::Error{holomq::ErrorCode::kFailure, "wait trigger has no usable wake descriptor"});
  }

  AttachmentEntry entry{};
  entry.handle = WaitSetHandle{next_handle_id_++};
  entry.trigger = true;
  entry.kernel_attached = true;
  // Readiness means "a subscriber may have a sample", which a trigger never does. Cancellation is
  // reported through WaitTrigger::is_set instead, so no foreign handle reaches ready_handles().
  entry.has_samples = []() { return holomq::expected<bool, holomq::ErrorCode>{false}; };
  // Drain unconditionally. The monotonic latch carries cancellation state, so leaving the
  // descriptor readable would only make a level-triggered attachment report forever.
  entry.drain_wake_events = [drain = std::move(drain)]() {
    drain();
    return holomq::expected<bool, holomq::ErrorCode>{false};
  };
  entry.set_attached_to_wait_set = [](const bool) {};
  entry.descriptor = std::make_unique<::iox2::FileDescriptor>(std::move(descriptor_or).value());

  auto guard_or = waitset_.attach_notification(entry.descriptor->as_view());
  if (!guard_or.has_value()) {
    auto error = map_wait_set_attachment_error(guard_or.error());
    HOLOMQ_LOG_ERROR("WaitSet attach_trigger failed: {}", holomq::error_message(error));
    return holomq::make_unexpected(std::move(error));
  }
  entry.guard.emplace(std::move(guard_or).value());
  // Retained so the wake object outlives both the guard and the descriptor above.
  entry.attachment_owner = std::move(owner);

  const WaitSetHandle handle = entry.handle;
  attachments_.push_back(std::move(entry));
  return handle;
}

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
auto Iox2WaitSetBackend::wait_kernel(const WaitStrategy& strategy,
                                     const bool precheck_ready) noexcept
    -> holomq::expected<WaitResult, holomq::ErrorCode> {
  const auto spin_until = std::chrono::steady_clock::now() + strategy.spin_duration;

  // Public waits poll readiness at least once before blocking. That protects an unbounded caller
  // when a queued sample has no corresponding notification. The Runtime uses the unchecked entry
  // point only after its pump has exhaustively drained every endpoint, so its zero-spin hot path
  // can skip this repeated O(N) scan.
  //
  // A checked wait yields once even at spin_duration zero, and that is deliberate: it is
  // load-bearing when a caller has not established the Runtime pump's exhaustive-drain invariant.
  // A single-slot receive and notification pair can otherwise fall into lockstep when the caller
  // blocks straight out of a not-ready poll: the publisher's notify is rejected because the slot
  // is still full, and progress reverts to the wait timeout instead of a wake.
  // Measured on the graph interprocess POD 1:1 saturated benchmark, dropping this yield cost
  // 175,919 -> 8,769 msg/s, moved end-to-end p99 from 9.9 us to 3.7 ms, and drove publisher
  // notification attempts per accepted notification from 9.5 to 300.8. The median is untouched
  // either way, so a latency-percentile or unit test will not catch its removal. The private
  // Runtime entry skips both this check and yield only after its preceding endpoint drain.
  if (precheck_ready || strategy.spin_duration > std::chrono::nanoseconds::zero()) {
    do {
      auto ready = collect_ready_handles();
      if (!ready.has_value()) {
        return holomq::make_unexpected(ready.error());
      }
      if (ready.value()) {
        return WaitResult::ready;
      }
      std::this_thread::yield();
    } while (std::chrono::steady_clock::now() < spin_until);
  }

  if (attachments_.empty()) {
    return holomq::make_unexpected(holomq::ErrorCode::kInvalidArgument);
  }

  bool saw_kernel_notification = false;
  // Tracked separately from saw_kernel_notification because a trigger contributes no ready handle:
  // it is the one wake that can end a kernel wait with nothing to dispatch, and a caller driving a
  // ready_handles() loop has to be able to tell that apart from a spurious wake.
  bool saw_trigger_notification = false;
  holomq::ErrorCode drain_error = holomq::ErrorCode::kSuccess;
  const auto callback = ::iox2::bb::StaticFunction<::iox2::CallbackProgression(
      ::iox2::WaitSetAttachmentId<::iox2::ServiceType::Ipc>)>(
      [this, &saw_kernel_notification, &saw_trigger_notification, &drain_error](
          const ::iox2::WaitSetAttachmentId<::iox2::ServiceType::Ipc>& attachment_id) {
        for (auto& attachment : attachments_) {
          if (attachment.kernel_attached && attachment.guard.has_value() &&
              attachment_id.has_event_from(attachment.guard.value())) {
            saw_kernel_notification = true;
            saw_trigger_notification = saw_trigger_notification || attachment.trigger;
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

  auto ready = collect_ready_handles();
  if (!ready.has_value()) {
    return holomq::make_unexpected(ready.error());
  }
  if (ready.value()) {
    // Data wins over cancellation. A wake that carried both is reported as ready so the sample is
    // not dropped in favour of a shutdown; the caller's WaitTrigger::is_set check still sees the
    // latch on this pass or the next one.
    return WaitResult::ready;
  }

  // A fired trigger ends the wait with nothing to dispatch. Reporting it as ready would leave a
  // ready_handles() loop spinning on an empty span until its deadline, because the latch is
  // monotonic and the descriptor has already been drained above.
  if (saw_trigger_notification) {
    return WaitResult::cancelled;
  }

  // Kernel wake without queued samples yet (spurious wake); callers loop on empty ready_handles().
  if (saw_kernel_notification || !timed_kernel_wait) {
    return WaitResult::ready;
  }

  return WaitResult::timeout;
}

auto Iox2WaitSetBackend::wait(const WaitStrategy& strategy, const bool precheck_ready) noexcept
    -> holomq::expected<WaitResult, holomq::ErrorCode> {
  ready_handles_.clear();
  detail::warn_if_ignored_timeout_on_return(strategy);

  if (attachments_.empty()) {
    HOLOMQ_LOG_ERROR("WaitSet wait failed: no attachments");
    return holomq::make_unexpected(holomq::ErrorCode::kInvalidArgument);
  }

  if (strategy.mode == WaitMode::KernelWait) {
    // A cancellation trigger is kernel attached but is not an event-wake subscriber. Counting it
    // here would admit a kernel wait that no publication can ever end, which is precisely the
    // silent non-waking fallback CHAN-066 prohibits.
    const bool has_kernel_attachment =
        std::any_of(attachments_.begin(), attachments_.end(), [](const AttachmentEntry& entry) {
          return entry.kernel_attached && !entry.trigger;
        });

    if (!has_kernel_attachment) {
      HOLOMQ_LOG_ERROR(
          "WaitSet wait failed: KernelWait requires event wake on at least one attachment");
      return holomq::make_unexpected(holomq::ErrorCode::kInvalidArgument);
    }

    return wait_kernel(strategy, precheck_ready);
  }

  return wait_poll(strategy);
}

auto create_wait_set_backend(Runtime& runtime)
    -> holomq::expected<Iox2WaitSetBackend, holomq::Error> {
  return Iox2WaitSetBackend::create(runtime);
}

}  // namespace holomq::detail::transport
