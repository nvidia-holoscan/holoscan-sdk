/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holomq/detail/transport/iox2/setup_error.hpp>
#include <holomq/detail/transport/iox2/subscriber_port.hpp>

#include <iox2/callback_progression.hpp>
#include <iox2/file_descriptor.hpp>
#include <iox2/node.hpp>
#include <iox2/service_type.hpp>
#include <iox2/waitset.hpp>

#include <holomq/errors.hpp>
#include <holomq/expected.hpp>
#include <holomq/logger.hpp>
#include <holomq/wait_set_handle.hpp>
#include <holomq/wait_strategy.hpp>

#include <algorithm>
#include <atomic>
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
  ~Iox2WaitSetBackend();

  /// Deleted rather than defaulted. A defaulted move-assignment runs in declaration order, so it
  /// would assign \c waitset_ while the destination's \c attachments_ still hold guards borrowed
  /// from it, dropping those guards against a wait set that has already been replaced. Nothing
  /// needs the operation -- \c create only ever move-constructs -- so deleting it closes the trap
  /// instead of leaving a correct-by-accident declaration order for a future edit to break.
  Iox2WaitSetBackend& operator=(Iox2WaitSetBackend&&) = delete;

  Iox2WaitSetBackend(const Iox2WaitSetBackend&) = delete;
  Iox2WaitSetBackend& operator=(const Iox2WaitSetBackend&) = delete;

  [[nodiscard]] static auto create(::iox2::Node<::iox2::ServiceType::Ipc>& node)
      -> holomq::expected<Iox2WaitSetBackend, holomq::Error>;

  [[nodiscard]] static auto create(Runtime& runtime)
      -> holomq::expected<Iox2WaitSetBackend, holomq::Error>;

  template <typename Payload>
  [[nodiscard]] auto attach(std::shared_ptr<SubscriberPort<Payload>> port)
      -> holomq::expected<WaitSetHandle, holomq::Error>;

  /// Register a cancellation wake descriptor so another thread can end a blocked kernel wait.
  ///
  /// The backend deliberately takes the descriptor and drain callback rather than the owning
  /// trigger type, so no public wake abstraction has to befriend a transport backend. \p owner is
  /// retained only to keep the descriptor alive for at least as long as the attachment.
  [[nodiscard]] auto attach_trigger(std::shared_ptr<void> owner, int descriptor,
                                    std::function<void()> drain)
      -> holomq::expected<WaitSetHandle, holomq::Error>;

  /// Install, or with \c nullptr remove, the readiness mask of one attached subscriber.
  ///
  /// While the mask is set, the subscriber's queued samples do not make a wait report ready and
  /// its handle is absent from \ref ready_handles. Its kernel notifications are still drained, and
  /// they still end a blocked wait, which then reports a spurious wake. The mask's owner may set or
  /// clear it from any thread; installing it is externally serialized with attach and wait.
  [[nodiscard]] auto set_readiness_mask(WaitSetHandle handle,
                                        std::shared_ptr<const std::atomic<bool>> mask) noexcept
      -> holomq::expected<void, holomq::ErrorCode>;

  /// Attach a reusable wake that \ref signal_wake posts from any thread.
  ///
  /// A posted wake ends a blocked kernel wait, or the next one, as a spurious wake: \ref wait
  /// reports \ref WaitResult::ready and the wake contributes no handle. Unlike a cancellation
  /// trigger it does not latch, so later waits block again once it is drained. At most one wake can
  /// be attached.
  [[nodiscard]] auto attach_wake() -> holomq::expected<void, holomq::Error>;

  /// Post the wake attached by \ref attach_wake. Posts coalesce until a wait drains them.
  ///
  /// Safe to call from any thread, concurrently with \ref wait, once \ref attach_wake has
  /// returned on the waiting thread and the backend has been shared from there.
  [[nodiscard]] auto signal_wake() const noexcept -> holomq::expected<void, holomq::ErrorCode>;

  void detach(WaitSetHandle handle) noexcept;

  [[nodiscard]] auto wait(const WaitStrategy& strategy, bool precheck_ready) noexcept
      -> holomq::expected<WaitResult, holomq::ErrorCode>;

  [[nodiscard]] std::span<const WaitSetHandle> ready_handles() const noexcept {
    return ready_handles_;
  }

  [[nodiscard]] std::size_t capacity() const noexcept;

  /// Number of attached subscribers. Cancellation triggers and the wake occupy an attachment slot
  /// but are not subscribers, so they are excluded to keep this count meaning what its callers
  /// assume.
  [[nodiscard]] std::size_t attachment_count() const noexcept {
    return static_cast<std::size_t>(std::count_if(
        attachments_.begin(), attachments_.end(), [](const AttachmentEntry& entry) noexcept {
          return entry.is_subscriber();
        }));
  }

  [[nodiscard]] std::uint64_t wait_set_id() const noexcept { return wait_set_id_; }

 private:
  struct AttachmentEntry {
    WaitSetHandle handle;
    bool kernel_attached{false};
    /// Set for a cancellation trigger rather than a subscriber. A trigger is excluded from the
    /// kernel-wait admission check so it cannot stand in for the event-wake capability, and its
    /// readiness query always reports false so it never appears in \ref ready_handles.
    bool trigger{false};
    /// Set for the reusable wake from \ref attach_wake. Like a trigger it is not a subscriber and
    /// never appears in \ref ready_handles, but its notification is reported as a spurious wake.
    bool wake{false};
    /// Retains a non-subscriber attachment owner: a \c WaitTrigger or the wake's descriptor.
    /// Declared before
    /// \c descriptor and \c guard so both are destroyed first: iceoryx2 requires the descriptor to
    /// outlive its guard, and the guard issues EPOLL_CTL_DEL on destruction.
    std::shared_ptr<void> attachment_owner;
    /// Heap held so growing \c attachments_ cannot move it. iceoryx2 requires the descriptor to
    /// outlive its guard, and a guard registered from a view of a moved-from descriptor would
    /// reference storage the vector has relocated.
    std::unique_ptr<::iox2::FileDescriptor> descriptor;
    std::optional<::iox2::WaitSetGuard<::iox2::ServiceType::Ipc>> guard;
    std::function<holomq::expected<bool, holomq::ErrorCode>()> has_samples;
    std::function<holomq::expected<bool, holomq::ErrorCode>()> drain_wake_events;
    std::function<void(bool)> set_attached_to_wait_set;
    /// Installed by \ref set_readiness_mask. While set, \ref collect_ready_handles skips this
    /// entry.
    std::shared_ptr<const std::atomic<bool>> readiness_mask;

    AttachmentEntry() = default;
    AttachmentEntry(AttachmentEntry&&) noexcept = default;
    ~AttachmentEntry() = default;
    AttachmentEntry(const AttachmentEntry&) = delete;
    AttachmentEntry& operator=(const AttachmentEntry&) = delete;

    /// Move-assignment is written out because the implicit one is unsafe here. It would run in
    /// declaration order and assign \c descriptor before \c guard, destroying this entry's
    /// descriptor while its own guard still borrows it; \c WaitSetGuard::operator=(&&) then drops
    /// that guard and issues EPOLL_CTL_DEL on released storage. Releasing \c guard first restores
    /// the same ordering the member declaration order gives destruction.
    ///
    /// This is not a corner case: \c std::vector::erase move-assigns every following element down,
    /// so detaching any entry but the last takes this path, which is what the Channel layer's
    /// cancellation re-arm does on every wait that follows a cancellation.
    AttachmentEntry& operator=(AttachmentEntry&& other) noexcept {
      if (this != &other) {
        guard.reset();
        handle = other.handle;
        kernel_attached = other.kernel_attached;
        trigger = other.trigger;
        wake = other.wake;
        attachment_owner = std::move(other.attachment_owner);
        descriptor = std::move(other.descriptor);
        guard = std::move(other.guard);
        has_samples = std::move(other.has_samples);
        drain_wake_events = std::move(other.drain_wake_events);
        set_attached_to_wait_set = std::move(other.set_attached_to_wait_set);
        readiness_mask = std::move(other.readiness_mask);
      }
      return *this;
    }

    [[nodiscard]] bool is_subscriber() const noexcept { return !trigger && !wake; }
  };

  explicit Iox2WaitSetBackend(::iox2::Node<::iox2::ServiceType::Ipc>& node,
                              ::iox2::WaitSet<::iox2::ServiceType::Ipc> waitset,
                              std::uint64_t wait_set_id) noexcept;

  [[nodiscard]] auto collect_ready_handles() noexcept -> holomq::expected<bool, holomq::ErrorCode>;

  [[nodiscard]] auto wait_poll(const WaitStrategy& strategy) noexcept
      -> holomq::expected<WaitResult, holomq::ErrorCode>;

  [[nodiscard]] auto wait_kernel(const WaitStrategy& strategy, bool precheck_ready) noexcept
      -> holomq::expected<WaitResult, holomq::ErrorCode>;

  ::iox2::Node<::iox2::ServiceType::Ipc>* node_{nullptr};
  ::iox2::WaitSet<::iox2::ServiceType::Ipc> waitset_;
  std::uint64_t wait_set_id_{0};
  std::uint64_t next_handle_id_{1};
  std::vector<AttachmentEntry> attachments_;
  std::vector<WaitSetHandle> ready_handles_;
  /// The wake's eventfd, owned by its attachment entry; -1 until \ref attach_wake succeeds.
  int wake_descriptor_{-1};
};

template <typename Payload>
auto Iox2WaitSetBackend::attach(std::shared_ptr<SubscriberPort<Payload>> port)
    -> holomq::expected<WaitSetHandle, holomq::Error> {
  if (port->is_attached_to_wait_set()) {
    HOLOMQ_LOG_ERROR("WaitSet attach failed: subscriber already attached");
    return holomq::make_unexpected_nothrow<WaitSetHandle>(
        holomq::ErrorCode::kInvalidArgument,
        port->is_attached_to_readiness_handle()
            ? "subscriber is already owned by a readiness handle"
            : "subscriber is already owned by a wait set");
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
