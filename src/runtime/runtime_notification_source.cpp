/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_notification_source.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "../core/readiness_source_internal.hpp"
#include "physical_readiness_binding.hpp"

namespace holoscan::detail {
namespace {

enum class NotificationSenderTerminal : std::uint8_t {
  kNotReady,
  kActive,
  kStale,
  kClosed,
};

struct NotificationSenderRecord {
  std::atomic<std::size_t> lifetime_references{1U};  // registry owner
  std::atomic<std::size_t> sender_references{};
  std::atomic<std::size_t> active_calls{};
  std::atomic_bool claimed{};
  std::atomic<NotificationSenderTerminal> terminal{NotificationSenderTerminal::kNotReady};
  ClockRef clock;
  PhysicalReadinessControl control;
  std::uint64_t source_generation{};
};

[[nodiscard]] Error terminal_error(NotificationSenderTerminal terminal) noexcept {
  switch (terminal) {
    case NotificationSenderTerminal::kNotReady:
      return Error{ErrorCode::kNotReady};
    case NotificationSenderTerminal::kActive:
      return Error{ErrorCode::kSuccess};
    case NotificationSenderTerminal::kStale:
      return Error{ErrorCode::kStaleEndpoint};
    case NotificationSenderTerminal::kClosed:
      return Error{ErrorCode::kClosed};
  }
  return Error{ErrorCode::kFailure};
}

void retain_sender_record(void* opaque) noexcept {
  auto& record = *static_cast<NotificationSenderRecord*>(opaque);
  record.sender_references.fetch_add(1U, std::memory_order_relaxed);
  record.lifetime_references.fetch_add(1U, std::memory_order_relaxed);
}

void release_sender_record(void* opaque) noexcept {
  auto* const record = static_cast<NotificationSenderRecord*>(opaque);
  if (record->sender_references.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
    record->claimed.store(false, std::memory_order_release);
  }
  if (record->lifetime_references.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
    delete record;
  }
}

void release_registry_record(NotificationSenderRecord* record) noexcept {
  if (record != nullptr &&
      record->lifetime_references.fetch_sub(1U, std::memory_order_acq_rel) == 1U) {
    delete record;
  }
}

void leave_sender_call(NotificationSenderRecord& record) noexcept {
  if (record.active_calls.fetch_sub(1U, std::memory_order_seq_cst) == 1U) {
    record.active_calls.notify_all();
  }
}

[[nodiscard]] expected<void, Error> post_notification(void* opaque) noexcept {
  auto& record = *static_cast<NotificationSenderRecord*>(opaque);
  NotificationSenderTerminal terminal = record.terminal.load(std::memory_order_seq_cst);
  if (terminal != NotificationSenderTerminal::kActive) {
    return make_unexpected(terminal_error(terminal));
  }
  record.active_calls.fetch_add(1U, std::memory_order_seq_cst);
  terminal = record.terminal.load(std::memory_order_seq_cst);
  if (terminal != NotificationSenderTerminal::kActive) {
    leave_sender_call(record);
    return make_unexpected(terminal_error(terminal));
  }

  auto accepted_at = record.clock.now();
  auto posted = accepted_at ? record.control.post_notification(*accepted_at)
                            : expected<void, Error>{make_unexpected(accepted_at.error())};
  leave_sender_call(record);
  return posted;
}

void transition_record(NotificationSenderRecord& record,
                       NotificationSenderTerminal terminal) noexcept {
  record.terminal.store(terminal, std::memory_order_seq_cst);
  std::size_t active = record.active_calls.load(std::memory_order_seq_cst);
  while (active != 0U) {
    record.active_calls.wait(active, std::memory_order_relaxed);
    active = record.active_calls.load(std::memory_order_seq_cst);
  }
}

[[nodiscard]] bool contains(std::span<const VertexId> owners, VertexId owner) noexcept {
  return std::find(owners.begin(), owners.end(), owner) != owners.end();
}

}  // namespace

struct RuntimeNotificationSourceRegistry::Impl {
  struct SourceEntry {
    VertexId owner{};
    RuntimePortBindingRequest request;
    ClockRef clock;
    std::vector<NotificationSenderRecord*> sender_records;
    PhysicalReadinessControl control;
    std::uint64_t current_generation{};
    bool restarting{};
  };

  std::shared_ptr<PhysicalReadinessBindingFactory> factory;
  std::vector<SourceEntry> sources;
  std::vector<PhysicalReadinessControl> restart_control_scratch;
  std::vector<NotificationSenderRecord*> records;
  bool startup_armed{};
  bool closed{};

  ~Impl() {
    for (NotificationSenderRecord* record : records) {
      release_registry_record(record);
    }
  }

  [[nodiscard]] SourceEntry* find(SourceId source) noexcept {
    const auto found = std::find_if(sources.begin(), sources.end(), [&](const SourceEntry& row) {
      return row.request.source == source;
    });
    return found == sources.end() ? nullptr : &*found;
  }
};

RuntimeNotificationSourceRegistry::RuntimeNotificationSourceRegistry(
    std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

RuntimeNotificationSourceRegistry::~RuntimeNotificationSourceRegistry() {
  close();
}

expected<std::unique_ptr<RuntimeNotificationSourceRegistry>, Error>
RuntimeNotificationSourceRegistry::create(
    std::span<const RuntimeNotificationSourceDescriptor> descriptors) {
  if (descriptors.empty()) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  try {
#endif
    std::vector<PhysicalReadinessBindingDescriptor> physical;
    physical.reserve(descriptors.size());
    auto impl = std::make_unique<Impl>();
    impl->sources.reserve(descriptors.size());
    std::size_t record_count = 0U;
    for (const auto& descriptor : descriptors) {
      if (!descriptor.owner || descriptor.request.kind != RuntimeEventKind::kNotification ||
          descriptor.sender_reference_capacity == 0U ||
          descriptor.clock.id() != descriptor.request.reference_clock ||
          record_count >
              std::numeric_limits<std::size_t>::max() - descriptor.sender_reference_capacity) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      record_count += descriptor.sender_reference_capacity;
      physical.push_back(PhysicalReadinessBindingDescriptor{
          .request = descriptor.request,
          .wake_requirement = descriptor.wake_requirement,
          .wake_mode = descriptor.wake_mode,
          .execution_lane = descriptor.request.execution_lane,
          .wake_state = descriptor.wake_state,
          .wake = descriptor.wake,
      });
      impl->sources.push_back(Impl::SourceEntry{
          .owner = descriptor.owner,
          .request = descriptor.request,
          .clock = descriptor.clock,
          .current_generation = descriptor.request.source_generation,
      });
    }
    auto factory = PhysicalReadinessBindingFactory::create(physical);
    if (!factory) {
      return make_unexpected(std::move(factory.error()));
    }
    impl->factory = std::move(*factory);
    impl->restart_control_scratch.resize(descriptors.size());
    impl->records.reserve(record_count);
    for (std::size_t source_index = 0U; source_index < descriptors.size(); ++source_index) {
      auto& entry = impl->sources[source_index];
      entry.sender_records.reserve(descriptors[source_index].sender_reference_capacity);
      for (std::uint32_t index = 0U; index < descriptors[source_index].sender_reference_capacity;
           ++index) {
        auto record = std::make_unique<NotificationSenderRecord>();
        entry.sender_records.push_back(record.get());
        impl->records.push_back(record.release());
      }
    }
    return std::unique_ptr<RuntimeNotificationSourceRegistry>(
        new RuntimeNotificationSourceRegistry(std::move(impl)));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
#endif
}

expected<std::unique_ptr<PortBinding>, Error> RuntimeNotificationSourceRegistry::bind(
    const RuntimePortBindingRequest& request, TimePoint start_coordinate) noexcept {
  if (impl_ == nullptr || impl_->closed) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  Impl::SourceEntry* const entry = impl_->find(request.source);
  if (entry == nullptr) {
    return make_unexpected(Error{ErrorCode::kNotFound});
  }
  if (request.plan_generation != entry->request.plan_generation ||
      request.run_generation != entry->request.run_generation ||
      request.source_generation != entry->request.source_generation) {
    return make_unexpected(Error{ErrorCode::kStaleEndpoint});
  }
  auto binding = impl_->factory->bind(request, start_coordinate);
  if (!binding) {
    return make_unexpected(std::move(binding.error()));
  }
  auto control = impl_->factory->control(
      request.source, request.plan_generation, request.run_generation, request.source_generation);
  if (!control) {
    return make_unexpected(std::move(control.error()));
  }
  entry->control = std::move(*control);
  return binding;
}

expected<NotificationSender, Error> RuntimeNotificationSourceRegistry::acquire_sender(
    VertexId owner, NotificationSource& source, LifecycleStage stage,
    LifecycleTransitionPhase phase) noexcept {
  if (impl_ == nullptr || impl_->closed) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  if (stage != LifecycleStage::kArm || (phase != LifecycleTransitionPhase::kStartupForward &&
                                        phase != LifecycleTransitionPhase::kRestartForward)) {
    return make_unexpected(Error{ErrorCode::kLifecycleInvalidCallingContext});
  }
  const NotificationSourceRef ref = ReadinessSourceAccess::ref(source);
  Impl::SourceEntry* const entry = impl_->find(ref.id());
  if (entry == nullptr || entry->owner != owner ||
      ref != ReadinessSourceAccess::make_ref<ReadinessSourceKind::kNotification>(
                 entry->request.source, entry->request.plan_generation)) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  if ((phase == LifecycleTransitionPhase::kStartupForward && impl_->startup_armed) ||
      (phase == LifecycleTransitionPhase::kRestartForward && !entry->restarting)) {
    return make_unexpected(Error{ErrorCode::kLifecycleInvalidCallingContext});
  }
  const std::uint64_t desired_generation = [&] {
    if (phase == LifecycleTransitionPhase::kStartupForward) {
      return entry->current_generation;
    }
    return entry->current_generation == std::numeric_limits<std::uint64_t>::max()
               ? std::uint64_t{0U}
               : entry->current_generation + 1U;
  }();
  if (desired_generation == 0U ||
      (phase == LifecycleTransitionPhase::kStartupForward && !entry->control)) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
  for (NotificationSenderRecord* record : entry->sender_records) {
    bool expected = false;
    if (!record->claimed.compare_exchange_strong(
            expected, true, std::memory_order_acq_rel, std::memory_order_acquire)) {
      continue;
    }
    record->clock = entry->clock;
    record->control = phase == LifecycleTransitionPhase::kStartupForward
                          ? entry->control
                          : PhysicalReadinessControl{};
    record->source_generation = desired_generation;
    record->sender_references.store(1U, std::memory_order_relaxed);
    record->lifetime_references.fetch_add(1U, std::memory_order_relaxed);
    record->terminal.store(NotificationSenderTerminal::kNotReady, std::memory_order_release);
    return ReadinessSourceAccess::make_sender(
        record, &retain_sender_record, &release_sender_record, &post_notification);
  }
  return make_unexpected(Error{ErrorCode::kResourceExhausted});
}

expected<void, Error> RuntimeNotificationSourceRegistry::arm_startup() noexcept {
  if (impl_ == nullptr || impl_->closed || impl_->startup_armed) {
    return make_unexpected(
        Error{impl_ == nullptr || impl_->closed ? ErrorCode::kClosed : ErrorCode::kDuplicateName});
  }
  for (const auto& entry : impl_->sources) {
    if (!entry.control) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
  }
  for (auto& entry : impl_->sources) {
    for (NotificationSenderRecord* record : entry.sender_records) {
      if (record->claimed.load(std::memory_order_acquire) &&
          record->source_generation == entry.current_generation) {
        record->terminal.store(NotificationSenderTerminal::kActive, std::memory_order_release);
      }
    }
  }
  impl_->startup_armed = true;
  return {};
}

void RuntimeNotificationSourceRegistry::begin_restart(std::span<const VertexId> owners) noexcept {
  if (impl_ == nullptr || impl_->closed) {
    return;
  }
  for (auto& entry : impl_->sources) {
    if (!contains(owners, entry.owner)) {
      continue;
    }
    entry.restarting = true;
    for (NotificationSenderRecord* record : entry.sender_records) {
      if (record->claimed.load(std::memory_order_acquire) &&
          record->source_generation == entry.current_generation) {
        transition_record(*record, NotificationSenderTerminal::kNotReady);
      }
    }
  }
}

expected<void, Error> RuntimeNotificationSourceRegistry::publish_restart(
    std::span<const VertexId> owners) noexcept {
  if (impl_ == nullptr || impl_->closed) {
    return make_unexpected(Error{ErrorCode::kClosed});
  }
  for (std::size_t index = 0U; index < impl_->sources.size(); ++index) {
    auto& entry = impl_->sources[index];
    if (!contains(owners, entry.owner) || !entry.restarting) {
      continue;
    }
    if (!entry.control) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    if (entry.current_generation == std::numeric_limits<std::uint64_t>::max()) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }
    const std::uint64_t next_generation = entry.current_generation + 1U;
    auto control = impl_->factory->control(entry.request.source,
                                           entry.request.plan_generation,
                                           entry.request.run_generation,
                                           next_generation);
    if (!control) {
      return make_unexpected(std::move(control.error()));
    }
    impl_->restart_control_scratch[index] = std::move(*control);
  }
  for (std::size_t index = 0U; index < impl_->sources.size(); ++index) {
    auto& entry = impl_->sources[index];
    if (!contains(owners, entry.owner) || !entry.restarting) {
      continue;
    }
    const std::uint64_t next_generation = entry.current_generation + 1U;
    auto& control = impl_->restart_control_scratch[index];
    for (NotificationSenderRecord* record : entry.sender_records) {
      if (!record->claimed.load(std::memory_order_acquire)) {
        continue;
      }
      if (record->source_generation == entry.current_generation) {
        transition_record(*record, NotificationSenderTerminal::kStale);
      } else if (record->source_generation == next_generation) {
        record->control = control;
        record->terminal.store(NotificationSenderTerminal::kActive, std::memory_order_release);
      }
    }
    entry.control = std::move(control);
    entry.current_generation = next_generation;
    entry.restarting = false;
  }
  return {};
}

void RuntimeNotificationSourceRegistry::rollback_restart(
    std::span<const VertexId> owners) noexcept {
  if (impl_ == nullptr || impl_->closed) {
    return;
  }
  for (auto& entry : impl_->sources) {
    if (!contains(owners, entry.owner) || !entry.restarting) {
      continue;
    }
    for (NotificationSenderRecord* record : entry.sender_records) {
      if (!record->claimed.load(std::memory_order_acquire)) {
        continue;
      }
      const NotificationSenderTerminal terminal =
          record->source_generation == entry.current_generation
              ? NotificationSenderTerminal::kActive
              : NotificationSenderTerminal::kStale;
      record->terminal.store(terminal, std::memory_order_release);
    }
    entry.restarting = false;
  }
}

void RuntimeNotificationSourceRegistry::close() noexcept {
  if (impl_ == nullptr || std::exchange(impl_->closed, true)) {
    return;
  }
  for (NotificationSenderRecord* record : impl_->records) {
    transition_record(*record, NotificationSenderTerminal::kClosed);
  }
  impl_->factory->close_all();
}

}  // namespace holoscan::detail
