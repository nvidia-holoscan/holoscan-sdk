/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "holoscan/pubsub/runtime/conditions/pending_export_condition.hpp"

#include <algorithm>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <utility>

#include <fmt/format.h>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

PendingExportCondition::~PendingExportCondition() {
  detach_adapter();
}

void PendingExportCondition::setup(ComponentSpec& spec) {
  spec.param(max_pending_,
             "max_pending",
             "Max pending exports",
             "The condition blocks execution while pending_export_count() is greater than or "
             "equal to this threshold.",
             static_cast<uint64_t>(1));
}

void PendingExportCondition::initialize() {
  Condition::initialize();

  if (max_pending_.get() == 0) {
    throw std::runtime_error(
        fmt::format("PendingExportCondition '{}' requires max_pending >= 1", name()));
  }

  auto adapter = resolve_adapter();
  if (!adapter || !adapter->is_initialized()) {
    HOLOSCAN_LOG_TRACE(
        "PendingExportCondition '{}': native buffer adapter not available yet; "
        "deferring attachment until runtime",
        name());
    last_state_change_.store(current_time_ns());
    state_.store(State::kReady);
    return;
  }

  HOLOSCAN_LOG_TRACE(
      "PendingExportCondition '{}': attached adapter, max_pending={}", name(), max_pending_.get());

  evaluate_pending_count(adapter->pending_export_count(), current_time_ns(), false);
}

void PendingExportCondition::update_state(int64_t timestamp) {
  auto adapter = resolve_adapter();
  if (!adapter || !adapter->is_initialized()) {
    return;
  }
  evaluate_pending_count(adapter->pending_export_count(), timestamp, false);
}

void PendingExportCondition::check(int64_t timestamp, SchedulingStatusType* status_type,
                                   int64_t* target_timestamp) const {
  if (status_type == nullptr) {
    throw std::runtime_error("PendingExportCondition::check received nullptr for status_type");
  }
  if (target_timestamp == nullptr) {
    throw std::runtime_error("PendingExportCondition::check received nullptr for target_timestamp");
  }

  if (state_.load() == State::kWaiting) {
    *status_type = SchedulingStatusType::kWaitEvent;
    *target_timestamp = 0;
  } else {
    *status_type = SchedulingStatusType::kReady;
    *target_timestamp = timestamp;
  }
}

void PendingExportCondition::on_execute(int64_t timestamp) {
  auto adapter = resolve_adapter();
  if (!adapter || !adapter->is_initialized()) {
    return;
  }
  evaluate_pending_count(adapter->pending_export_count(), timestamp, false);
}

void PendingExportCondition::adapter(std::shared_ptr<NativeBufferProtocolAdapter> adapter) {
  attach_adapter(std::move(adapter));
}

std::shared_ptr<NativeBufferProtocolAdapter> PendingExportCondition::adapter() const {
  std::lock_guard<std::mutex> lock(adapter_mutex_);
  return adapter_;
}

void PendingExportCondition::adapter_resolver(AdapterResolver resolver) {
  std::lock_guard<std::mutex> lock(adapter_mutex_);
  adapter_resolver_ = std::move(resolver);
}

int64_t PendingExportCondition::current_time_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Called on each update_state()/on_execute() until the adapter is successfully
// resolved and attached.  Repeated attempts are intentional: the adapter may not
// be initialized at the time this condition is first checked (e.g. the network
// context is still starting up) but becomes available on a later scheduler tick.
std::shared_ptr<NativeBufferProtocolAdapter> PendingExportCondition::resolve_adapter() {
  std::shared_ptr<NativeBufferProtocolAdapter> adapter;
  AdapterResolver resolver;
  {
    std::lock_guard<std::mutex> lock(adapter_mutex_);
    if (adapter_) {
      return adapter_;
    }
    resolver = adapter_resolver_;
  }

  if (!resolver) {
    return nullptr;
  }

  adapter = resolver();
  if (adapter && adapter->is_initialized()) {
    attach_adapter(adapter);
    HOLOSCAN_LOG_TRACE("PendingExportCondition '{}': attached adapter at runtime, max_pending={}",
                       name(),
                       max_pending_.get());
  }
  return adapter;
}

void PendingExportCondition::attach_adapter(
    const std::shared_ptr<NativeBufferProtocolAdapter>& adapter) {
  if (!adapter) {
    detach_adapter();
    return;
  }

  std::shared_ptr<NativeBufferProtocolAdapter> previous;
  {
    std::lock_guard<std::mutex> lock(adapter_mutex_);
    if (adapter_ == adapter) {
      return;
    }
    previous = std::move(adapter_);
    adapter_ = adapter;
  }
  if (previous) {
    previous->set_on_pending_export_count_changed({});
  }

  {
    std::lock_guard<std::mutex> lock(callback_bridge_->mutex);
    callback_bridge_->owner = this;
  }

  auto callback_bridge = callback_bridge_;
  adapter->set_on_pending_export_count_changed([callback_bridge](size_t pending_count) {
    std::lock_guard<std::mutex> lock(callback_bridge->mutex);
    if (callback_bridge->owner != nullptr) {
      callback_bridge->owner->on_pending_export_count_changed(pending_count);
    }
  });
}

void PendingExportCondition::detach_adapter() {
  std::shared_ptr<NativeBufferProtocolAdapter> adapter;
  {
    std::lock_guard<std::mutex> lock(adapter_mutex_);
    adapter = std::move(adapter_);
  }
  if (adapter) {
    adapter->set_on_pending_export_count_changed({});
  }

  std::lock_guard<std::mutex> lock(callback_bridge_->mutex);
  callback_bridge_->owner = nullptr;
}

void PendingExportCondition::evaluate_pending_count(size_t pending_count, int64_t timestamp,
                                                    bool notify_on_ready) {
  last_pending_count_.store(pending_count);

  const auto next_state = pending_count >= max_pending_.get() ? State::kWaiting : State::kReady;
  const auto previous_state = state_.exchange(next_state);

  if (previous_state != next_state) {
    last_state_change_.store(timestamp);
    if (next_state == State::kWaiting) {
      HOLOSCAN_LOG_TRACE(
          "PendingExportCondition '{}': blocking publisher (pending={} >= max_pending={})",
          name(),
          pending_count,
          max_pending_.get());
    } else {
      HOLOSCAN_LOG_TRACE("PendingExportCondition '{}': ready again (pending={} < max_pending={})",
                         name(),
                         pending_count,
                         max_pending_.get());
    }
    if (notify_on_ready && next_state == State::kReady) {
      notify_scheduler();
    }
  }
}

void PendingExportCondition::on_pending_export_count_changed(size_t pending_count) {
  evaluate_pending_count(pending_count, current_time_ns(), true);
}

}  // namespace holoscan
