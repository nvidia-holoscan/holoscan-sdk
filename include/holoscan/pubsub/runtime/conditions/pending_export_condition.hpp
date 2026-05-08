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

#ifndef HOLOSCAN_PUBSUB_RUNTIME_CONDITIONS_PENDING_EXPORT_CONDITION_HPP
#define HOLOSCAN_PUBSUB_RUNTIME_CONDITIONS_PENDING_EXPORT_CONDITION_HPP

#ifdef HOLOSCAN_HAS_PENDING_EXPORT_CONDITION

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>

#include "holoscan/core/condition.hpp"

#include "holoscan/pubsub/common/native_buffer_protocol_adapter.hpp"

namespace holoscan {

/// Native Holoscan condition that blocks execution while too many native-buffer
/// exports are in flight on a NativeBufferProtocolAdapter.
///
/// The condition is intended for publisher-side gating with CUDA IPC / holoipc:
/// - before and after execute it queries pending_export_count()
/// - when blocked it returns kWaitEvent
/// - when pending exports are released, the adapter callback wakes the scheduler
class PendingExportCondition : public Condition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS(PendingExportCondition)

  using AdapterResolver = std::function<std::shared_ptr<NativeBufferProtocolAdapter>()>;

  PendingExportCondition() = default;
  ~PendingExportCondition() override;

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  void update_state(int64_t timestamp) override;
  void check(int64_t timestamp, SchedulingStatusType* status_type,
             int64_t* target_timestamp) const override;
  void on_execute(int64_t timestamp) override;

  void adapter(std::shared_ptr<NativeBufferProtocolAdapter> adapter);
  std::shared_ptr<NativeBufferProtocolAdapter> adapter() const;

  void adapter_resolver(AdapterResolver resolver);

 private:
  enum class State {
    kReady,
    kWaiting,
  };

  struct CallbackBridge {
    std::mutex mutex;
    PendingExportCondition* owner = nullptr;
  };

  static int64_t current_time_ns();

  std::shared_ptr<NativeBufferProtocolAdapter> resolve_adapter();
  void attach_adapter(const std::shared_ptr<NativeBufferProtocolAdapter>& adapter);
  void detach_adapter();
  void evaluate_pending_count(size_t pending_count, int64_t timestamp, bool notify_on_ready);
  void on_pending_export_count_changed(size_t pending_count);

  Parameter<uint64_t> max_pending_;

  mutable std::mutex adapter_mutex_;
  std::shared_ptr<NativeBufferProtocolAdapter> adapter_;
  AdapterResolver adapter_resolver_;
  std::shared_ptr<CallbackBridge> callback_bridge_ = std::make_shared<CallbackBridge>();

  std::atomic<State> state_{State::kReady};
  std::atomic<size_t> last_pending_count_{0};
  std::atomic<int64_t> last_state_change_{0};
};

}  // namespace holoscan

#endif  // HOLOSCAN_HAS_PENDING_EXPORT_CONDITION

#endif /* HOLOSCAN_PUBSUB_RUNTIME_CONDITIONS_PENDING_EXPORT_CONDITION_HPP */
