/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_CONDITIONS_GXF_CUDA_STREAM_HPP
#define HOLOSCAN_CORE_CONDITIONS_GXF_CUDA_STREAM_HPP

#include <cuda_runtime_api.h>
#include <gxf/core/gxf.h>

#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gxf/std/receiver.hpp>
#include "../../component_spec.hpp"
#include "../../condition.hpp"
#include "../../resources/gxf/receiver.hpp"

namespace holoscan {

/**
 * @brief Native condition class for CUDA stream synchronization with multi-message support.
 *
 * This condition supports:
 * - Multiple messages in the input queue (queue_size > 1)
 * - Multiple CudaStreamId components per message
 * - Multiple receiver ports (both regular and IOSpec::kAnySize multi-receiver inputs)
 *
 * By default, this condition examines ALL messages in all receiver queues and waits for
 * GPU work on ALL associated CUDA streams to complete before allowing the operator to execute.
 * This behavior can be changed by setting `check_all_messages` to `false`, which makes it
 * check only the first message per receiver.
 *
 * The condition uses `cudaLaunchHostFunc()` to register callbacks that fire when GPU work
 * on each stream completes. It returns `kWaitEvent` status while waiting for callbacks,
 * then transitions to `kReady` when all callbacks have fired.
 *
 * **Note**: This condition does NOT consume messages - it only peeks at them. The operator's
 * compute() method is responsible for actually receiving the messages.
 *
 * ==Parameters==
 *
 * - **receiver** (std::string): **DEPRECATED** - Use `receivers` instead. Legacy API for a single
 *   input port to monitor. Cannot be used together with `receivers`. Using this parameter will
 *   log a deprecation warning.
 *
 * - **receivers** (std::string or std::vector<std::string>): Name(s) of input port(s) to
 *   monitor for CUDA streams. Can be a single string like "input" or a vector like
 *   {"input1", "input2"}. Works with both:
 *   - Regular ports: specified by exact name (e.g., "input")
 *   - Multi-receiver ports (IOSpec::kAnySize): specified by base name (e.g., "receivers"),
 *     which automatically expands to find all ports matching the pattern ("receivers:0",
 *     "receivers:1", etc.)
 *   Cannot be used together with `receiver`.
 *
 * - **check_all_messages** (bool, default=true): If true, checks ALL messages in the queue(s)
 *   for CudaStreamId components and waits for all associated streams. If false, only checks
 *   the first message in each queue.
 *
 * ==Usage==
 *
 * ```cpp
 * // Single regular input port (legacy API)
 * auto stream_cond = make_condition<CudaStreamCondition>(
 *     "cuda_sync", Arg("receiver", "input"));
 *
 * // Single regular input port (new API)
 * auto stream_cond = make_condition<CudaStreamCondition>(
 *     "cuda_sync", Arg("receivers", "input"));
 *
 * // Multiple regular input ports
 * auto stream_cond_multi = make_condition<CudaStreamCondition>(
 *     "cuda_sync", Arg("receivers", std::vector<std::string>{"input1", "input2"}));
 *
 * // Multi-receiver port (e.g., HolovizOp's "receivers")
 * auto stream_cond_any = make_condition<CudaStreamCondition>(
 *     "cuda_sync", Arg("receivers", "receivers"));  // finds receivers:0, receivers:1, etc.
 *
 * // Check only first message per receiver
 * auto stream_cond_first = make_condition<CudaStreamCondition>(
 *     "cuda_sync", Arg("receivers", "input"), Arg("check_all_messages", false));
 *
 * auto my_op = make_operator<MyOperator>("my_op", stream_cond);
 * ```
 */
class CudaStreamCondition : public Condition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS(CudaStreamCondition)

  CudaStreamCondition() = default;

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  void update_state(int64_t timestamp) override;
  void check(int64_t timestamp, SchedulingStatusType* type,
             int64_t* target_timestamp) const override;
  void on_execute(int64_t timestamp) override;

  /// Set receiver for this condition (legacy single-port API)
  void receiver(std::shared_ptr<Receiver> receiver) { receiver_ = std::move(receiver); }

  /// Get the receiver for this condition (nullptr if not set)
  std::shared_ptr<Receiver> receiver() { return receiver_.has_value() ? receiver_.get() : nullptr; }

  /// Set receivers for this condition (for all input ports to monitor)
  void receivers(std::vector<std::shared_ptr<Receiver>> receivers) {
    receivers_ = std::move(receivers);
  }

  /// Get the receivers for this condition (empty vector if not set)
  std::vector<std::shared_ptr<Receiver>> receivers() {
    return receivers_.has_value() ? receivers_.get() : std::vector<std::shared_ptr<Receiver>>{};
  }

  /// Set whether to check all messages in the queue
  void check_all_messages(bool value) { check_all_messages_ = value; }

  /// Get whether to check all messages in the queue
  bool check_all_messages() const { return check_all_messages_.get(); }

 private:
  /// State machine states for the condition
  enum class State {
    UNSET,                 ///< No messages to process or condition was reset
    CALLBACKS_REGISTERED,  ///< Host callbacks registered, waiting for completion
    DATA_AVAILABLE         ///< All callbacks fired, operator ready to execute
  };

  /// Data passed to CUDA host callbacks
  struct CallbackData {
    CudaStreamCondition* condition;
  };

  /// CUDA host callback function - called when GPU work on a stream completes
  static void CUDART_CB cuda_host_callback(void* user_data);

  /// Register host callbacks for all streams found in messages across all receiver queues
  void register_callbacks_for_receivers();

  /// Register host callbacks for streams in a single receiver's queue
  void register_callbacks_for_single_receiver(nvidia::gxf::Receiver* gxf_recv,
                                              gxf_context_t gxf_context, size_t& streams_found);

  /// Get the GXF context from the fragment
  void* get_gxf_context() const;

  /// Collect all receivers from the receivers_ parameter
  std::vector<std::shared_ptr<Receiver>> get_all_receivers() const;

  Parameter<std::shared_ptr<Receiver>> receiver_;  ///< Legacy single-receiver parameter
  Parameter<std::vector<std::shared_ptr<Receiver>>> receivers_;
  Parameter<bool> check_all_messages_;

  std::atomic<State> state_{State::UNSET};
  std::atomic<size_t> pending_callbacks_{0};
  int64_t last_state_change_{0};
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CONDITIONS_GXF_CUDA_STREAM_HPP */
