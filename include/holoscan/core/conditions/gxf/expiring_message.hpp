/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_CONDITIONS_GXF_EXPIRING_MESSAGE_HPP
#define HOLOSCAN_CORE_CONDITIONS_GXF_EXPIRING_MESSAGE_HPP

#include <memory>

#include <gxf/std/scheduling_terms.hpp>

#include "../../gxf/gxf_condition.hpp"
#include "../../resources/gxf/clock.hpp"
#include "../../resources/gxf/realtime_clock.hpp"

namespace holoscan {

/**
 * @brief Condition class to allow an operator to execute when a specified number of messages have
 * arrived, or a specified time interval has elapsed since the first message was created.
 *
 * This condition applies to a specific input port of the operator as determined by setting the
 * "receiver" argument.
 *
 * This condition can also be set via the `Operator::setup` method using `IOSpec::condition` with
 * `ConditionType::kExpiringMessageAvailable`. In that case, the receiver is already known from the
 * port corresponding to the `IOSpec` object, so the "receiver" argument is unnecessary.
 *
 * **Note:** The `max_delay_ns` used by this condition type is relative to the timestamp of the
 * oldest message in the receiver queue. Use of this condition requires that the upstream operator
 * emitted a timestamp for at least one message in the queue. Holoscan Operators do not emit a
 * timestamp by default, but only when it is explicitly requested in the `Operator::emit` call. The
 * built-in operators of the SDK do not currently emit a timestamp, so this condition cannot be
 * easily used with the provided operators. As a potential alternative, please see
 * `MultiMessageAvailableTimeoutCondition` which can be configured to use a single port and a
 * timeout interval without needing a timestamp. A timestamp is not needed in the case of
 * `MultiMessageAvailableTimeoutCondition` because the interval measured is the time since the same
 * operator previously ticked.
 *
 * ==Parameters==
 *
 * - **max_batch_size** (int64_t): The maximum number of messages that can arrive before the
 * operator will be considered READY. The operator can still be considered READY with fewer
 * messages once `max_delay_ns` has elapsed.
 * - **max_delay_ns** (int64_t): The maximum delay to wait from the time of the first message
 * before the operator is considered READY. The units are in nanoseconds. A constructor is also
 * provided which allows setting this via a `std::chrono::duration` instead.
 * - **clock** (std::shared_ptr<holoscan::Clock>): The clock used by the scheduler to define the
 * flow of time. If not provided, a default-constructed `holoscan::RealtimeClock` will be used.
 * - **receiver** (std::string): The receiver whose message queue will be checked. This should be
 * specified by the name of the Operator's input port the condition will apply to. The Holoscan SDK
 * will then automatically replace the port name with the actual receiver object at application run
 * time.
 */
class ExpiringMessageAvailableCondition : public gxf::GXFCondition {
 public:
  HOLOSCAN_CONDITION_FORWARD_ARGS_SUPER(ExpiringMessageAvailableCondition, GXFCondition)

  ExpiringMessageAvailableCondition() = default;

  explicit ExpiringMessageAvailableCondition(int64_t max_batch_size)
      : max_batch_size_(max_batch_size) {}

  ExpiringMessageAvailableCondition(int64_t max_batch_size, int64_t max_delay_ns)
      : max_batch_size_(max_batch_size), max_delay_ns_(max_delay_ns) {}

  template <typename Rep, typename Period>
  explicit ExpiringMessageAvailableCondition(int64_t max_batch_size,
                                             std::chrono::duration<Rep, Period> max_delay)
      : max_batch_size_(max_batch_size) {
    max_delay_ns_ = std::chrono::duration_cast<std::chrono::nanoseconds>(max_delay).count();
  }

  const char* gxf_typename() const override {
    return "nvidia::gxf::ExpiringMessageAvailableSchedulingTerm";
  }

  void receiver(std::shared_ptr<gxf::GXFResource> receiver) { receiver_ = receiver; }
  std::shared_ptr<gxf::GXFResource> receiver() { return receiver_.get(); }

  void setup(ComponentSpec& spec) override;

  void initialize() override;

  void max_batch_size(int64_t max_batch_size);
  int64_t max_batch_size() { return max_batch_size_; }

  /**
   * @brief Set max delay.
   *
   * Note that calling this method doesn't affect the behavior of the condition once the condition
   * is initialized.
   *
   * @param max_delay_ns The integer representing max delay in nanoseconds.
   */
  void max_delay(int64_t max_delay_ns);

  /**
   * @brief Set max delay.
   *
   * Note that calling this method doesn't affect the behavior of the condition once the
   * condition is initialized.
   *
   * @param max_delay_duration The max delay of type `std::chrono::duration`.
   */
  template <typename Rep, typename Period>
  void max_delay(std::chrono::duration<Rep, Period> max_delay_duration) {
    int64_t max_delay_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(max_delay_duration).count();
    max_delay(max_delay_ns);
  }

  /**
   * @brief Get max delay in nano seconds.
   *
   * @return The minimum time which needs to elapse between two executions (in nano seconds)
   */
  int64_t max_delay_ns();

  nvidia::gxf::ExpiringMessageAvailableSchedulingTerm* get() const;

  // TODO(GXF4):   Expected<void> setReceiver(Handle<Receiver> value)

 private:
  // TODO(GXF4): this is now a std::set<Handle<Receiver>> receivers_
  Parameter<std::shared_ptr<gxf::GXFResource>> receiver_;
  Parameter<int64_t> max_batch_size_;
  Parameter<int64_t> max_delay_ns_;
  Parameter<std::shared_ptr<Clock>> clock_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_CONDITIONS_GXF_EXPIRING_MESSAGE_HPP */
