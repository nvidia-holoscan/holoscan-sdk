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
