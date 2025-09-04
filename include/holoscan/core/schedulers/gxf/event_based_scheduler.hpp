/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_SCHEDULERS_GXF_EVENT_BASED_SCHEDULER_HPP
#define HOLOSCAN_CORE_SCHEDULERS_GXF_EVENT_BASED_SCHEDULER_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gxf/std/event_based_scheduler.hpp>
#include "../../gxf/gxf_scheduler.hpp"
#include "../../resources/gxf/clock.hpp"

namespace holoscan {

/**
 * @brief Event-based scheduler.
 *
 * This is a multi-thread scheduler that uses an event-based design. Unlike the
 * `MultiThreadScheduler`, it does not utilize a dedicated polling thread that is constantly
 * polling operators to check which are ready to execute. Instead, certain events in the underlying
 * framework will indicate that the scheduling status of an operator should be checked.
 *
 * ==Parameters==
 *
 * - **worker_thread_number** (int64_t): The number of (CPU) worker threads to use for executing
 * operators. Defaults to 1.
 * - **pin_cores** (list of int, optional): CPU core IDs to pin the worker threads to (empty means
 * no core pinning).
 * - **stop_on_deadlock** (bool): If True, the application will terminate if a deadlock state is
 * reached. Defaults to true.
 * - **stop_on_deadlock_timeout** (int64_t): The amount of time (in ms) before an application is
 * considered to be in deadlock. Defaults to 0.
 * - **max_duration_ms_** (int64_t, optional): Terminate the application after the specified
 * duration even if deadlock does not occur. If unspecified, the application can run indefinitely.
 */
class EventBasedScheduler : public gxf::GXFScheduler {
 public:
  HOLOSCAN_SCHEDULER_FORWARD_ARGS_SUPER(EventBasedScheduler, gxf::GXFScheduler)
  EventBasedScheduler() = default;

  const char* gxf_typename() const override { return "nvidia::gxf::EventBasedScheduler"; }

  std::shared_ptr<Clock> clock() override;

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  // Parameter getters used for printing scheduler description (e.g. for Python __repr__)
  int64_t worker_thread_number() { return worker_thread_number_; }
  bool stop_on_deadlock() { return stop_on_deadlock_; }
  int64_t stop_on_deadlock_timeout() { return stop_on_deadlock_timeout_; }
  // could return std::optional<int64_t>, but just using int64_t simplifies the Python bindings
  int64_t max_duration_ms() { return max_duration_ms_.has_value() ? max_duration_ms_.get() : -1; }
  std::vector<uint32_t> pin_cores() {
    return pin_cores_.has_value() ? pin_cores_.get() : std::vector<uint32_t>{};
  }

  nvidia::gxf::EventBasedScheduler* get() const;

 private:
  Parameter<std::shared_ptr<gxf::Clock>> clock_;
  Parameter<int64_t> worker_thread_number_;
  Parameter<bool> stop_on_deadlock_;
  Parameter<int64_t> max_duration_ms_;
  Parameter<int64_t> stop_on_deadlock_timeout_;  // in ms
  Parameter<std::vector<uint32_t>> pin_cores_;  // CPU core IDs to pin the worker threads to
  // The following parameter needs to wait on ThreadPool support
  // Parameter<bool> thread_pool_allocation_auto_;

  void* clock_gxf_cptr() const override;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_SCHEDULERS_GXF_EVENT_BASED_SCHEDULER_HPP */
