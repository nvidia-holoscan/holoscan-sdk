/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_SCHEDULERS_GXF_GREEDY_SCHEDULER_HPP
#define HOLOSCAN_CORE_SCHEDULERS_GXF_GREEDY_SCHEDULER_HPP

#include <cstdint>
#include <memory>
#include <string>

#include <gxf/std/greedy_scheduler.hpp>
#include "../../gxf/gxf_scheduler.hpp"
#include "../../resources/gxf/clock.hpp"

namespace holoscan {

/**
 * @brief Greedy Scheduler.
 *
 * This is a single-threaded scheduler that will execute operators serially in a deterministic
 * order. Holoscan sorts operators so that execution will occur in topological order (moving from
 * the root to the leaves of the computation graph).
 *
 * ==Parameters==
 *
 * - **stop_on_deadlock** (bool): If True, the application will terminate if a deadlock state is
 * reached. Defaults to true.
 * - **stop_on_deadlock_timeout** (int64_t): The amount of time (in ms) before an application is
 * considered to be in deadlock. Defaults to 0.
 * - **check_recession_period_ms** (double): Duration to sleep before checking the condition of
 * the next operator (default: 0 ms). The units are in ms.
 * - **max_duration_ms_** (int64_t, optional): Terminate the application after the specified
 * duration even if deadlock does not occur. If unspecified, the application can run indefinitely.
 */
class GreedyScheduler : public gxf::GXFScheduler {
 public:
  HOLOSCAN_SCHEDULER_FORWARD_ARGS_SUPER(GreedyScheduler, gxf::GXFScheduler)
  GreedyScheduler() = default;

  const char* gxf_typename() const override { return "nvidia::gxf::GreedyScheduler"; }

  std::shared_ptr<Clock> clock() override;

  void setup(ComponentSpec& spec) override;
  void initialize() override;

  // Parameter getters used for printing scheduler description (e.g. for Python __repr__)
  bool stop_on_deadlock() const { return stop_on_deadlock_; }
  double check_recession_period_ms() const { return check_recession_period_ms_; }
  int64_t stop_on_deadlock_timeout() const { return stop_on_deadlock_timeout_; }
  int64_t network_connection_timeout() const { return network_connection_timeout_; }
  // could return std::optional<int64_t>, but just using int64_t simplifies the Python bindings
  int64_t max_duration_ms() const {
    return max_duration_ms_.has_value() ? max_duration_ms_.get() : -1;
  }

  nvidia::gxf::GreedyScheduler* get() const;

 private:
  Parameter<std::shared_ptr<gxf::Clock>> clock_;
  Parameter<bool> stop_on_deadlock_;
  Parameter<int64_t> max_duration_ms_;
  Parameter<double> check_recession_period_ms_;
  Parameter<int64_t> stop_on_deadlock_timeout_;    // in ms
  Parameter<int64_t> network_connection_timeout_;  // in ms
  void* clock_gxf_cptr() const override;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_SCHEDULERS_GXF_GREEDY_SCHEDULER_HPP */
