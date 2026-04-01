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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../core/component_util.hpp"
#include "./event_based_scheduler_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/component_traits.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_component.hpp"
#include "holoscan/core/gxf/gxf_scheduler.hpp"
#include "holoscan/core/resources/gxf/clock.hpp"
#include "holoscan/core/resources/gxf/realtime_clock.hpp"
#include "holoscan/core/schedulers/gxf/event_based_scheduler.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

/* Trampoline classes for handling Python kwargs
 *
 * These add a constructor that takes a Fragment for which to initialize the scheduler.
 * The explicit parameter list and default arguments take care of providing a Pythonic
 * kwarg-based interface with appropriate default values matching the scheduler's
 * default parameters in the C++ API `setup` method.
 *
 * The sequence of events in this constructor is based on Fragment::make_scheduler<SchedulerT>
 */

class PyEventBasedScheduler : public EventBasedScheduler {
 public:
  /* Inherit the constructors */
  using EventBasedScheduler::EventBasedScheduler;

  // Define a constructor that fully initializes the object.
  explicit PyEventBasedScheduler(
      Fragment* fragment, std::shared_ptr<gxf::Clock> clock = nullptr,
      int64_t worker_thread_number = 1LL, bool stop_on_deadlock = true,
      int64_t max_duration_ms = -1LL, int64_t stop_on_deadlock_timeout = 0LL,
      int64_t network_connection_timeout = 5000LL,
      std::optional<std::vector<uint32_t>> pin_cores = std::nullopt,
      bool enable_queue_stealing = false, int64_t steal_scan_limit = 0LL,
      bool enable_worker_postcheck_fastpath = false,
      int64_t postcheck_fallback_notify_interval = 256LL,
      int64_t postcheck_fallback_notify_min_workers = 8LL,
      int64_t postcheck_fallback_notify_min_period_ns = 100000LL,
      int64_t internal_event_shard_count = 0LL, int64_t dispatcher_internal_pop_batch_size = 32LL,
      int64_t wait_state_shard_count = 1LL, bool log_perf_stats = false,
      const std::string& name = scheduler_default_name_v<EventBasedScheduler>)
      : EventBasedScheduler(ArgList{
            Arg{"worker_thread_number", worker_thread_number},
            Arg{"stop_on_deadlock", stop_on_deadlock},
            Arg{"stop_on_deadlock_timeout", stop_on_deadlock_timeout},
            Arg{"network_connection_timeout", network_connection_timeout},
            Arg{"enable_queue_stealing", enable_queue_stealing},
            Arg{"steal_scan_limit", steal_scan_limit},
            Arg{"enable_worker_postcheck_fastpath", enable_worker_postcheck_fastpath},
            Arg{"postcheck_fallback_notify_interval", postcheck_fallback_notify_interval},
            Arg{"postcheck_fallback_notify_min_workers", postcheck_fallback_notify_min_workers},
            Arg{"postcheck_fallback_notify_min_period_ns", postcheck_fallback_notify_min_period_ns},
            Arg{"internal_event_shard_count", internal_event_shard_count},
            Arg{"dispatcher_internal_pop_batch_size", dispatcher_internal_pop_batch_size},
            Arg{"wait_state_shard_count", wait_state_shard_count},
            Arg{"log_perf_stats", log_perf_stats}}) {
    if (max_duration_ms >= 0) {
      this->add_arg(Arg{"max_duration_ms", max_duration_ms});
    }
    if (pin_cores.has_value()) {
      this->add_arg(Arg("pin_cores", pin_cores.value()));
    }
    if (!fragment) {
      throw std::invalid_argument("fragment cannot be None");
    }
    if (clock) {
      this->add_arg(Arg{"clock", clock});
    } else {
      this->add_arg(Arg{"clock", fragment->make_resource<RealtimeClock>("realtime_clock")});
    }
    init_component_base(this, fragment, name);
  }
};

void init_event_based_scheduler(py::module_& m) {
  py::class_<EventBasedScheduler,
             PyEventBasedScheduler,
             gxf::GXFScheduler,
             Component,
             gxf::GXFComponent,
             std::shared_ptr<EventBasedScheduler>>(
      m, "EventBasedScheduler", doc::EventBasedScheduler::doc_EventBasedScheduler)
      .def(py::init<Fragment*,
                    std::shared_ptr<gxf::Clock>,
                    int64_t,
                    bool,
                    int64_t,
                    int64_t,
                    int64_t,
                    std::optional<std::vector<uint32_t>>,
                    bool,
                    int64_t,
                    bool,
                    int64_t,
                    int64_t,
                    int64_t,
                    int64_t,
                    int64_t,
                    int64_t,
                    bool,
                    const std::string&>(),
           "fragment"_a,
           py::kw_only(),
           "clock"_a = py::none(),
           "worker_thread_number"_a = 1LL,
           "stop_on_deadlock"_a = true,
           "max_duration_ms"_a = -1LL,
           "stop_on_deadlock_timeout"_a = 0LL,
           "network_connection_timeout"_a = 5000LL,
           "pin_cores"_a = std::nullopt,
           "enable_queue_stealing"_a = false,
           "steal_scan_limit"_a = 0LL,
           "enable_worker_postcheck_fastpath"_a = false,
           "postcheck_fallback_notify_interval"_a = 256LL,
           "postcheck_fallback_notify_min_workers"_a = 8LL,
           "postcheck_fallback_notify_min_period_ns"_a = 100000LL,
           "internal_event_shard_count"_a = 0LL,
           "dispatcher_internal_pop_batch_size"_a = 32LL,
           "wait_state_shard_count"_a = 1LL,
           "log_perf_stats"_a = false,
           "name"_a = std::string(scheduler_default_name_v<EventBasedScheduler>),
           doc::EventBasedScheduler::doc_EventBasedScheduler)
      .def_property_readonly("clock", &EventBasedScheduler::clock)
      .def_property_readonly("worker_thread_number", &EventBasedScheduler::worker_thread_number)
      .def_property_readonly("max_duration_ms", &EventBasedScheduler::max_duration_ms)
      .def_property_readonly("stop_on_deadlock", &EventBasedScheduler::stop_on_deadlock)
      .def_property_readonly("stop_on_deadlock_timeout",
                             &EventBasedScheduler::stop_on_deadlock_timeout)
      .def_property_readonly("network_connection_timeout",
                             &EventBasedScheduler::network_connection_timeout)
      .def_property_readonly("pin_cores", &EventBasedScheduler::pin_cores)
      .def_property_readonly("enable_queue_stealing", &EventBasedScheduler::enable_queue_stealing)
      .def_property_readonly("steal_scan_limit", &EventBasedScheduler::steal_scan_limit)
      .def_property_readonly("enable_worker_postcheck_fastpath",
                             &EventBasedScheduler::enable_worker_postcheck_fastpath)
      .def_property_readonly("postcheck_fallback_notify_interval",
                             &EventBasedScheduler::postcheck_fallback_notify_interval)
      .def_property_readonly("postcheck_fallback_notify_min_workers",
                             &EventBasedScheduler::postcheck_fallback_notify_min_workers)
      .def_property_readonly("postcheck_fallback_notify_min_period_ns",
                             &EventBasedScheduler::postcheck_fallback_notify_min_period_ns)
      .def_property_readonly("internal_event_shard_count",
                             &EventBasedScheduler::internal_event_shard_count)
      .def_property_readonly("dispatcher_internal_pop_batch_size",
                             &EventBasedScheduler::dispatcher_internal_pop_batch_size)
      .def_property_readonly("wait_state_shard_count", &EventBasedScheduler::wait_state_shard_count)
      .def_property_readonly("log_perf_stats", &EventBasedScheduler::log_perf_stats);
}  // PYBIND11_MODULE
}  // namespace holoscan
