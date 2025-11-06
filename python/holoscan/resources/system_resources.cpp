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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "../core/component_util.hpp"
#include "./system_resources_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/resources/gxf/system_resources.hpp"
#include "holoscan/core/subgraph.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

class PyThreadPool : public ThreadPool {
 public:
  /* Inherit the constructors */
  using ThreadPool::ThreadPool;

  // Define a constructor that fully initializes the object.
  explicit PyThreadPool(const std::variant<Fragment*, Subgraph*>& fragment_or_subgraph,
                        int64_t initial_size = 1, const std::string& name = "thread_pool")
      : ThreadPool(ArgList{Arg("initial_size", initial_size)}) {
    init_component_base(this, fragment_or_subgraph, name, "resource");
  }
};

void init_system_resources(py::module_& m) {
  py::enum_<SchedulingPolicy>(m, "SchedulingPolicy")
      .value("SCHED_FIFO", SchedulingPolicy::kFirstInFirstOut)
      .value("SCHED_RR", SchedulingPolicy::kRoundRobin)
      .value("SCHED_DEADLINE", SchedulingPolicy::kDeadline);

  py::class_<ThreadPool,
             PyThreadPool,
             gxf::GXFSystemResourceBase,
             gxf::GXFResource,
             std::shared_ptr<ThreadPool>>(m, "ThreadPool", doc::ThreadPool::doc_ThreadPool_kwargs)
      .def(py::init<std::variant<Fragment*, Subgraph*>, int64_t, const std::string&>(),
           "fragment"_a,
           "initial_size"_a = 1,
           "name"_a = "thread_pool"s,
           doc::ThreadPool::doc_ThreadPool_kwargs)
      .def("add",
           py::overload_cast<const std::shared_ptr<Operator>&, bool, std::vector<uint32_t>>(
               &ThreadPool::add),
           "op"_a,
           "pin_operator"_a = false,
           "pin_cores"_a = std::vector<uint32_t>())
      .def_property_readonly("operators", &ThreadPool::operators, doc::ThreadPool::doc_operators)
      .def("add",
           py::overload_cast<std::vector<std::shared_ptr<Operator>>, bool, std::vector<uint32_t>>(
               &ThreadPool::add),
           "ops"_a,
           "pin_operator"_a = false,
           "pin_cores"_a = std::vector<uint32_t>(),
           doc::ThreadPool::doc_add)
      // use std::variant for policy to also accept an integer or string value parsed from YAML
      .def(
          "add_realtime",
          [](ThreadPool& self,
             const std::shared_ptr<Operator>& op,
             std::variant<SchedulingPolicy, int, std::string>
                 sched_policy_variant,
             bool pin_operator = true,
             std::vector<uint32_t> pin_cores = std::vector<uint32_t>(),
             uint32_t sched_priority = 0,
             uint64_t sched_runtime = 0,
             uint64_t sched_deadline = 0,
             uint64_t sched_period = 0) {
            // Convert variant to SchedulingPolicy enum
            SchedulingPolicy policy;
            std::visit(
                [&policy](auto&& arg) {
                  using T = std::decay_t<decltype(arg)>;
                  if constexpr (std::is_same_v<T, SchedulingPolicy>) {
                    // Already the correct enum type
                    policy = arg;
                  } else if constexpr (std::is_same_v<T, int>) {
                    // Convert integer to enum
                    switch (arg) {
                      case static_cast<int>(SchedulingPolicy::kFirstInFirstOut):
                        policy = SchedulingPolicy::kFirstInFirstOut;
                        break;
                      case static_cast<int>(SchedulingPolicy::kRoundRobin):
                        policy = SchedulingPolicy::kRoundRobin;
                        break;
                      case static_cast<int>(SchedulingPolicy::kDeadline):
                        policy = SchedulingPolicy::kDeadline;
                        break;
                      default:
                        throw py::value_error("Invalid scheduling policy integer: " +
                                              std::to_string(arg));
                    }
                  } else if constexpr (std::is_same_v<T, std::string>) {
                    // Convert string to enum
                    if (arg == "SCHED_FIFO") {
                      policy = SchedulingPolicy::kFirstInFirstOut;
                    } else if (arg == "SCHED_RR") {
                      policy = SchedulingPolicy::kRoundRobin;
                    } else if (arg == "SCHED_DEADLINE") {
                      policy = SchedulingPolicy::kDeadline;
                    } else {
                      throw py::value_error(
                          "Invalid scheduling policy string: '" + arg +
                          "'. Valid values are: 'SCHED_FIFO', 'SCHED_RR', 'SCHED_DEADLINE'");
                    }
                  }
                },
                sched_policy_variant);

            self.add_realtime(op,
                              policy,
                              pin_operator,
                              pin_cores,
                              sched_priority,
                              sched_runtime,
                              sched_deadline,
                              sched_period);
          },
          "op"_a,
          "sched_policy"_a,
          "pin_operator"_a = true,
          "pin_cores"_a = std::vector<uint32_t>(),
          "sched_priority"_a = 0,
          "sched_runtime"_a = 0,
          "sched_deadline"_a = 0,
          "sched_period"_a = 0,
          doc::ThreadPool::doc_add_realtime);
}
}  // namespace holoscan
