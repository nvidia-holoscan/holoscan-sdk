/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <vector>

#include "./system_resources_pydoc.hpp"
#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/resources/gxf/system_resources.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

class PyThreadPool : public ThreadPool {
 public:
  /* Inherit the constructors */
  using ThreadPool::ThreadPool;

  // Define a constructor that fully initializes the object.
  explicit PyThreadPool(Fragment* fragment, int64_t initial_size = 1,
                        const std::string& name = "thread_pool")
      : ThreadPool(ArgList{Arg("initial_size", initial_size)}) {
    name_ = name;
    fragment_ = fragment;
    spec_ = std::make_shared<ComponentSpec>(fragment);
    setup(*spec_);
  }
};

void init_system_resources(py::module_& m) {
  py::class_<ThreadPool,
             PyThreadPool,
             gxf::GXFSystemResourceBase,
             gxf::GXFResource,
             std::shared_ptr<ThreadPool>>(m, "ThreadPool", doc::ThreadPool::doc_ThreadPool_kwargs)
      .def(py::init<Fragment*, int64_t, const std::string&>(),
           "fragment"_a,
           "initial_size"_a = 1,
           "name"_a = "thread_pool"s,
           doc::ThreadPool::doc_ThreadPool_kwargs)
      .def("add",
           py::overload_cast<const std::shared_ptr<Operator>&, bool>(&ThreadPool::add),
           "op"_a,
           "pin_operator"_a = false)
      .def_property_readonly("operators", &ThreadPool::operators, doc::ThreadPool::doc_operators)
      .def("add",
           py::overload_cast<std::vector<std::shared_ptr<Operator>>, bool>(&ThreadPool::add),
           "ops"_a,
           "pin_operator"_a = false,
           doc::ThreadPool::doc_add);
}
}  // namespace holoscan
