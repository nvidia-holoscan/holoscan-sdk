/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOSCAN_CORE_RESOURCE_HPP
#define PYHOLOSCAN_CORE_RESOURCE_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>

#include "holoscan/core/component_spec.hpp"
#include "holoscan/core/resource.hpp"

namespace py = pybind11;

namespace holoscan {

// Forward declarations
class Fragment;
class PyComponentSpec;

class PYBIND11_EXPORT PyResource : public Resource {
 public:
  /* Inherit the constructors */
  using Resource::Resource;

  // Define a kwargs-based constructor that can create an ArgList
  // for passing on to the variadic-template based constructor.
  PyResource(py::object resource, Fragment* fragment, const py::args& args,
             const py::kwargs& kwargs);

  // Override spec() method
  std::shared_ptr<PyComponentSpec> py_shared_spec();

  /* Trampolines (need one for each virtual function) */
  void initialize() override;
  void setup(ComponentSpec& spec) override;

 private:
  py::object py_resource_ = py::none();
};

void init_resource(py::module_& m);

}  // namespace holoscan

#endif  // PYHOLOSCAN_CORE_RESOURCE_HPP
