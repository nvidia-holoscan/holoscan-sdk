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

#include "fragment_service.hpp"

#include <pybind11/pybind11.h>

#include <memory>
#include <string>

#include "fragment_service_pydoc.hpp"
#include "resource.hpp"

using std::string_literals::operator""s;  // NOLINT(misc-unused-using-decls)
using pybind11::literals::operator""_a;   // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

std::shared_ptr<Resource> PyFragmentService::resource() const {
  PYBIND11_OVERRIDE_PURE(std::shared_ptr<Resource>, FragmentService, resource);
}

void PyFragmentService::resource(const std::shared_ptr<Resource>& resource) {
  PYBIND11_OVERRIDE_PURE(void, FragmentService, resource, resource);
}

std::shared_ptr<Resource> PyDefaultFragmentService::resource() const {
  // Don't use PYBIND11_OVERRIDE here because resource is exposed as a property
  // The property getter will handle Python overrides
  return DefaultFragmentService::resource();
}

void PyDefaultFragmentService::resource(const std::shared_ptr<Resource>& resource) {
  // Don't use PYBIND11_OVERRIDE here because resource is exposed as a property
  // The property setter will handle Python overrides
  DefaultFragmentService::resource(resource);
}

void init_fragment_service(py::module_& m) {
  py::class_<FragmentService, PyFragmentService, std::shared_ptr<FragmentService>>(
      m, "FragmentService", doc::FragmentService::doc_FragmentService)
      .def(py::init<>(), doc::FragmentService::doc_FragmentService_default)
      .def_property("resource",
                    py::overload_cast<>(&FragmentService::resource, py::const_),
                    py::overload_cast<const std::shared_ptr<Resource>&>(&FragmentService::resource),
                    doc::FragmentService::doc_resource);

  py::class_<DefaultFragmentService,
             PyDefaultFragmentService,
             FragmentService,
             std::shared_ptr<DefaultFragmentService>>(
      m, "DefaultFragmentService", doc::DefaultFragmentService::doc_DefaultFragmentService)
      .def(py::init<>(), doc::DefaultFragmentService::doc_DefaultFragmentService_default)
      .def(py::init<const std::shared_ptr<Resource>&>(),
           "resource"_a,
           doc::DefaultFragmentService::doc_DefaultFragmentService_resource);
}

}  // namespace holoscan
