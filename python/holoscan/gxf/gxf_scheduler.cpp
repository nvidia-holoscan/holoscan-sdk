/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>

#include "gxf_scheduler_pydoc.hpp"

#include "holoscan/core/gxf/gxf_component.hpp"
#include "holoscan/core/gxf/gxf_scheduler.hpp"
#include "holoscan/core/resources/gxf/clock.hpp"

#include "gxf/core/gxf.h"

namespace py = pybind11;

namespace holoscan {

class PyGXFScheduler : public gxf::GXFScheduler {
 public:
  /* Inherit the constructors */
  using gxf::GXFScheduler::GXFScheduler;

  /* Trampolines (need one for each virtual function) */
  const char* gxf_typename() const override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE_PURE(const char*, gxf::GXFScheduler, gxf_typename);
  }
  std::shared_ptr<Clock> clock() override {
    /* <Return type>, <Parent Class>, <Name of C++ function>, <Argument(s)> */
    PYBIND11_OVERRIDE_PURE(std::shared_ptr<Clock>, gxf::GXFScheduler, clock);
  }
};

void init_gxf_scheduler(py::module_& m) {
  py::class_<gxf::GXFScheduler,
             PyGXFScheduler,
             Scheduler,
             gxf::GXFComponent,
             std::shared_ptr<gxf::GXFScheduler>>(
      m, "GXFScheduler", doc::GXFScheduler::doc_GXFScheduler)
      .def("initialize", &gxf::GXFScheduler::initialize, doc::GXFScheduler::doc_initialize)
      .def_property_readonly("clock", &gxf::GXFScheduler::clock);
}  // PYBIND11_MODULE

}  // namespace holoscan
