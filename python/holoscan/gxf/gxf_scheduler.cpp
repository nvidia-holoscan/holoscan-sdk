/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>

#include <memory>

#include "gxf_scheduler_pydoc.hpp"

#include <holoscan/core/gxf/gxf_component.hpp>
#include <holoscan/core/gxf/gxf_scheduler.hpp>
#include <holoscan/core/resources/gxf/clock.hpp>

#include <gxf/core/gxf.h>

namespace py = pybind11;

namespace holoscan {

void init_gxf_scheduler(py::module_& m) {
  py::class_<gxf::GXFScheduler, Scheduler, gxf::GXFComponent, std::shared_ptr<gxf::GXFScheduler>>(
      m, "GXFScheduler", doc::GXFScheduler::doc_GXFScheduler);
}  // PYBIND11_MODULE

}  // namespace holoscan
