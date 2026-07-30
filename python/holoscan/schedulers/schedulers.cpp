/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>

#include <cstdint>
#include <memory>
#include <string>

#include <holoscan/core/component_spec.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/gxf/gxf_component.hpp>
#include <holoscan/core/gxf/gxf_scheduler.hpp>
#include <holoscan/core/resources/gxf/manual_clock.hpp>
#include <holoscan/core/resources/gxf/realtime_clock.hpp>
#include <holoscan/core/schedulers/gxf/greedy_scheduler.hpp>
#include <holoscan/core/schedulers/gxf/multithread_scheduler.hpp>

namespace py = pybind11;

namespace holoscan {

void init_event_based_scheduler(py::module_&);
void init_greedy_scheduler(py::module_&);
void init_multithread_scheduler(py::module_&);

PYBIND11_MODULE(_schedulers, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Schedulers Python Bindings
        ---------------------------------------
        .. currentmodule:: _schedulers
    )pbdoc";

  init_event_based_scheduler(m);
  init_greedy_scheduler(m);
  init_multithread_scheduler(m);
}  // PYBIND11_MODULE
}  // namespace holoscan
