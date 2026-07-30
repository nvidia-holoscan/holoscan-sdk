/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/chrono.h>  // will include timedelta.h for us
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace holoscan {

void init_allocators(py::module_&);
void init_receivers(py::module_&);
void init_transmitters(py::module_&);
void init_clocks(py::module_&);
void init_gxf_component_resource(py::module_&);
void init_serialization_buffers(py::module_&);
void init_component_serializers(py::module_&);
void init_condition_combiners(py::module_&);
void init_entity_serializers(py::module_&);
void init_std_entity_serializer(py::module_&);
void init_system_resources(py::module_&);

PYBIND11_MODULE(_resources, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Resources Python Bindings
        --------------------------------------
        .. currentmodule:: _resources
    )pbdoc";

  init_allocators(m);
  init_receivers(m);
  init_transmitters(m);
  init_clocks(m);
  init_gxf_component_resource(m);
  init_serialization_buffers(m);
  init_component_serializers(m);
  init_condition_combiners(m);
  init_entity_serializers(m);
  init_std_entity_serializer(m);
  init_system_resources(m);
}  // PYBIND11_MODULE
}  // namespace holoscan
