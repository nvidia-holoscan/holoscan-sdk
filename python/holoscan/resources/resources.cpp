/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <pybind11/chrono.h>  // will include timedelta.h for us

namespace py = pybind11;

namespace holoscan {

void init_allocators(py::module_&);
void init_receivers(py::module_&);
void init_transmitters(py::module_&);
void init_clocks(py::module_&);
void init_gxf_component_resource(py::module_&);
void init_serialization_buffers(py::module_&);
void init_component_serializers(py::module_&);
void init_entity_serializers(py::module_&);
void init_std_entity_serializer(py::module_&);

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
  init_entity_serializers(m);
  init_std_entity_serializer(m);
}  // PYBIND11_MODULE
}  // namespace holoscan
