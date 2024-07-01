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

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

namespace holoscan {

void init_asynchronous(py::module_&);
void init_boolean(py::module_&);
void init_count(py::module_&);
void init_periodic(py::module_&);
void init_downstream_message_affordable(py::module_&);
void init_message_available(py::module_&);
void init_expiring_message_available(py::module_&);

PYBIND11_MODULE(_conditions, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Conditions Python Bindings
        ---------------------------------------
        .. currentmodule:: _conditions
    )pbdoc";

  init_asynchronous(m);
  init_boolean(m);
  init_count(m);
  init_periodic(m);
  init_downstream_message_affordable(m);
  init_message_available(m);
  init_expiring_message_available(m);
}  // PYBIND11_MODULE
}  // namespace holoscan
