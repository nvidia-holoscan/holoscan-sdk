/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace py = pybind11;

namespace holoscan {

void init_asynchronous(py::module_&);
void init_boolean(py::module_&);
void init_count(py::module_&);
void init_cuda_buffer_available(py::module_&);
void init_cuda_event(py::module_&);
void init_cuda_stream(py::module_&);
void init_periodic(py::module_&);
void init_downstream_message_affordable(py::module_&);
void init_memory_available(py::module_&);
void init_message_available(py::module_&);
void init_multi_message_available(py::module_&);
void init_multi_message_available_timeout(py::module_&);
void init_expiring_message_available(py::module_&);
void init_publisher_available(py::module_&);
void init_subscriber_available(py::module_&);
#ifdef HOLOSCAN_PYTHON_HAS_PENDING_EXPORT_CONDITION
void init_pending_export(py::module_&);
#endif

PYBIND11_MODULE(_conditions, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Conditions Python Bindings
        ---------------------------------------
        .. currentmodule:: _conditions
    )pbdoc";

  init_asynchronous(m);
  init_boolean(m);
  init_count(m);
  init_cuda_buffer_available(m);
  init_cuda_event(m);
  init_cuda_stream(m);
  init_periodic(m);
  init_downstream_message_affordable(m);
  init_memory_available(m);
  init_message_available(m);
  init_multi_message_available(m);
  init_multi_message_available_timeout(m);
  init_expiring_message_available(m);
  init_publisher_available(m);
  init_subscriber_available(m);
#ifdef HOLOSCAN_PYTHON_HAS_PENDING_EXPORT_CONDITION
  init_pending_export(m);
#endif
}  // PYBIND11_MODULE
}  // namespace holoscan
