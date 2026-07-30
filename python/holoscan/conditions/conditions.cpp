/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
