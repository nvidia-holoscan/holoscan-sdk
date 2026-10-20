/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <vector>

#include <holoscan/authoring/registry.hpp>
#include "binding_services.hpp"
#include "callback_ownership.hpp"
#include "channel.hpp"
#include "clocks.hpp"
#include "config.hpp"
#include "errors.hpp"
#include "execution_authoring.hpp"
#include "graph.hpp"
#include "lifecycle_authoring.hpp"
#include "native_io.hpp"
#include "native_operator.hpp"
#include "payload.hpp"
#include "plan.hpp"
#include "py_object_payload.hpp"
#include "registry.hpp"
#include "sensor_io.hpp"
#include "session.hpp"
#include "temporal.hpp"
#include "tensor.hpp"
#include "tensor_authoring.hpp"
#include "time.hpp"
#include "topics.hpp"

// NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays)
NB_MODULE(_core, module) {
  module.doc() = "Holoscan SDK core Python bindings.";
  holoscan::init_errors(module);
  holoscan::init_time(module);
  holoscan::init_clocks(module);
  holoscan::init_py_object_payload(module);
  holoscan::init_payload(module);
  holoscan::init_lifecycle_authoring(module);
  holoscan::init_native_operator(module);
  holoscan::init_native_io(module);
  holoscan::init_sensor_io(module);
  holoscan::init_execution_authoring(module);
  holoscan::init_temporal(module);
  holoscan::init_tensor_authoring(module);
  holoscan::init_topics(module);
  holoscan::init_callback_ownership(module);
  holoscan::init_graph(module);
  holoscan::init_config(module);
  holoscan::init_plan(module);
  holoscan::init_channel(module);
  holoscan::init_registry(module);
  // atexit is LIFO: stop sessions before joining the imported-Tensor release worker.
  holoscan::init_tensor(module);
  holoscan::init_session(module);
  module.def("_registered_operator_names", [] {
    std::vector<std::string> names;
    for (const auto& row : holoscan::authoring::authoring_registry().operators()) {
      names.emplace_back(row.get().name);
    }
    return names;
  });
  holoscan::python::init_binding_services(module);
}
// NOLINTEND(cppcoreguidelines-avoid-c-arrays)
