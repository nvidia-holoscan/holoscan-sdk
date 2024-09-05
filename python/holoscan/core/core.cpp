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

#include "arg.hpp"
#include "core.hpp"
#include "execution_context.hpp"
#include "io_context.hpp"
#include "io_spec.hpp"
#include "kwarg_handling.hpp"
#include "tensor.hpp"
#include "operator.hpp"

namespace py = pybind11;

namespace holoscan {

PYBIND11_MODULE(_core, m) {
  m.doc() = R"pbdoc(
        Holoscan SDK Core Python Bindings
        ---------------------------------
        .. currentmodule:: _core
    )pbdoc";

  init_arg(m);
  init_kwarg_handling(m);
  init_component(m);
  init_condition(m);
  init_resource(m);
  init_io_context(m);
  init_execution_context(m);
  init_io_spec(m);
  init_metadata(m);
  init_operator(m);
  init_scheduler(m);
  init_network_context(m);
  init_executor(m);
  init_fragment(m);
  init_application(m);
  init_data_flow_tracker(m);
  init_tensor(m);
  init_cli(m);
}  // PYBIND11_MODULE NOLINT

}  // namespace holoscan
