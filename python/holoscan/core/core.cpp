/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <pybind11/pybind11.h>

#include "arg.hpp"
#include "clock.hpp"
#include "core.hpp"
#include "data_logger.hpp"
#include "execution_context.hpp"
#include "fragment_service.hpp"
#include "io_context.hpp"
#include "io_spec.hpp"
#include "kwarg_handling.hpp"
#include "messagelabel.hpp"
#include "operator.hpp"
#include "subgraph.hpp"
#include "tensor.hpp"

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
  init_clock(m);
  init_data_logger(m);
  init_io_context(m);
  init_execution_context(m);
  init_io_spec(m);
  init_metadata(m);
  init_messagelabel(m);
  init_operator(m);
  init_scheduler(m);
  init_network_context(m);
  init_executor(m);
  init_fragment(m);
  init_subgraph(m);
  init_application(m);
  init_data_flow_tracker(m);
  init_tensor(m);
  init_cli(m);
  init_fragment_service(m);
}  // PYBIND11_MODULE NOLINT

}  // namespace holoscan
