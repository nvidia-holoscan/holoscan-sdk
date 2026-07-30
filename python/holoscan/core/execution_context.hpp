/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef PYHOLOSCAN_CORE_EXECUTION_CONTEXT_HPP
#define PYHOLOSCAN_CORE_EXECUTION_CONTEXT_HPP

#include <pybind11/pybind11.h>

#include <memory>

#include <holoscan/core/gxf/gxf_execution_context.hpp>
#include "io_context.hpp"

namespace py = pybind11;

namespace holoscan {

class PyOperator;

void init_execution_context(py::module_&);

class PyExecutionContext : public gxf::GXFExecutionContext {
 public:
  /* Inherit the constructors */
  using gxf::GXFExecutionContext::GXFExecutionContext;

  explicit PyExecutionContext(gxf_context_t context,
                              const std::shared_ptr<PyOperator>& py_op = nullptr);

  std::shared_ptr<PyInputContext> py_input() const;

  std::shared_ptr<PyOutputContext> py_output() const;

 protected:
  // Make PyOperator a friend so it can call protected init_cuda_object_handler
  friend class PyOperator;
  using gxf::GXFExecutionContext::clear_received_streams;
  using gxf::GXFExecutionContext::cuda_object_handler;
  using gxf::GXFExecutionContext::init_cuda_object_handler;

 private:
  std::weak_ptr<PyOperator> py_op_;
  std::shared_ptr<PyInputContext> py_input_context_;
  std::shared_ptr<PyOutputContext> py_output_context_;
};

}  // namespace holoscan

#endif /* PYHOLOSCAN_CORE_EXECUTION_CONTEXT_HPP */
