/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "execution_context.hpp"

#include <pybind11/pybind11.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "execution_context_pydoc.hpp"
#include "holoscan/core/execution_context.hpp"
#include "operator.hpp"

using pybind11::literals::operator""_a;  // NOLINT(misc-unused-using-decls)

namespace py = pybind11;

namespace holoscan {

void init_execution_context(py::module_& m) {
  // NOLINTNEXTLINE(bugprone-unused-raii)
  py::class_<ExecutionContext, std::shared_ptr<ExecutionContext>>(
      m, "ExecutionContext", doc::ExecutionContext::doc_ExecutionContext);

  py::class_<PyExecutionContext, ExecutionContext, std::shared_ptr<PyExecutionContext>>(
      m, "PyExecutionContext", R"doc(Execution context class.)doc")
      .def_property_readonly("input", &PyExecutionContext::py_input)
      .def_property_readonly("output", &PyExecutionContext::py_output)
      .def(
          "allocate_cuda_stream",
          [](ExecutionContext& context, const std::string& name) -> intptr_t {
            auto maybe_cuda_stream = context.allocate_cuda_stream(name);
            if (maybe_cuda_stream) {
              // return the memory address correspondingt to the cudaStream_t
              auto cuda_stream = maybe_cuda_stream.value();
              auto stream_ptr = reinterpret_cast<intptr_t>(static_cast<void*>(cuda_stream));
              return stream_ptr;
            }
            return 0;
          },
          "stream_name"_a = "",
          doc::ExecutionContext::doc_allocate_cuda_stream)
      .def(
          "synchronize_streams",
          [](ExecutionContext& context,
             std::vector<std::optional<intptr_t>>
                 cuda_stream_ptrs,
             intptr_t target_cuda_stream_ptr) {
            // cast memory addresses to the corresponding cudaStream_t (void* pointer)
            std::vector<std::optional<cudaStream_t>> cuda_streams;
            cuda_streams.reserve(cuda_stream_ptrs.size());
            for (const auto& stream_ptr : cuda_stream_ptrs) {
              if (stream_ptr) {
                cuda_streams.push_back(reinterpret_cast<cudaStream_t>(stream_ptr.value()));
              } else {
                cuda_streams.push_back(std::nullopt);
              }
            }
            auto target_cuda_stream = reinterpret_cast<cudaStream_t>(target_cuda_stream_ptr);
            context.synchronize_streams(cuda_streams, target_cuda_stream);
            return;
          },
          "cuda_stream_ptrs"_a,
          "target_cuda_stream_ptr"_a,
          doc::ExecutionContext::doc_synchronize_streams)
      .def(
          "device_from_stream",
          [](ExecutionContext& context, intptr_t cuda_stream_ptr) -> std::optional<int> {
            auto maybe_device =
                context.device_from_stream(reinterpret_cast<cudaStream_t>(cuda_stream_ptr));
            if (maybe_device) {
              return maybe_device.value();
            } else {
              HOLOSCAN_LOG_ERROR("Failed to get device from stream: {}",
                                 maybe_device.error().what());
              return std::nullopt;
            }
          },
          "cuda_stream_ptr"_a,
          doc::ExecutionContext::doc_device_from_stream);
}

PyExecutionContext::PyExecutionContext(gxf_context_t context,
                                       std::shared_ptr<PyInputContext>& py_input_context,
                                       std::shared_ptr<PyOutputContext>& py_output_context,
                                       py::object op)
    : gxf::GXFExecutionContext(context, py_input_context, py_output_context),
      py_op_(std::move(op)),
      py_input_context_(py_input_context),
      py_output_context_(py_output_context) {}

std::shared_ptr<PyInputContext> PyExecutionContext::py_input() const {
  return py_input_context_;
}

std::shared_ptr<PyOutputContext> PyExecutionContext::py_output() const {
  return py_output_context_;
}

}  // namespace holoscan
