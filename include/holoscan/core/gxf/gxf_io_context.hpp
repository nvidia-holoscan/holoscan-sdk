/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_GXF_GXF_IO_CONTEXT_HPP
#define HOLOSCAN_CORE_GXF_GXF_IO_CONTEXT_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../expected.hpp"
#include "../io_context.hpp"
#include "./gxf_cuda.hpp"
#include "gxf/core/handle.hpp"
#include "gxf/std/receiver.hpp"

namespace holoscan::gxf {

nvidia::gxf::Receiver* get_gxf_receiver(const std::shared_ptr<IOSpec>& input_spec);

/**
 * @brief Class to hold the input context for a GXF Operator.
 *
 * This class provides the interface to receive the input data from the operator using GXF.
 */
class GXFInputContext : public InputContext {
 public:
  /**
   * @brief Construct a new GXFInputContext object.
   *
   * @param execution_context The pointer to the execution context.
   * @param op The pointer to the GXFOperator object.
   */
  GXFInputContext(ExecutionContext* execution_context, Operator* op);

  /**
   * @brief Construct a new GXFInputContext object
   *
   * @param execution_context The pointer to the execution context.
   * @param op The pointer to the GXFOperator object.
   * @param inputs inputs The references to the map of the input specs.
   */
  GXFInputContext(ExecutionContext* execution_context, Operator* op,
                  std::unordered_map<std::string, std::shared_ptr<IOSpec>>& inputs);

  /**
   * @brief Get a pointer to the GXF execution runtime.
   * @return The pointer to the GXF context.
   */
  gxf_context_t gxf_context() const;

  /** @brief Synchronize any streams found on this port to the operator's internal CUDA stream.
   *
   * The `receive` method must have been called for `input_port_name` prior to calling this method
   * in order for any received streams to be found. This method will call `cudaSetDevice` to make
   * the device corresponding to the operator's internal stream current.
   *
   * If no `CudaStreamPool` resource was available on the operator, the operator will not have an
   * internal stream. In that case, the first stream received on the input port will be returned
   * and any additional streams on the input will have been synchronized to it. If no streams were
   * found on the input and no `CudaStreamPool` resource was available, `cudaStreamDefault` is
   * returned.
   *
   * @param input_port_name The name of the input port. Can be omitted if the operator only has a
   * single input port.
   * @param allocate Whether to allocate a new stream if no stream is found. If false or the
   * operator does not have a `cuda_stream_pool` parameter set, returns cudaStreamDefault.
   * @param sync_to_default Whether to also synchronize any received streams to the default stream.
   * @returns The operator's internal CUDA stream, when possible. Returns `cudaStreamDefault`
   * instead if no CudaStreamPool resource was available and no stream was found on the input port.
   */
  cudaStream_t receive_cuda_stream(const char* input_port_name = nullptr, bool allocate = true,
                                   bool sync_to_default = false) override;

  /** @brief Retrieve the CUDA streams found an input port.
   *
   * This method is intended for advanced use cases where it is the users responsibility to
   * manage any necessary stream synchronization. In most cases, it is recommended to use
   * `receive_cuda_stream` instead.
   *
   * @param input_port_name The name of the input port. Can be omitted if the operator only has a
   * single input port.
   * @returns Vector of (optional) cudaStream_t. The length of the vector will match the number of
   * messages on the input port. Any messages that do not contain a stream will have value of
   * std::nullopt.
   */
  std::vector<std::optional<cudaStream_t>> receive_cuda_streams(
      const char* input_port_name = nullptr) override;

 protected:
  bool empty_impl(const char* name = nullptr) override;
  std::any receive_impl(const char* name = nullptr, InputType in_type = InputType::kAny,
                        bool no_error_message = false) override;

  gxf_result_t retrieve_cuda_streams(nvidia::gxf::Entity& message, const std::string& input_name);

  std::shared_ptr<gxf::CudaObjectHandler> gxf_cuda_object_handler() {
    return std::dynamic_pointer_cast<gxf::CudaObjectHandler>(cuda_object_handler_);
  }
};

/**
 * @brief Class to hold the output context for a GXF Operator.
 *
 * This class provides the interface to send data to the output ports of the operator using GXF.
 */
class GXFOutputContext : public OutputContext {
 public:
  /**
   * @brief Construct a new GXFOutputContext object.
   *
   * @param execution_context The pointer to the execution context.
   * @param op The pointer to the GXFOperator object.
   */
  GXFOutputContext(ExecutionContext* execution_context, Operator* op);

  /**
   * @brief Construct a new GXFOutputContext object
   *
   * @param execution_context The pointer to the execution context.
   * @param op The pointer to the GXFOperator object.
   * @param outputs outputs The references to the map of the output specs.
   */
  GXFOutputContext(ExecutionContext* execution_context, Operator* op,
                   std::unordered_map<std::string, std::shared_ptr<IOSpec>>& outputs);

  /**
   * @brief Get pointer to the GXF execution runtime.
   * @return The pointer to the GXF context.
   */
  gxf_context_t gxf_context() const;

  /**
   * @brief Set a stream to be emitted on a given output port.
   *
   * The actual creation of the stream component in the output message will occur on any subsequent
   * `emit` calls on this output port, so the call to this function should occur prior to the
   * `emit` call(s) for a given port.
   *
   * @param stream The CUDA stream
   * @param output_port_name The name of the output port.
   */
  void set_cuda_stream(const cudaStream_t stream, const char* output_port_name = nullptr) override;

 protected:
  void emit_impl(std::any data, const char* name = nullptr, OutputType out_type = OutputType::kAny,
                 const int64_t acq_timestamp = -1) override;

  std::shared_ptr<gxf::CudaObjectHandler> gxf_cuda_object_handler() {
    return std::dynamic_pointer_cast<gxf::CudaObjectHandler>(cuda_object_handler_);
  }

 private:
  void populate_output_metadata(nvidia::gxf::Handle<MetadataDictionary> metadata);
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_IO_CONTEXT_HPP */
