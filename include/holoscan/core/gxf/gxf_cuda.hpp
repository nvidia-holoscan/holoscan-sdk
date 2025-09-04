/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_GXF_GXF_CUDA_HPP
#define HOLOSCAN_CORE_GXF_GXF_CUDA_HPP

#include <gxf/core/gxf.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../cuda_object_handler.hpp"
#include "../operator_spec.hpp"
#include "../parameter.hpp"
#include "../resources/gxf/cuda_stream_pool.hpp"
#include "../resources/gxf/cuda_green_context.hpp"
#include "../resources/gxf/cuda_green_context_pool.hpp"
#include "gxf/cuda/cuda_stream.hpp"
#include "gxf/cuda/cuda_stream_id.hpp"
#include "gxf/cuda/cuda_stream_pool.hpp"

namespace holoscan::gxf {

using CudaStreamId = nvidia::gxf::CudaStreamId;
using CudaStream = nvidia::gxf::CudaStream;
using CudaStreamHandle = nvidia::gxf::Handle<nvidia::gxf::CudaStream>;

/**
 * This class handles usage of CUDA streams for operators.
 *
 * When using CUDA operations the default stream '0' synchronizes with all other streams in the same
 * context, see
 * https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html#stream-sync-behavior.
 * This can reduce performance. The CudaObjectHandler class manages CUDA streams and events across
 * operators and makes sure that CUDA operations are properly chained.
 *
 * Usage:
 * - This class is automatically added as an internal data member of each operators
 * `ExecutionContext`. It will be automatically configured by
 *  `ExecutionContext::init_cuda_object_handler(op)` by `GXFWrapper::start()`, just before
 * `Operator::start` is called.
 * - A stream pool for use by `CudaObjectHandler` can be added to the operator either by explicitly
 * adding a parameter with type `std::shared_ptr<CudaStreamPool>` and name `cuda_stream_pool` or by
 * passing an `Arg<std::shared_ptr<CudaStreamPool>>` to `Fragment::make_operator` when creating the
 * operator. It is not required to provide a stream pool, but allocation of an internal stream or
 * allocation of additional streams via `allocate_cuda_stream` is only possible if a stream pool is
 * present.
 * - This class is not intended for direct use by Application authors, but instead to support the
 * public methods available on `InputContext`, `OutputContext` and `ExecutionContext` as described
 * below.
 * - When the `InputContext::receive` method is called for a given port, the operator's
 * `CudaObjectHandler` class will update its internal mapping of the streams available on the input
 * ports.
 * - When `InputContext::receive_cuda_stream` is called, any received streams found by the prior
 * `receive` call for the specified port will be synchronized to the operator's internal stream and
 * then that internal stream will returned as a standard CUDA Runtime API `cudaStream_t`. If no
 * `CudaStreamPool` was configured, it will not be possible to create the internal stream, so in
 * that case, the first CUDA stream found on the input will be returned and any remaining streams
 * on the input are synchronized to it. If there are no streams on the input port and there is no
 * internal `CudaStreamPool`, then `cudaStreamDefault` is returned. When a non-default stream is
 * returned, this method calls `cudaSetDevice` to set the active device to match the stream that is
 * returned. When a non-default stream is returned, this method also will have automatically
 * configured the output ports of the operator to emit that stream, so manually calling
 * `OutputContext::set_cuda_stream` is not necessary when using this method.
 * - The `InputContext::receive_cuda_streams` method is intended for advanced use cases where the
 * user wants to handle all streams found and their synchronization manually. It just returns a
 * `vector<std::optional<cudaStream_t>>` where the size of the vector is equal to the number of
 * messages found on the input port. Any messages without a stream will have a `std::nullopt` entry
 * in the vector.
 * - The `ExecutionContext::allocate_cuda_stream` method can be used if it is necessary to allocate
 * an additional stream for use by the operator. In most cases, this will not be necessary and the
 * stream that is returned by `InputContext::receive_cuda_stream` can be used.
 * - The `ExecutionContext::device_from_stream` method can be used to determined which CUDA device
 * id a given `cudaStream_t` returned by `InputContext::receive_cuda_stream` or
 * `InputContext::receive_cuda_streams` belongs to.
 * - The `OutputContext::set_cuda_stream` method can be used to emit specific streams on specific
 * output ports. Any non-default stream received by `InputContext::receive_cuda_stream` would
 * already automatically be output, so this method is mainly useful if doing manual management of
 * the streams received via `InputContext::receive_cuda_streams` or if additional internal streams
 * were allocated via `ExecutionContext::allocate_cuda_stream`.
 */
class CudaObjectHandler : public holoscan::CudaObjectHandler {
 public:
  /**
   * @brief Destroy the CudaObjectHandler object
   */
  ~CudaObjectHandler() override;

  /**
   * @brief Use a CudaStreamPool from the specified Operator if one is present.
   *
   * @param op : The operator this instance of CudaObjectHandler is attached to. This operator must
   * have already been initialized.
   */
  void init_from_operator(Operator* op) override;

  /**
   * Add stream to output port (must be called before any emit call using that port)
   *
   * @param stream_handle The stream to add
   * @param output_port_name The name of the output port
   * @return gxf_result_t
   */
  gxf_result_t add_stream(const CudaStreamHandle& stream_handle,
                          const std::string& output_port_name);

  /**
   * Add stream to output port (must be called before any emit call using that port)
   *
   * @param stream The stream to add
   * @param output_port_name The name of the output port
   * @return gxf_result_t
   */
  int add_stream(const cudaStream_t stream, const std::string& output_port_name) override;

  /**
   * @brief Get the CUDA stream handle which should be used for CUDA commands involving data
   * from the specified input port.
   *
   * For multi-receivers or input ports with queue size > 1, the first stream found is returned
   * after any remaining streams are synchronized to it.
   *
   * See `get_cuda_stream_handles()` instead to receive a vector of (optional) CUDA stream handles
   * (one for each message).
   *
   * If no message stream is set and the `allocate` flag is true, a stream will be allocated
   * from the internal CudaStreamPool. Only if this allocation fails, would an unexpected be
   * returned.
   *
   * @param context The GXF context of the operator.
   * @param input_port_name The name of the input port from which to retrieve the stream.
   * @param allocate If true, allocate a new stream via a cuda_stream_pool parameter if
   * no stream is found.
   * @param sync_to_default If true, synchronize any streams to the default stream. If false,
   * synchronization is done to the internal stream instead.
   * @return CudaStreamHandle
   */
  expected<CudaStreamHandle, RuntimeError> get_cuda_stream_handle(
      gxf_context_t context, const std::string& input_port_name, bool allocate = true,
      bool sync_to_default = false);

  /**
   * @brief Get the CUDA stream handles which should be used for CUDA commands involving data from
   * the specified input port.
   *
   * The size of the vector returned will be equal to the number of messages received on the input
   * port. Any messages which did not contain a stream will result in a std::nullopt in the vector.
   *
   * @param context The GXF context of the operator.
   * @param input_port_name The name of the input port from which to retrieve the stream.
   * @return vector<std::optional<CudaStreamHandle>>
   */
  expected<std::vector<std::optional<CudaStreamHandle>>, RuntimeError> get_cuda_stream_handles(
      gxf_context_t context, const std::string& input_port_name);

  /**
   * @brief Get the CUDA stream which should be used for CUDA commands involving data from
   * the specified input port.
   *
   * For multi-receivers or input ports with queue size > 1, see `get_cuda_streams()` instead
   * to receive a vector of CUDA streams (one for each message).
   *
   * If no message stream is set and no stream can be allocated from the internal CudaStreamPool,
   * returns CudaStreamDefault.
   *
   * @param context The GXF context of the operator.
   * @param input_port_name The name of the input port from which to retrieve the stream
   * @param allocate If true, allocate a new stream via a cuda_stream_pool parameter if
   * none is found on the input port. Otherwise, cudaStreamDefault will be returned.
   * @param sync_to_default If true, synchronize any streams to the default stream. If false,
   * synchronization is done to the first stream found on the port instead.
   * @return cudaStream_t
   */
  cudaStream_t get_cuda_stream(void* context, const std::string& input_port_name,
                               bool allocate = false, bool sync_to_default = true) override;

  /**
   * @brief Get the CUDA stream which should be used for CUDA commands involving data from
   * the specified input port.
   *
   * The size of the vector returned will be equal to the number of messages received on the input
   * port. Any messages which did not contain a stream will result in a cudaStreamDefault in the
   * vector.
   *
   * @param context The GXF context of the operator.
   * @param input_port_name The name of the input port from which to retrieve the stream
   * @return vector<std::optional<cudaStream_t>>
   */
  std::vector<std::optional<cudaStream_t>> get_cuda_streams(
      void* context, const std::string& input_port_name) override;

  /**
   * @brief Sync all streams in stream_handles with target_stream_handle.
   *
   * Any streams in stream_handles that are not valid will be ignored.
   *
   * @param stream_handles The vector of streams to sync.
   * @param target_stream_handle The stream to sync to.
   * @param sync_to_default_stream If true, also synchronize the target stream to the default stream
   * @return gxf_result_t GXF_SUCCESS if all streams were successfully synced.
   */
  gxf_result_t synchronize_streams(std::vector<std::optional<CudaStreamHandle>> stream_handles,
                                   CudaStreamHandle target_stream_handle,
                                   bool sync_to_default_stream = true);

  /**
   * @brief Sync all streams in stream_handles with target_stream_handle.
   *
   * Any streams in stream_handles that are not valid will be ignored.
   *
   * @param cuda_streams The vector of streams to sync.
   * @param target_stream The stream to sync to.
   * @param sync_to_default_stream If true, also synchronize the target stream to the default stream
   * @return int 0 if all streams were successfully synced, otherwise an error code
   */
  int synchronize_streams(std::vector<cudaStream_t> cuda_streams, cudaStream_t target_stream,
                          bool sync_to_default_stream = true) override;

  /**
   * @brief Get the cudaStream_t value corresponding to a CudaStreamHandle
   *
   * @param stream_handle The CudaStreamHandle
   * @return The CUDA stream contained within the CudaStream object
   */
  cudaStream_t stream_from_stream_handle(CudaStreamHandle stream_handle);

  /**
   * @brief Get the CudaStreamHandle corresponding to a cudaStream_t
   *
   * @param stream The CUDA stream
   * @return GXF Handle to the CudaStream object if found, otherwise an unexpected is returned.
   */
  expected<CudaStreamHandle, RuntimeError> stream_handle_from_stream(cudaStream_t stream);

  /**
   * @brief Get the GXF component ID for any stream to be emitted on the specified output port
   *
   * @param output_port_name The name of the output port
   * @return expected<gxf_uid_t>
   */
  expected<gxf_uid_t, ErrorCode> get_output_stream_cid(const std::string& output_port_name);

  /**
   * @brief Get the GXF component IDs for any events to be emitted on the specified output port
   *
   * @param context The GXF context
   * @param message The GXF message entity
   * @param input_name The name of the input port
   * @return expected<std::vector<gxf_uid_t>>
   */
  gxf_result_t streams_from_message(gxf_context_t context, const nvidia::gxf::Entity& message,
                                    const std::string& input_name);

  /**
   * Allocate an internal CUDA stream and store it in the mapping for the given input port
   *
   * @param context The GXF context
   * @param stream_name The name of the stream
   * @return GXF Handle to the allocated CudaStream component
   */
  expected<CudaStreamHandle, RuntimeError> allocate_internal_stream(gxf_context_t context,
                                                                    const std::string& stream_name);

  /// @brief Release all internally allocated CUDA streams
  int release_internal_streams(void* context) override;

  /** @brief Retain the existing unordered_maps and vectors of received streams, but clear the
   * contents.
   *
   * This is used to refresh the state of the received streams before each `Operator::compute` call.
   */
  void clear_received_streams() override;

 private:
  /// @brief allocate a new stream from the internal stream pool
  expected<CudaStreamHandle, RuntimeError> allocate_cuda_stream(gxf_context_t context);

  gxf_result_t from_messages(gxf_context_t context, size_t message_count,
                             const nvidia::gxf::Entity* messages);

  /// CUDA stream pool used to allocate the internal CUDA stream
  Parameter<std::shared_ptr<CudaStreamPool>> cuda_stream_pool_;

  /// CUDA green context used to create cuda stream pool
  Parameter<std::shared_ptr<CudaGreenContext>> cuda_green_context_;

  /// CUDA green context pool used to allocate the internal CUDA stream using green context
  /// partitions.
  Parameter<std::shared_ptr<CudaGreenContextPool>> cuda_green_context_pool_;

  expected<nvidia::gxf::Handle<nvidia::gxf::CudaStreamPool>, RuntimeError> cuda_stream_pool_handle(
      gxf_context_t context);

  /// If the CUDA stream pool is not set and we can't use the incoming CUDA stream, issue
  /// a warning once.
  bool default_stream_warning_ = false;

  /// CUDA event used to synchronize the internal CUDA stream with multiple incoming streams
  cudaEvent_t cuda_event_ = 0;

  /// Flag to indicate if the internal CUDA event has been created yet
  bool event_created_ = false;

  /// Mapping from input port name to any CUDA stream found in the incoming Message
  std::unordered_map<std::string, std::vector<std::optional<CudaStreamId>>>
      received_cuda_stream_ids_{};
  std::unordered_map<std::string, std::vector<std::optional<CudaStreamHandle>>>
      received_cuda_stream_handles_{};

  /// Allocated internal CUDA stream handles
  /// Mapping from input port name to an internally allocated CUDA stream.
  /// (This will be populated if a stream is requested for an input, but none was found on the
  /// incoming message)
  std::unordered_map<std::string, CudaStreamHandle> allocated_cuda_stream_handles_{};

  /// Mapping from output port name to the GXF Component Id of any stream to be emitted on that port
  std::unordered_map<std::string, gxf_uid_t> emitted_cuda_stream_cids_{};

  // If we want to allow emitting via a cudaStream_t, we need a way to get the handle it
  // corresponds to.
  std::unordered_map<cudaStream_t, CudaStreamHandle> stream_to_stream_handle_{};
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_CUDA_HPP */
