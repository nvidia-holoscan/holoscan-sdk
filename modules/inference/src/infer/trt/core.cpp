/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Array subscript access to TensorRT dimension arrays is performance-critical for inference.
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)

#include "core.hpp"

#include <NvInferPlugin.h>

#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <holoinfer/holoinfer_utils.hpp>
#include "manager/fallback_runtime.hpp"

namespace holoscan {
namespace inference {

TrtInfer::TrtInfer(const std::string& model_path, const std::string& model_name,
                   const std::vector<std::string>& trt_opt_profile, int device_id, int device_id_dt,
                   bool enable_fp16, bool enable_cuda_graphs, int32_t dla_core,
                   bool dla_gpu_fallback, bool is_engine_path, bool cuda_buf_in, bool cuda_buf_out,
                   std::function<cudaStream_t(int32_t device_id)> allocate_cuda_stream,
                   CUcontext build_cuda_context, int32_t build_sm_count, int32_t max_batch_size,
                   std::size_t max_memory)
    : model_path_(model_path),
      model_name_(model_name),
      trt_opt_profile_(trt_opt_profile),
      device_id_(device_id),
      enable_fp16_(enable_fp16),
      enable_cuda_graphs_(enable_cuda_graphs),
      dla_core_(dla_core),
      dla_gpu_fallback_(dla_gpu_fallback),
      is_engine_path_(is_engine_path),
      cuda_buf_in_(cuda_buf_in),
      cuda_buf_out_(cuda_buf_out),
      allocate_cuda_stream_(std::move(allocate_cuda_stream)) {
  auto num_of_opt_profiles = trt_opt_profile.size();

  if (num_of_opt_profiles > 0) {
    network_options_.batch_sizes.clear();
    for (const auto& opt_profile_str : trt_opt_profile) {
      // parse opt_profile here
      std::vector<std::string> tokens;
      std::vector<int32_t> opt_profile = {};
      string_split(opt_profile_str, tokens, ',');

      if (tokens.size() != 0) {
        for (const auto& t : tokens) {
          opt_profile.push_back(std::stoi(t));
        }
      }

      if (opt_profile.size() == 0 || opt_profile.size() % 3 != 0) {
        HOLOSCAN_LOG_WARN(
            "TRT Inference: Optimization profile must be a multiple of 3. Size of one of the "
            "optimization profiles from inference parameters: {}",
            opt_profile.size());
        HOLOSCAN_LOG_INFO("Invalid optimization profile ignored.");
        HOLOSCAN_LOG_INFO("TRT Engine will use the default optimization profile of {1,1,1}");
        HOLOSCAN_LOG_WARN(
            "Default optimization profile only supports dynamic batches. If this input requires "
            "dynamism in multiple dimensions, ingest the correct optimization profile.");
        network_options_.batch_sizes.push_back({1, 1, 1});
      } else {
        network_options_.batch_sizes.push_back(std::move(opt_profile));
      }
    }
  } else {
    HOLOSCAN_LOG_INFO("TRT Inference: Using the default optimization profile for model {}",
                      model_name_);
  }

  // Set the device index
  network_options_.device_index = device_id_;

  network_options_.use_fp16 = enable_fp16_;
  network_options_.dla_core = dla_core_;
  network_options_.dla_gpu_fallback = dla_gpu_fallback_;
  network_options_.build_cuda_context = build_cuda_context;
  network_options_.build_sm_count = build_sm_count;
  network_options_.max_batch_size = max_batch_size;
  network_options_.max_memory = max_memory;
  initLibNvInferPlugins(nullptr, "");

  if (!is_engine_path_) {
    HOLOSCAN_LOG_INFO("TRT Inference: converting ONNX model at {}", model_path_);

    bool status = generate_engine_path(network_options_, model_path_, engine_path_);
    if (!status) {
      throw std::runtime_error("TRT Inference: could not generate TRT engine path.");
    }

    status = build_engine(model_path_, engine_path_, network_options_, logger_);
    if (!status) {
      HOLOSCAN_LOG_ERROR("Engine file creation failed for {}", model_path_);
      HOLOSCAN_LOG_INFO(
          "If the input path {} is an engine file, set 'is_engine_path' parameter to true in "
          "the inference settings in the application config.",
          model_path_);
      throw std::runtime_error("TRT Inference: failed to build TRT engine file.");
    }
  } else {
    engine_path_ = model_path_;
  }

  // If a CUDA stream pool is provided, try to allocate a stream from it.
  // For CUDA Graphs, only use it when it's non-blocking to avoid capture invalidation from
  // implicit synchronization with the legacy default stream.
  if (allocate_cuda_stream_) {
    cuda_stream_ = allocate_cuda_stream_(device_id_);
    // The pool-supplied stream cannot host a CUDA-Graph capture. The
    // swap itself is benign (TRT just uses its own non-blocking stream), but it silently costs
    // the caller the "everything runs on the stream I handed you" property, so it is reported
    // in the standard ladder format at INFO level alongside the legacy warning below.
    std::string blocking_stream_detail;
    if (cuda_stream_ && enable_cuda_graphs_) {
      // The legacy/default stream is not valid for stream capture.
      if (cuda_stream_ == cudaStreamDefault) {
        HOLOSCAN_LOG_WARN(
            "TRT Inference: CUDA Graphs enabled but allocated default CUDA stream; "
            "falling back to internally-created non-blocking stream.");
        blocking_stream_detail =
            "the allocated stream is the legacy/default stream, which cannot be used for CUDA "
            "stream capture";
        cuda_stream_ = nullptr;
      } else {
        unsigned int stream_flags = 0;
        cudaError_t stream_status = cudaStreamGetFlags(cuda_stream_, &stream_flags);
        if (stream_status != cudaSuccess) {
          HOLOSCAN_LOG_WARN(
              "TRT Inference: failed to query allocated CUDA stream flags ({}); "
              "falling back to internally-created non-blocking stream.",
              cudaGetErrorString(stream_status));
          blocking_stream_detail =
              std::string("the allocated stream's flags could not be queried (") +
              cudaGetErrorString(stream_status) + ")";
          // Clear sticky error state from failed stream query before continuing.
          cudaGetLastError();
          cuda_stream_ = nullptr;
        } else if ((stream_flags & cudaStreamNonBlocking) == 0U) {
          HOLOSCAN_LOG_WARN(
              "TRT Inference: CUDA Graphs enabled but allocated blocking CUDA stream; "
              "falling back to internally-created non-blocking stream.");
          blocking_stream_detail =
              "the allocated stream is a blocking stream, which implicitly synchronizes with the "
              "legacy stream and would invalidate an in-progress CUDA-Graph capture";
          cuda_stream_ = nullptr;
        }
      }
    }
    if (!blocking_stream_detail.empty()) {
      emit_runtime_fallback(
          FallbackReason::kBlockingStreamSwapped,
          fallback::DispatchKind::kFast,
          fallback::CaptureProtectionOutcome::kNotApplicable,
          fallback::CaptureProtectionOutcome::kNotApplicable,
          "model '" + model_name_ +
              "': pool-supplied CUDA stream auto-swapped for an "
              "internally-created non-blocking stream because " +
              blocking_stream_detail,
          "create the cuda_stream_pool's streams with cudaStreamNonBlocking, or set "
          "enable_cuda_graphs=false if the caller's stream must be used verbatim");
    }
  }

  // If no stream could be allocated, create a new one
  if (!cuda_stream_) {
    check_cuda(cudaSetDevice(device_id_));
    // Create the CUDA stream with the non-blocking flags set. This is needed for CUDA stream
    // capturing since capturing fails if another thread is scheduling work to stream '0' while
    // we capture in this thread. We explicitly synchronize with the caller using events so stream
    // '0' does not need to sync with us.
    check_cuda(cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking));
    owns_cuda_stream_ = true;
  }
  // create the CUDA event used to synchronize with the caller
  check_cuda(cudaEventCreateWithFlags(&cuda_event_, cudaEventDisableTiming));

  bool status = load_engine();
  if (!status) {
    throw std::runtime_error("TRT Inference: failed to load TRT engine file.");
  }

  status = initialize_parameters();
  if (!status) {
    throw std::runtime_error("TRT Inference: Initialization error.");
  }
}

TrtInfer::~TrtInfer() {
  if (context_) {
    context_.reset();
  }
  if (engine_) {
    engine_.reset();
  }
  // Only destroy the stream if we created it. Streams provided by the allocation callback
  // are owned by the caller (e.g., CudaStreamPool) and will be destroyed during pool
  // deinitialization.
  if (cuda_stream_ && owns_cuda_stream_) {
    cudaStreamDestroy(cuda_stream_);
  }
  if (cuda_event_) {
    cudaEventDestroy(cuda_event_);
  }
  if (cuda_graph_instance_) {
    cudaGraphExecDestroy(cuda_graph_instance_);
  }
  if (infer_runtime_) {
    infer_runtime_.reset();
  }
}

bool TrtInfer::load_engine() {
  HOLOSCAN_LOG_INFO("Loading Engine: {}", engine_path_);
  std::ifstream file(engine_path_, std::ios::binary | std::ios::ate);
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  if (!file.read(buffer.data(), size)) {
    HOLOSCAN_LOG_ERROR("Load Engine: File read error: {}", engine_path_);
    return false;
  }

  infer_runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!infer_runtime_) {
    HOLOSCAN_LOG_ERROR("Load Engine: Error in creating inference runtime.");
    return false;
  }

  // Set the device index
  auto status = cudaSetDevice(network_options_.device_index);
  if (status != 0) {
    HOLOSCAN_LOG_ERROR("Load Engine: Setting cuda device failed.");
    throw std::runtime_error("Error setting cuda device in load engine.");
  }

  if (network_options_.dla_core > -1) {
    // set the DLA core
    const int32_t available_dla_cores = infer_runtime_->getNbDLACores();
    if (network_options_.dla_core > available_dla_cores - 1) {
      HOLOSCAN_LOG_ERROR("DLA core '{}' is requested but max DLA core index is '{}'",
                         network_options_.dla_core,
                         available_dla_cores - 1);
      throw std::runtime_error("Error setting DLA core in load engine.");
    }
    infer_runtime_->setDLACore(network_options_.dla_core);
  }

  engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
      infer_runtime_->deserializeCudaEngine(buffer.data(), buffer.size()));
  if (!engine_) {
    HOLOSCAN_LOG_ERROR("Load Engine: Error in deserializing cuda engine.");
    return false;
  }

  context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  if (!context_) {
    HOLOSCAN_LOG_ERROR("Load Engine: Error in creating execution context.");
    return false;
  }

  HOLOSCAN_LOG_INFO("Engine loaded: {}", engine_path_);

  return true;
}

std::vector<std::vector<int64_t>> TrtInfer::get_input_dims() const {
  return input_dims_;
}

std::vector<std::vector<int64_t>> TrtInfer::get_output_dims() const {
  return output_dims_;
}

std::vector<holoinfer_datatype> TrtInfer::get_input_datatype() const {
  return in_data_types_;
}

std::vector<holoinfer_datatype> TrtInfer::get_output_datatype() const {
  return out_data_types_;
}

bool TrtInfer::initialize_parameters() {
  if (!engine_) {
    HOLOSCAN_LOG_ERROR("Engine is Null.");
    return false;
  }
  const int num_bindings = engine_->getNbIOTensors();

  for (int i = 0; i < num_bindings; i++) {
    auto tensor_name = engine_->getIOTensorName(i);
    auto dims = engine_->getTensorShape(tensor_name);
    auto data_type = engine_->getTensorDataType(tensor_name);

    holoinfer_datatype holoinfer_type = holoinfer_datatype::h_Float32;
    switch (data_type) {
      case nvinfer1::DataType::kFLOAT: {
        holoinfer_type = holoinfer_datatype::h_Float32;
        break;
      }
      case nvinfer1::DataType::kINT32: {
        holoinfer_type = holoinfer_datatype::h_Int32;
        break;
      }
      case nvinfer1::DataType::kINT8: {
        holoinfer_type = holoinfer_datatype::h_Int8;
        break;
      }
      case nvinfer1::DataType::kUINT8: {
        holoinfer_type = holoinfer_datatype::h_UInt8;
        break;
      }
      case nvinfer1::DataType::kHALF: {
        holoinfer_type = holoinfer_datatype::h_Float16;
        break;
      }
      case nvinfer1::DataType::kINT64: {
        holoinfer_type = holoinfer_datatype::h_Int64;
        break;
      }
      case nvinfer1::DataType::kBOOL: {
        holoinfer_type = holoinfer_datatype::h_Bool;
        break;
      }
      default: {
        HOLOSCAN_LOG_INFO(
            "TensorRT backend supports float, float16, int64, int8, int32, uint8, bool data "
            "types.");
        HOLOSCAN_LOG_ERROR("Data type not supported.");
        return false;
      }
    }

    switch (engine_->getTensorIOMode(tensor_name)) {
      case nvinfer1::TensorIOMode::kINPUT: {
        if (dims.nbDims > 8) {
          HOLOSCAN_LOG_INFO("All tensors must have dimension size less than or equal to 8.");
          return false;
        }
        nvinfer1::Dims in_dimensions;
        in_dimensions.nbDims = dims.nbDims;

        for (size_t in = 0; in < dims.nbDims; in++) {
          in_dimensions.d[in] = dims.d[in];
        }

        auto set_status = context_->setInputShape(tensor_name, in_dimensions);
        if (!set_status) {
          HOLOSCAN_LOG_ERROR("Dimension setup for input tensor {} failed.", tensor_name);
          return false;
        }

        std::vector<int64_t> indim;
        for (size_t in = 0; in < dims.nbDims; in++) {
          indim.push_back(dims.d[in]);
        }
        input_dims_.push_back(std::move(indim));

        in_data_types_.push_back(holoinfer_type);
        input_tensor_names_.push_back(tensor_name);
      } break;
      case nvinfer1::TensorIOMode::kOUTPUT: {
        std::vector<int64_t> outdim;
        for (size_t in = 0; in < dims.nbDims; in++) {
          outdim.push_back(dims.d[in]);
        }

        output_dims_.push_back(std::move(outdim));
        out_data_types_.push_back(holoinfer_type);
        output_tensor_names_.push_back(tensor_name);
      } break;
      default: {
        HOLOSCAN_LOG_ERROR("Input index {} is neither input nor output.", i);
        return false;
      }
    }
  }

  if (!context_->allInputDimensionsSpecified()) {
    HOLOSCAN_LOG_ERROR("Error, not all input dimensions specified.");
    return false;
  }

  return true;
}

bool TrtInfer::set_dynamic_input_dimension(
    const std::vector<std::string>& input_holoscan_tensors,
    const std::map<std::string, std::vector<int>>& dims_per_tensor) {
  if (!engine_) {
    HOLOSCAN_LOG_ERROR("Engine is Null.");
    return false;
  }
  const int num_bindings = engine_->getNbIOTensors();

  // Capture the previous per-input dims before we overwrite so we can detect an actual shape
  // change vs a same-dim call. Dynamic-shape callers (dynamic_input_dims=true) invoke this on
  // every tick regardless of whether dims changed; unconditional graph invalidation would flip
  // first_phase_ true every tick and defeat CUDA-graph capture entirely.
  const std::vector<std::vector<int64_t>> previous_input_dims = input_dims_;

  input_dims_.clear();
  int input_index = 0;

  for (int i = 0; i < num_bindings; i++) {
    auto tensor_name = engine_->getIOTensorName(i);

    if (engine_->getTensorIOMode(tensor_name) == nvinfer1::TensorIOMode::kINPUT) {
      const auto& holoscan_tensor_name = input_holoscan_tensors[input_index++];
      const auto& dims = dims_per_tensor.at(holoscan_tensor_name);

      if (dims.size() > 8) {
        HOLOSCAN_LOG_INFO("All tensors must have dimension size less than or equal to 8.");
        return false;
      }
      nvinfer1::Dims in_dimensions;
      in_dimensions.nbDims = dims.size();

      for (size_t in = 0; in < dims.size(); in++) {
        in_dimensions.d[in] = dims[in];
      }

      auto set_status = context_->setInputShape(tensor_name, in_dimensions);
      if (!set_status) {
        HOLOSCAN_LOG_ERROR("Dimension setup for input tensor {} failed.", tensor_name);
        return false;
      }

      std::vector<int64_t> indim;
      for (size_t in = 0; in < dims.size(); in++) {
        indim.push_back(dims[in]);
      }
      input_dims_.push_back(std::move(indim));
    }
  }

  if (!context_->allInputDimensionsSpecified()) {
    HOLOSCAN_LOG_ERROR("Error, not all input dimensions specified.");
    return false;
  }

  // Only invalidate the captured graph when the shape actually changed. On a real change,
  // the recorded kernel launches, memory bindings, and internal TRT bookkeeping are sized
  // against the old shapes; destroy the exec instance and reset first_phase_ so the next
  // call runs enqueueV3() UNCAPTURED (letting TRT do any un-captureable per-shape re-init),
  // and the call after that captures fresh against the new shapes. If dims are unchanged,
  // leave the graph state alone so steady-state replay can proceed.
  if (input_dims_ != previous_input_dims) {
    if (cuda_graph_instance_) {
      cudaGraphExecDestroy(cuda_graph_instance_);
      cuda_graph_instance_ = nullptr;
    }
    first_phase_ = true;
    graph_needs_recapture_ = true;
  }

  return true;
}

InferStatus TrtInfer::do_inference(const std::vector<std::shared_ptr<DataBuffer>>& input_buffers,
                                   std::vector<std::shared_ptr<DataBuffer>>& output_buffers,
                                   cudaEvent_t cuda_event_data, cudaEvent_t* cuda_event_inference) {
  InferStatus status = InferStatus(holoinfer_code::H_ERROR);
  auto io_index = 0;

  // synchronize the CUDA stream used for inference with the CUDA event recorded when preparing
  // the input data
  check_cuda(cudaStreamWaitEvent(cuda_stream_, cuda_event_data));

  // Input Binding Stability Cache
  if (!inputs_pinned_ && last_input_addr_.empty()) {
    last_input_addr_.assign(input_buffers.size(), nullptr);
  }
  size_t in_tensor_index = 0;

  for (auto& input_buffer : input_buffers) {
    if (input_buffer->device_buffer_ == nullptr) {
      status.set_message(" TRT inference core: Input Device buffer is null.");
      return status;
    }

    if (cuda_buf_in_) {
      if (input_buffer->device_buffer_->data() == nullptr) {
        status.set_message(" TRT inference core: Data in Input Device buffer is null.");
        return status;
      }
    } else {
      // Host to Device transfer
      if (input_buffer->host_buffer_->size() == 0) {
        status.set_message(" TRT inference core: Empty input host buffer.");
        return status;
      }

      input_buffer->device_buffer_->resize(input_buffer->host_buffer_->size());

      auto cstatus = cudaMemcpyAsync(input_buffer->device_buffer_->data(),
                                     input_buffer->host_buffer_->data(),
                                     input_buffer->host_buffer_->get_bytes(),
                                     cudaMemcpyHostToDevice,
                                     cuda_stream_);
      if (cstatus != cudaSuccess) {
        status.set_message(" TRT inference core: Host to device transfer failed.");
        return status;
      }
      // When copying from pagable memory to device memory cudaMemcpyAsync() is copying the memory
      // to staging memory first and therefore is synchronous with the host execution. No need to
      // synchronize here.
    }
    auto tensor_name = engine_->getIOTensorName(io_index++);
    if (engine_->getTensorIOMode(tensor_name) != nvinfer1::TensorIOMode::kINPUT) {
      HOLOSCAN_LOG_ERROR("Tensor name {} not an input binding.", tensor_name);
      status.set_message(" TRT inference core: Incorrect input tensor name.");
      return status;
    }
    void* cur_in_addr = input_buffer->device_buffer_->data();

    // Input Binding Stability Cache
    if (inputs_pinned_ && in_tensor_index < last_input_addr_.size() &&
        last_input_addr_[in_tensor_index] == cur_in_addr) {
      ++in_tensor_index;
      continue;
    }

    auto set_flag = context_->setTensorAddress(tensor_name, cur_in_addr);
    if (!set_flag) {
      HOLOSCAN_LOG_ERROR("Buffer binding failed for {} in inference core.", tensor_name);
      status.set_message(" TRT inference core: Error binding input buffer.");
      return status;
    }
    // Address rebind — the captured graph holds this pointer; force a recapture.
    graph_needs_recapture_ = true;
    if (in_tensor_index < last_input_addr_.size()) {
      last_input_addr_[in_tensor_index] = cur_in_addr;
    }
    ++in_tensor_index;
  }
  // Input Binding Stability Cache
  // Pin the input bindings only after a full loop completes — guards against partial
  // initialization if any input failed mid-loop.
  if (!inputs_pinned_) {
    inputs_pinned_ = true;
  }

  size_t out_tensors_index = 0;

  // Lazily size the binding-stability caches the first time we run.
  if (!outputs_pinned_ && last_output_addr_.empty()) {
    last_output_addr_.assign(output_buffers.size(), nullptr);
    last_output_dims_.assign(output_buffers.size(), {});
  }

  for (auto& output_buffer : output_buffers) {
    if (output_buffer->device_buffer_ == nullptr) {
      status.set_message(" TRT inference core: Output Device buffer is null.");
      return status;
    }
    void* cur_addr = output_buffer->device_buffer_->data();
    if (cur_addr == nullptr) {
      status.set_message(" TRT inference core: Data in Output Device buffer is null.");
      return status;
    }

    if (output_buffer->device_buffer_->size() == 0) {
      status.set_message(" TRT inference core: Output Device buffer size is 0.");
      return status;
    }
    auto tensor_name = engine_->getIOTensorName(io_index++);

    if (engine_->getTensorIOMode(tensor_name) != nvinfer1::TensorIOMode::kOUTPUT) {
      HOLOSCAN_LOG_ERROR("Tensor name {} not an output binding.", tensor_name);
      status.set_message(" TRT inference core: Incorrect output tensor name.");
      return status;
    }

    // Get the output dimensions
    nvinfer1::Dims outputDims = context_->getTensorShape(tensor_name);
    for (int ai = 0; ai < outputDims.nbDims; ai++) {
      output_dims_[out_tensors_index][ai] = outputDims.d[ai];
    }
    // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)

    size_t dynamic_buffer_size = accumulate(output_dims_[out_tensors_index].begin(),
                                            output_dims_[out_tensors_index].end(),
                                            1,
                                            std::multiplies<size_t>());

    if (dynamic_buffer_size != output_buffer->device_buffer_->size()) {
      output_buffer->device_buffer_->resize(dynamic_buffer_size);
      output_buffer->host_buffer_->resize(dynamic_buffer_size);
      // Resize changes the underlying pointer — make sure the address we re-bind below is
      // the post-resize one. Also invalidate any previously pinned state for this slot since
      // a graph captured against the old pointer would need to be re-captured.
      cur_addr = output_buffer->device_buffer_->data();
      graph_needs_recapture_ = true;
    }

    // Output Binding Stability Cache — skip only the setTensorAddress call
    // when the bound address is unchanged; shape/size refresh above still ran.
    if (outputs_pinned_ && last_output_addr_[out_tensors_index] == cur_addr) {
      last_output_dims_[out_tensors_index] = output_dims_[out_tensors_index];
      ++out_tensors_index;
      continue;
    }

    auto set_flag = context_->setTensorAddress(tensor_name, cur_addr);

    if (!set_flag) {
      HOLOSCAN_LOG_ERROR("Buffer binding failed for {} in inference core.", tensor_name);
      status.set_message(" TRT inference core: Error binding output buffer.");
      return status;
    }
    // Address rebind — the captured graph holds this pointer; force a recapture.
    graph_needs_recapture_ = true;

    // Output Binding Stability Cache
    last_output_addr_[out_tensors_index] = cur_addr;
    last_output_dims_[out_tensors_index] = output_dims_[out_tensors_index];
    ++out_tensors_index;
  }

  // Pin outputs only after the first successful pass through the loop.
  if (!outputs_pinned_) {
    outputs_pinned_ = true;
  }

  // CUDA Graphs dispatch — three regimes:
  //   1. Graphs disabled OR TRT's uncaptureable first-phase (first_phase_): run enqueueV3
  //      directly. No capture, no launch. See TRT docs "two phases" note:
  //      https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#cuda-graphs
  //   2. Change since the last capture (graph_needs_recapture_) or first capture ever
  //      (!cuda_graph_instance_): begin capture → enqueueV3 (captured) → end capture →
  //      cudaGraphExecUpdate (or instantiate on failure) → cudaGraphLaunch. Clear
  //      graph_needs_recapture_.
  //   3. Steady state (graphs enabled, past first phase, no change since last capture, exec
  //      instance exists): skip enqueueV3 entirely and just cudaGraphLaunch the existing
  //      instance. This is what CUDA graphs are for.
  const bool graphs_active = enable_cuda_graphs_ && !first_phase_;
  const bool can_replay_cached =
      graphs_active && !graph_needs_recapture_ && cuda_graph_instance_ != nullptr;
  bool capturing_graph = graphs_active && !can_replay_cached;

  if (capturing_graph) {
    check_cuda(cudaStreamBeginCapture(cuda_stream_, cudaStreamCaptureModeThreadLocal));
  }

  bool infer_status = true;
  if (!can_replay_cached) {
    infer_status = context_->enqueueV3(cuda_stream_);
    if (!infer_status) {
      if (capturing_graph) {
        // end the capture and destroy the graph when inference failed and we are capturing
        cudaGraph_t cuda_graph = nullptr;
        check_cuda(cudaStreamEndCapture(cuda_stream_, &cuda_graph));
        check_cuda(cudaGraphDestroy(cuda_graph));
        capturing_graph = false;

        // if inference failed with CUDA Graphs enabled, retry without
        HOLOSCAN_LOG_WARN(
            "TRT inference failed with CUDA Graphs enabled, retrying with without CUDA Graphs. "
            "Consider disabling CUDA Graphs by setting the `enable_cuda_graphs` parameter to "
            "`false`.");
        enable_cuda_graphs_ = false;
        infer_status = context_->enqueueV3(cuda_stream_);
      }

      if (!infer_status) {
        status.set_message(" TRT inference core: Inference failure.");
        return status;
      }
    }
  }

  if (capturing_graph) {
    cudaGraph_t cuda_graph = nullptr;
    check_cuda(cudaStreamEndCapture(cuda_stream_, &cuda_graph));

    // If we've already instantiated the graph, try to update it directly and avoid the
    // instantiation overhead
    cudaGraphExecUpdateResultInfo update_result;
    if (cuda_graph_instance_) {
      check_cuda(cudaGraphExecUpdate(cuda_graph_instance_, cuda_graph, &update_result));
    }

    // Instantiate during the first iteration or whenever the update fails for any reason
    if (!cuda_graph_instance_ || (update_result.result != cudaGraphExecUpdateSuccess)) {
      // If a previous update failed, destroy the cudaGraphExec_t before re-instantiating it
      if (cuda_graph_instance_ != NULL) {
        check_cuda(cudaGraphExecDestroy(cuda_graph_instance_));
      }

      // Instantiate graphExec from graph. The error node and error message parameters are unused
      // here.
      check_cuda(cudaGraphInstantiate(&cuda_graph_instance_, cuda_graph, 0));
    }

    check_cuda(cudaGraphDestroy(cuda_graph));
    graph_needs_recapture_ = false;
  }

  // Launch the graph in regimes 2 and 3 (i.e. any time graphs_active was true when we entered
  // the dispatch block). Regime 1 already ran enqueueV3 directly and needs no launch.
  if (graphs_active && cuda_graph_instance_) {
    check_cuda(cudaGraphLaunch(cuda_graph_instance_, cuda_stream_));
  }

  if (enable_cuda_graphs_) {
    first_phase_ = false;
  }

  if (!cuda_buf_out_) {
    for (auto& output_buffer : output_buffers) {
      if (output_buffer->host_buffer_->size() == 0) {
        status.set_message(" TRT inference core: Empty output host buffer.");
        return status;
      }
      if (output_buffer->device_buffer_->size() != output_buffer->host_buffer_->size()) {
        status.set_message(" TRT inference core: Output Host and Device buffer size mismatch.");
        return status;
      }
      // Copy the results back to CPU memory
      auto cstatus = cudaMemcpyAsync(output_buffer->host_buffer_->data(),
                                     output_buffer->device_buffer_->data(),
                                     output_buffer->device_buffer_->get_bytes(),
                                     cudaMemcpyDeviceToHost,
                                     cuda_stream_);
      if (cstatus != cudaSuccess) {
        status.set_message(" TRT: Device to host transfer failed");
        return status;
      }
      // When copying from device memory to pagable memory the call is synchronous with the host
      // execution. No need to synchronize here.
    }
  }

  // record a CUDA event and pass it back to the caller
  check_cuda(cudaEventRecord(cuda_event_, cuda_stream_));
  *cuda_event_inference = cuda_event_;
  return InferStatus();
}

void TrtInfer::init_gr_inference(void* input_buffer, void* output_buffer) {
  context_->setTensorAddress(input_tensor_names_[0].c_str(), input_buffer);
  context_->setTensorAddress(output_tensor_names_[0].c_str(), output_buffer);

  auto status = cudaStreamCreate(&first_call_stream_);
  if (status != cudaSuccess) {
    throw std::runtime_error("Failed to create CUDA stream in init_gr_inference");
  }

  bool infer_status = context_->enqueueV3(first_call_stream_);

  if (!infer_status) {
    HOLOSCAN_LOG_ERROR("TRT inference core: Inference failure in init_gr_inference.");
    throw std::runtime_error("TRT inference core: Inference failure in init_gr_inference.");
  }

  cudaStreamSynchronize(first_call_stream_);
}

void TrtInfer::do_gr_inference(void* input_buffer, void* output_buffer, cudaStream_t cuda_stream) {
  context_->setTensorAddress(input_tensor_names_[0].c_str(), input_buffer);
  context_->setTensorAddress(output_tensor_names_[0].c_str(), output_buffer);

  bool infer_status = context_->enqueueV3(cuda_stream);

  if (!infer_status) {
    throw std::runtime_error("TRT inference core: Inference failure in do_gr_inference.");
  }
}

// ============================================================================
// Stream-Shared Sequential Dispatch
// ============================================================================
void TrtInfer::bind_runtime_green_context(CUcontext ctx, uint32_t partition_index) {
  runtime_green_context_ = ctx;
  runtime_green_context_partition_ = partition_index;
}

namespace {

// RAII guard: pushes a green-context CUcontext as the current CUDA context for the guard's
// lifetime (including CUDA-Graph capture and enqueueV3), popping it back off on scope exit —
// covering every early-return path in do_inference_on_stream(). Mirrors the CtxPopGuard pattern
// already used for TRT engine building's green-context push in utils.cpp. No-op when ctx is
// null, i.e. this model has no runtime green-context binding (the pre-Phase-2.3 shared-context
// dispatch path).
class GreenContextDispatchGuard {
 public:
  explicit GreenContextDispatchGuard(CUcontext ctx) {
    if (ctx == nullptr) {
      return;
    }
    const CUresult result = cuCtxPushCurrent(ctx);
    if (result != CUDA_SUCCESS) {
      // Green-context isolation is NOT actually in effect for this dispatch. Raise the
      // process-wide signal so ManagerInfer::run_core_inference_par_fast() stops assuming it can
      // skip capture_serialization_mutex_ (see manager/fallback_runtime.hpp), and report the
      // degradation once, in the standard ladder format.
      auto& flag = runtime_fallback_flags().green_context_push_failed;
      const bool already_reported = flag.exchange(true, std::memory_order_relaxed);
      if (!already_reported) {
        emit_runtime_fallback(
            FallbackReason::kGreenContextPushFailed,
            fallback::DispatchKind::kParFast,
            fallback::CaptureProtectionOutcome::kGreenContextIsolated,
            fallback::CaptureProtectionOutcome::kMutexProtected,
            "cuCtxPushCurrent() for the assigned green context partition failed (CUresult=" +
                std::to_string(static_cast<int>(result)) +
                "); dispatching in the currently active context and engaging the "
                "capture-serialization mutex instead of relying on context isolation",
            "verify the cuda_green_context_pool partitions are still valid for this device/thread "
            "(a destroyed pool or a device reset invalidates them)");
      }
      HOLOSCAN_LOG_WARN(
          "TRT inference (on_stream): failed to push green context (CUresult={}); dispatching "
          "in the currently active context instead.",
          static_cast<int>(result));
      return;
    }
    pushed_ = true;
  }
  GreenContextDispatchGuard(const GreenContextDispatchGuard&) = delete;
  GreenContextDispatchGuard& operator=(const GreenContextDispatchGuard&) = delete;
  ~GreenContextDispatchGuard() {
    if (pushed_) {
      CUcontext popped = nullptr;
      const CUresult result = cuCtxPopCurrent(&popped);
      if (result != CUDA_SUCCESS) {
        try {
          HOLOSCAN_LOG_WARN("TRT inference (on_stream): cuCtxPopCurrent failed (CUresult={}).",
                            static_cast<int>(result));
        } catch (...) {
          // Swallow: must not throw out of a destructor during stack unwinding.
        }
      }
    }
  }

 private:
  bool pushed_ = false;
};

}  // namespace

InferStatus TrtInfer::do_inference_on_stream(
    const std::vector<std::shared_ptr<DataBuffer>>& input_buffers,
    std::vector<std::shared_ptr<DataBuffer>>& output_buffers, cudaStream_t stream) {
  InferStatus status(holoinfer_code::H_ERROR);
  auto io_index = 0;

  // Lazy input/output binding-stability caches (shared with the canonical path).
  if (!inputs_pinned_ && last_input_addr_.empty()) {
    last_input_addr_.assign(input_buffers.size(), nullptr);
  }
  size_t in_tensor_index = 0;
  for (auto& input_buffer : input_buffers) {
    if (input_buffer->device_buffer_ == nullptr) {
      status.set_message(" TRT inference (on_stream): input device buffer null");
      return status;
    }
    auto tensor_name = engine_->getIOTensorName(io_index++);
    void* cur_in_addr = input_buffer->device_buffer_->data();
    if (inputs_pinned_ && in_tensor_index < last_input_addr_.size() &&
        last_input_addr_[in_tensor_index] == cur_in_addr) {
      ++in_tensor_index;
      continue;
    }
    if (!context_->setTensorAddress(tensor_name, cur_in_addr)) {
      status.set_message(" TRT inference (on_stream): setTensorAddress(input) failed");
      return status;
    }
    // Address rebind — force a recapture on the next captureable frame.
    graph_needs_recapture_ = true;
    if (in_tensor_index < last_input_addr_.size()) {
      last_input_addr_[in_tensor_index] = cur_in_addr;
    }
    ++in_tensor_index;
  }
  if (!inputs_pinned_) {
    inputs_pinned_ = true;
  }

  if (!outputs_pinned_ && last_output_addr_.empty()) {
    last_output_addr_.assign(output_buffers.size(), nullptr);
    last_output_dims_.assign(output_buffers.size(), {});
  }
  size_t out_tensors_index = 0;
  for (auto& output_buffer : output_buffers) {
    void* cur_addr = output_buffer->device_buffer_->data();
    auto tensor_name = engine_->getIOTensorName(io_index++);

    // Refresh dims + size unconditionally (see do_inference for rationale) —
    // dynamic-shape outputs can change per compute() call even when the
    // underlying device pointer stays stable.
    nvinfer1::Dims outputDims = context_->getTensorShape(tensor_name);
    for (int ai = 0; ai < outputDims.nbDims; ++ai) {
      output_dims_[out_tensors_index][ai] = outputDims.d[ai];
    }
    size_t dynamic_buffer_size = std::accumulate(output_dims_[out_tensors_index].begin(),
                                                 output_dims_[out_tensors_index].end(),
                                                 1,
                                                 std::multiplies<size_t>());
    if (dynamic_buffer_size != output_buffer->device_buffer_->size()) {
      output_buffer->device_buffer_->resize(dynamic_buffer_size);
      output_buffer->host_buffer_->resize(dynamic_buffer_size);
      cur_addr = output_buffer->device_buffer_->data();
      graph_needs_recapture_ = true;
    }
    // Skip only the setTensorAddress call when the bound address is stable.
    if (outputs_pinned_ && last_output_addr_[out_tensors_index] == cur_addr) {
      last_output_dims_[out_tensors_index] = output_dims_[out_tensors_index];
      ++out_tensors_index;
      continue;
    }
    if (!context_->setTensorAddress(tensor_name, cur_addr)) {
      status.set_message(" TRT inference (on_stream): setTensorAddress(output) failed");
      return status;
    }
    // Address rebind — force a recapture on the next captureable frame.
    graph_needs_recapture_ = true;
    last_output_addr_[out_tensors_index] = cur_addr;
    last_output_dims_[out_tensors_index] = output_dims_[out_tensors_index];
    ++out_tensors_index;
  }
  if (!outputs_pinned_) {
    outputs_pinned_ = true;
  }

  // Runtime green-context binding: when this model was assigned a
  // CudaGreenContextPool partition (see infer_manager_setup.cpp), push its CUcontext for the
  // remainder of this call — covering CUDA-Graph capture begin/end and enqueueV3 below — so
  // dispatch actually executes within that SM partition's isolated context rather than
  // whatever context happened to be current. Popped automatically on every return path.
  GreenContextDispatchGuard green_context_guard(runtime_green_context_);

  // CUDA Graphs dispatch — see the parallel comment in do_inference() for the three-regime
  // rationale. State (first_phase_, cuda_graph_instance_, graph_needs_recapture_) is shared
  // with do_inference(); callers commit to one entry point per operator lifecycle.
  const bool graphs_active = enable_cuda_graphs_ && !first_phase_;
  const bool can_replay_cached =
      graphs_active && !graph_needs_recapture_ && cuda_graph_instance_ != nullptr;
  bool capturing = graphs_active && !can_replay_cached;

  if (capturing) {
    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  }

  if (!can_replay_cached) {
    if (!context_->enqueueV3(stream)) {
      if (capturing) {
        cudaGraph_t g = nullptr;
        check_cuda(cudaStreamEndCapture(stream, &g));
        if (g)
          check_cuda(cudaGraphDestroy(g));
        capturing = false;
        HOLOSCAN_LOG_WARN("TRT inference (on_stream): retrying without CUDA Graphs");
        enable_cuda_graphs_ = false;
        if (!context_->enqueueV3(stream)) {
          status.set_message(" TRT inference (on_stream): enqueueV3 failed");
          return status;
        }
      } else {
        status.set_message(" TRT inference (on_stream): enqueueV3 failed");
        return status;
      }
    }
  }

  if (capturing) {
    cudaGraph_t g = nullptr;
    check_cuda(cudaStreamEndCapture(stream, &g));
    cudaGraphExecUpdateResultInfo upd;
    if (cuda_graph_instance_)
      check_cuda(cudaGraphExecUpdate(cuda_graph_instance_, g, &upd));
    if (!cuda_graph_instance_ || upd.result != cudaGraphExecUpdateSuccess) {
      if (cuda_graph_instance_)
        check_cuda(cudaGraphExecDestroy(cuda_graph_instance_));
      check_cuda(cudaGraphInstantiate(&cuda_graph_instance_, g, 0));
    }
    check_cuda(cudaGraphDestroy(g));
    graph_needs_recapture_ = false;
  }

  // Launch in regimes 2 (freshly captured) and 3 (replay cached). Regime 1 already
  // dispatched via enqueueV3 uncaptured.
  if (graphs_active && cuda_graph_instance_) {
    check_cuda(cudaGraphLaunch(cuda_graph_instance_, stream));
  }

  if (enable_cuda_graphs_) {
    first_phase_ = false;
  }

  return InferStatus();
}

}  // namespace inference
}  // namespace holoscan
