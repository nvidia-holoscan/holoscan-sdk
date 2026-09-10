/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/utils/holoinfer_utils.hpp>

#include <cuda_runtime.h>

#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/logger/logger.hpp>

namespace holoscan::utils {

namespace HoloInfer = holoscan::inference;

DLDataType holoinfer_dtype_to_dlpack(HoloInfer::holoinfer_datatype dtype) noexcept {
  switch (dtype) {
    case HoloInfer::holoinfer_datatype::h_Float32:
      return DLDataType{kDLFloat, 32, 1};
    case HoloInfer::holoinfer_datatype::h_Float16:
      return DLDataType{kDLFloat, 16, 1};
    case HoloInfer::holoinfer_datatype::h_Int8:
      return DLDataType{kDLInt, 8, 1};
    case HoloInfer::holoinfer_datatype::h_Int32:
      return DLDataType{kDLInt, 32, 1};
    case HoloInfer::holoinfer_datatype::h_Int64:
      return DLDataType{kDLInt, 64, 1};
    case HoloInfer::holoinfer_datatype::h_UInt8:
      return DLDataType{kDLUInt, 8, 1};
    case HoloInfer::holoinfer_datatype::h_Bool:
      return DLDataType{kDLUInt, 8, 1};
    case HoloInfer::holoinfer_datatype::h_Unsupported:
    default:
      return DLDataType{};
  }
}

HoloInfer::holoinfer_datatype dlpack_dtype_to_holoinfer(DLDataType dtype) noexcept {
  if (dtype.lanes != 1) {
    return HoloInfer::holoinfer_datatype::h_Unsupported;
  }
  switch (dtype.code) {
    case kDLFloat:
      if (dtype.bits == 32) {
        return HoloInfer::holoinfer_datatype::h_Float32;
      }
      if (dtype.bits == 16) {
        return HoloInfer::holoinfer_datatype::h_Float16;
      }
      break;
    case kDLInt:
      if (dtype.bits == 8) {
        return HoloInfer::holoinfer_datatype::h_Int8;
      }
      if (dtype.bits == 32) {
        return HoloInfer::holoinfer_datatype::h_Int32;
      }
      if (dtype.bits == 64) {
        return HoloInfer::holoinfer_datatype::h_Int64;
      }
      break;
    case kDLUInt:
      if (dtype.bits == 8) {
        return HoloInfer::holoinfer_datatype::h_UInt8;
      }
      break;
    default:
      break;
  }
  return HoloInfer::holoinfer_datatype::h_Unsupported;
}

namespace {

[[nodiscard]] expected<void, Error> copy_tensor_into_buffer(const Tensor& tensor,
                                                            HoloInfer::DataBuffer& buffer,
                                                            bool destination_on_cuda,
                                                            cudaStream_t stream,
                                                            const std::string& module) {
  const int64_t nbytes = tensor.nbytes();
  if (nbytes < 0) {
    HOLOSCAN_LOG_ERROR("{}: tensor reports a negative byte size", module);
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
  void* destination =
      destination_on_cuda ? buffer.device_buffer_->data() : buffer.host_buffer_->data();
  // kDLCUDAHost is CUDA-pinned host memory (allocated via cudaMallocHost) — it is
  // reachable from the device but resides on the host, so cudaMemcpyAsync must use
  // a HostTo* kind, not a DeviceTo* kind.  Only kDLCUDA and kDLCUDAManaged are
  // treated as device-resident here.
  const bool source_on_cuda =
      tensor.device().device_type == kDLCUDA || tensor.device().device_type == kDLCUDAManaged;
  cudaMemcpyKind kind = cudaMemcpyHostToHost;
  if (source_on_cuda && destination_on_cuda) {
    kind = cudaMemcpyDeviceToDevice;
  } else if (source_on_cuda && !destination_on_cuda) {
    kind = cudaMemcpyDeviceToHost;
  } else if (!source_on_cuda && destination_on_cuda) {
    kind = cudaMemcpyHostToDevice;
  }
  const cudaError_t status =
      cudaMemcpyAsync(destination, tensor.data(), static_cast<size_t>(nbytes), kind, stream);
  if (status != cudaSuccess) {
    HOLOSCAN_LOG_ERROR("{}: cudaMemcpyAsync failed: {}", module, cudaGetErrorString(status));
    return make_unexpected(Error{ErrorCode::kFailure});
  }
  return {};
}

}  // namespace

expected<void, Error> get_data_per_model(const std::map<std::string, Input<Tensor>*>& ports,
                                         HoloInfer::DataMap& data_per_tensor,
                                         std::map<std::string, std::vector<int>>& dims_per_tensor,
                                         bool input_on_cuda, const std::string& module,
                                         cudaStream_t stream) {
  for (const auto& [name, port] : ports) {
    if (port == nullptr) {
      HOLOSCAN_LOG_ERROR("{}: no bound input port for '{}'", module, name);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto tensor = port->receive_data();
    if (!tensor) {
      HOLOSCAN_LOG_ERROR("{}: failed to receive input tensor '{}': {}",
                         module,
                         name,
                         runtime_error_message(tensor.error()));
      return make_unexpected(Error{ErrorCode::kReceiveError});
    }

    const std::vector<int64_t> shape64 = tensor->shape();
    std::vector<int> dims(shape64.begin(), shape64.end());
    dims_per_tensor[name] = dims;

    const HoloInfer::holoinfer_datatype dtype = dlpack_dtype_to_holoinfer(tensor->dtype());
    if (dtype == HoloInfer::holoinfer_datatype::h_Unsupported) {
      HOLOSCAN_LOG_ERROR("{}: unsupported dtype for input tensor '{}'", module, name);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    const size_t element_count = static_cast<size_t>(
        std::accumulate(shape64.begin(), shape64.end(), int64_t{1}, std::multiplies<int64_t>()));

    auto it = data_per_tensor.find(name);
    if (it == data_per_tensor.end()) {
      std::vector<int64_t> alloc_dims = shape64;
      const auto status =
          HoloInfer::allocate_buffers(data_per_tensor, alloc_dims, dtype, name, input_on_cuda, 0);
      if (status.get_code() != HoloInfer::holoinfer_code::H_SUCCESS) {
        HOLOSCAN_LOG_ERROR("{}: {}", module, status.get_message());
        return make_unexpected(Error{ErrorCode::kFailure});
      }
      it = data_per_tensor.find(name);
    } else {
      if (input_on_cuda) {
        it->second->device_buffer_->resize(element_count);
      } else {
        it->second->host_buffer_->resize(element_count);
      }
    }

    auto copied = copy_tensor_into_buffer(*tensor, *it->second, input_on_cuda, stream, module);
    if (!copied) {
      return copied;
    }
  }
  return {};
}

expected<void, Error> transmit_data_per_model(const HoloInfer::MultiMappings& model_to_tensor_map,
                                              const std::map<std::string, Output<Tensor>*>& ports,
                                              HoloInfer::DataMap& output_per_tensor,
                                              const HoloInfer::DimType& model_dims,
                                              bool output_on_cuda, bool transmit_on_cuda,
                                              const std::string& module,
                                              holoscan::ExecutionContext& context) {
  const cudaStream_t stream = context.cuda_stream().get();

  for (const auto& [model_name, tensor_names] : model_to_tensor_map) {
    const auto dims_it = model_dims.find(model_name);
    if (dims_it == model_dims.end() || dims_it->second.size() != tensor_names.size()) {
      HOLOSCAN_LOG_ERROR(
          "{}: missing or mismatched output dimensions for model '{}'", module, model_name);
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }

    for (size_t index = 0; index < tensor_names.size(); ++index) {
      const std::string& tensor_name = tensor_names[index];
      const auto buffer_it = output_per_tensor.find(tensor_name);
      if (buffer_it == output_per_tensor.end()) {
        HOLOSCAN_LOG_ERROR("{}: no output buffer for tensor '{}'", module, tensor_name);
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      const auto port_it = ports.find(tensor_name);
      if (port_it == ports.end()) {
        // No externally-declared Output<Tensor> port for this tensor name: it is an
        // internal/chained tensor (produced by this model, consumed by another model within the
        // same InferenceOp) rather than an external output. Nothing to transmit.
        continue;
      }
      if (port_it->second == nullptr) {
        HOLOSCAN_LOG_ERROR("{}: no bound output port for tensor '{}'", module, tensor_name);
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }

      const std::vector<int64_t>& shape64 = dims_it->second[index];
      const HoloInfer::holoinfer_datatype dtype = buffer_it->second->get_datatype();
      const DLDataType dl_dtype = holoinfer_dtype_to_dlpack(dtype);
      if (dl_dtype.bits == 0) {
        HOLOSCAN_LOG_ERROR("{}: unsupported dtype for output tensor '{}'", module, tensor_name);
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }

      // Borrow the emitted-tensor storage from the port's runtime pool declared in
      // InferenceOp::setup(). No general-purpose allocator on the hot path.
      auto loan = port_it->second->allocate_tensor(
          holoscan::TensorLoanRequest{.shape = shape64, .dtype = dl_dtype});
      if (!loan) {
        HOLOSCAN_LOG_ERROR("{}: failed to loan output tensor '{}': {}",
                           module,
                           tensor_name,
                           runtime_error_message(loan.error()));
        return make_unexpected(Error{ErrorCode::kFailure});
      }

      auto writer = loan->write(context.cuda_stream());
      if (!writer) {
        HOLOSCAN_LOG_ERROR("{}: failed to open writer for output tensor '{}': {}",
                           module,
                           tensor_name,
                           runtime_error_message(writer.error()));
        return make_unexpected(Error{ErrorCode::kFailure});
      }

      void* source = output_on_cuda ? buffer_it->second->device_buffer_->data()
                                    : buffer_it->second->host_buffer_->data();
      cudaMemcpyKind kind = cudaMemcpyHostToHost;
      if (output_on_cuda && transmit_on_cuda) {
        kind = cudaMemcpyDeviceToDevice;
      } else if (output_on_cuda && !transmit_on_cuda) {
        kind = cudaMemcpyDeviceToHost;
      } else if (!output_on_cuda && transmit_on_cuda) {
        kind = cudaMemcpyHostToDevice;
      }
      const cudaError_t status =
          cudaMemcpyAsync(writer->data(), source, writer->byte_size(), kind, stream);
      if (status != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("{}: cudaMemcpyAsync failed for output tensor '{}': {}",
                           module,
                           tensor_name,
                           cudaGetErrorString(status));
        return make_unexpected(Error{ErrorCode::kFailure});
      }

      // commit() records the readiness event on the compute stream so downstream consumers do
      // not read the tensor before the copy has landed (B3).
      auto committed = std::move(*writer).commit();
      if (!committed) {
        HOLOSCAN_LOG_ERROR("{}: failed to commit output tensor '{}': {}",
                           module,
                           tensor_name,
                           runtime_error_message(committed.error()));
        return make_unexpected(Error{ErrorCode::kFailure});
      }

      auto emitted = port_it->second->emit(
          std::move(*loan), holoscan::EmitOptions{.capture_time = context.activation_time()});
      if (!emitted) {
        HOLOSCAN_LOG_ERROR("{}: failed to emit output tensor '{}': {}",
                           module,
                           tensor_name,
                           runtime_error_message(emitted.error()));
        return make_unexpected(Error{ErrorCode::kFailure});
      }
    }
  }
  return {};
}

}  // namespace holoscan::utils
