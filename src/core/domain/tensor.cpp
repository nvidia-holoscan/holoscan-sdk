/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda_runtime.h>

#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/common.hpp"

#define CUDA_TRY(stmt)                                                                  \
  {                                                                                     \
    cuda_status = stmt;                                                                 \
    if (cudaSuccess != cuda_status) {                                                   \
      HOLOSCAN_LOG_ERROR("Runtime call {} in line {} of file {} failed with '{}' ({})", \
                         #stmt,                                                         \
                         __LINE__,                                                      \
                         __FILE__,                                                      \
                         cudaGetErrorString(cuda_status),                               \
                         cuda_status);                                                  \
    }                                                                                   \
  }

namespace holoscan {

Tensor::Tensor(DLManagedTensor* dl_managed_tensor_ptr) {
  dl_ctx_ = std::make_shared<DLManagedTensorCtx>();
  dl_ctx_->memory_ref = std::make_shared<DLManagedMemoryBuffer>(dl_managed_tensor_ptr);

  auto& dl_managed_tensor = dl_ctx_->tensor;
  dl_managed_tensor = *dl_managed_tensor_ptr;
}

DLManagedTensor* Tensor::to_dlpack() {
  auto dl_managed_tensor_ctx = new DLManagedTensorCtx;
  auto& dl_managed_tensor = dl_managed_tensor_ctx->tensor;

  dl_managed_tensor_ctx->memory_ref = dl_ctx_->memory_ref;

  dl_managed_tensor.manager_ctx = dl_managed_tensor_ctx;
  dl_managed_tensor.deleter = [](DLManagedTensor* self) {
    auto dl_managed_tensor_ctx = static_cast<DLManagedTensorCtx*>(self->manager_ctx);
    dl_managed_tensor_ctx->memory_ref.reset();
    delete dl_managed_tensor_ctx;
  };

  // Copy the DLTensor struct data
  DLTensor& dl_tensor = dl_managed_tensor.dl_tensor;
  dl_tensor = dl_ctx_->tensor.dl_tensor;

  return &dl_managed_tensor;
}

std::vector<int64_t> Tensor::shape() const {
  const auto ndim = dl_ctx_->tensor.dl_tensor.ndim;
  const auto shape_ptr = dl_ctx_->tensor.dl_tensor.shape;
  std::vector<int64_t> shape;
  shape.resize(ndim);
  std::copy(shape_ptr, shape_ptr + ndim, shape.begin());
  return shape;
}

std::vector<int64_t> Tensor::strides() const {
  const DLTensor& dl_tensor = dl_ctx_->tensor.dl_tensor;
  std::vector<int64_t> strides;
  calc_strides(dl_tensor, strides, false);
  return strides;
}

int64_t Tensor::size() const {
  const auto ndim = dl_ctx_->tensor.dl_tensor.ndim;
  const auto shape_ptr = dl_ctx_->tensor.dl_tensor.shape;
  int64_t size = 1;
  for (int i = 0; i < ndim; ++i) { size *= shape_ptr[i]; }
  return size;
}

DLDevice dldevice_from_pointer(void* ptr) {
  cudaError_t cuda_status;

  DLDevice device{.device_type = kDLCUDA, .device_id = 0};

  cudaPointerAttributes attributes;
  CUDA_TRY(cudaPointerGetAttributes(&attributes, ptr));
  if (cuda_status != cudaSuccess) {
    throw std::runtime_error(fmt::format("Unable to get pointer attributes from 0x{:x}", ptr));
  }

  switch (attributes.type) {
    case cudaMemoryTypeUnregistered:
      device.device_type = kDLCPU;
      break;
    case cudaMemoryTypeHost:
      device = {.device_type = kDLCUDAHost, .device_id = attributes.device};
      break;
    case cudaMemoryTypeDevice:
      device = {.device_type = kDLCUDA, .device_id = attributes.device};
      break;
    case cudaMemoryTypeManaged:
      device = {.device_type = kDLCUDAManaged, .device_id = attributes.device};
      break;
  }
  return device;
}

void calc_strides(const DLTensor& tensor, std::vector<int64_t>& strides, bool to_num_elements) {
  int64_t ndim = tensor.ndim;
  strides.resize(ndim);
  int64_t elem_size = (to_num_elements) ? 1 : tensor.dtype.bits / 8;
  if (tensor.strides == nullptr) {
    int64_t step = 1;
    for (int64_t index = ndim - 1; index >= 0; --index) {
      strides[index] = step * elem_size;
      step *= tensor.shape[index];
    }
  } else {
    for (int64_t index = 0; index < ndim; ++index) {
      strides[index] = tensor.strides[index] * elem_size;
    }
  }
}

DLDataType dldatatype_from_typestr(const std::string& typestr) {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes = 1;
  if (typestr.substr(0, 1) == ">") { throw std::runtime_error("big endian types not supported"); }
  std::string kind = typestr.substr(1, 1);
  if (kind == "i") {
    code = kDLInt;
  } else if (kind == "u") {
    code = kDLUInt;
  } else if (kind == "f") {
    code = kDLFloat;
  } else if (kind == "c") {
    code = kDLComplex;
  } else {
    throw std::logic_error(fmt::format("dtype.kind: {} is not supported!", kind));
  }
  bits = std::stoi(typestr.substr(2)) * 8;
  DLDataType data_type{code, bits, lanes};
  return data_type;
}

const char* numpy_dtype(const DLDataType dtype) {
  // TODO: consider bfloat16: https://github.com/dmlc/dlpack/issues/45
  // TODO: consider other byte-order
  uint8_t code = dtype.code;
  uint8_t bits = dtype.bits;
  switch (code) {
    case kDLInt:
      switch (bits) {
        case 8:
          return "|i1";
        case 16:
          return "<i2";
        case 32:
          return "<i4";
        case 64:
          return "<i8";
      }
      throw std::logic_error(
          fmt::format("DLDataType(code: kDLInt, bits: {}) is not supported!", bits));
    case kDLUInt:
      switch (bits) {
        case 8:
          return "|u1";
        case 16:
          return "<u2";
        case 32:
          return "<u4";
        case 64:
          return "<u8";
      }
      throw std::logic_error(
          fmt::format("DLDataType(code: kDLUInt, bits: {}) is not supported!", bits));
    case kDLFloat:
      switch (bits) {
        case 16:
          return "<f2";
        case 32:
          return "<f4";
        case 64:
          return "<f8";
      }
      break;
    case kDLComplex:
      switch (bits) {
        case 64:
          return "<c8";
        case 128:
          return "<c16";
      }
      break;
    case kDLBfloat:
      throw std::logic_error(
          fmt::format("DLDataType(code: kDLBfloat, bits: {}) is not supported!", bits));
  }
  throw std::logic_error(
      fmt::format("DLDataType(code: {}, bits: {}) is not supported!", code, bits));
}

}  // namespace holoscan
