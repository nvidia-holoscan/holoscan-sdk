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

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "gxf/std/dlpack_utils.hpp"
#include "holoscan/core/common.hpp"
#include "holoscan/core/domain/tensor.hpp"

namespace holoscan {

Tensor::Tensor(DLManagedTensor* dl_managed_tensor_ptr) {
  dl_ctx_ = std::make_shared<DLManagedTensorContext>();
  dl_ctx_->memory_ref = std::make_shared<DLManagedMemoryBuffer>(dl_managed_tensor_ptr);

  auto& dl_managed_tensor = dl_ctx_->tensor;
  dl_managed_tensor = *dl_managed_tensor_ptr;
}

bool Tensor::is_contiguous() const {
  int32_t r = static_cast<uint64_t>(ndim());  // rank
  int64_t expected_stride = itemsize();       // size of a single element
  auto tensor_strides = strides();
  auto tensor_shape = shape();
  for (int32_t i = r - 1; i >= 0; --i) {
    int64_t s = tensor_strides[i];                       // stride
    int64_t d = static_cast<uint64_t>(tensor_shape[i]);  // dimension
    if (s != expected_stride) { return false; }
    expected_stride *= d;
  }
  return true;
}

DLManagedTensor* Tensor::to_dlpack() {
  auto dl_managed_tensor_ctx = new DLManagedTensorContext;
  auto& dl_managed_tensor = dl_managed_tensor_ctx->tensor;

  dl_managed_tensor_ctx->memory_ref = dl_ctx_->memory_ref;

  dl_managed_tensor.manager_ctx = dl_managed_tensor_ctx;
  dl_managed_tensor.deleter = [](DLManagedTensor* self) {
    auto dl_managed_tensor_ctx = static_cast<DLManagedTensorContext*>(self->manager_ctx);
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
  auto maybe_device = nvidia::gxf::DLDeviceFromPointer(ptr);
  if (!maybe_device) {
    throw std::runtime_error(
        fmt::format("Failed to determine DLDevice based on pointer with error: {}",
                    GxfResultStr(maybe_device.error())));
  }
  return maybe_device.value();
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
  auto maybe_dtype = nvidia::gxf::DLDataTypeFromTypeString(typestr);
  if (!maybe_dtype) {
    throw std::runtime_error(
        fmt::format("Failed to determine DLDataType from type string with error: {}",
                    GxfResultStr(maybe_dtype.error())));
  }
  return maybe_dtype.value();
}

const char* numpy_dtype(const DLDataType dtype) {
  auto maybe_typestr = nvidia::gxf::numpyTypestr(dtype);
  if (!maybe_typestr) {
    throw std::runtime_error(
        fmt::format("Failed to determine type string from DLDataType with error: {}",
                    GxfResultStr(maybe_typestr.error())));
  }
  return maybe_typestr.value();
}

}  // namespace holoscan
