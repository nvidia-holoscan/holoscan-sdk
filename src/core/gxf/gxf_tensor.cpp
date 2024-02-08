/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/gxf/gxf_tensor.hpp"

#include <memory>
#include <optional>
#include <utility>

#include "holoscan/core/common.hpp"

namespace holoscan::gxf {

struct GXFDataType {
  nvidia::gxf::PrimitiveType element_type;
  int64_t bytes_per_element;
};

static GXFDataType dtype2gxfdtype(DLDataType dtype) {
  GXFDataType gxf_dtype;
  int64_t bits = dtype.bits;
  gxf_dtype.bytes_per_element = dtype.bits / 8;

  switch (dtype.code) {
    case kDLInt:
      switch (bits) {
        case 8:
          gxf_dtype.element_type = nvidia::gxf::PrimitiveType::kInt8;
          break;
        case 16:
          gxf_dtype.element_type = nvidia::gxf::PrimitiveType::kInt16;
          break;
        case 32:
          gxf_dtype.element_type = nvidia::gxf::PrimitiveType::kInt32;
          break;
        case 64:
          gxf_dtype.element_type = nvidia::gxf::PrimitiveType::kInt64;
          break;
        default:
          throw std::runtime_error(
              fmt::format("Unsupported DLPack data type (code: {}, bits: {}, lanes: {})",
                          dtype.code,
                          dtype.bits,
                          dtype.lanes));
      }
      break;
    case kDLUInt:
      switch (bits) {
        case 8:
          gxf_dtype.element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
          break;
        case 16:
          gxf_dtype.element_type = nvidia::gxf::PrimitiveType::kUnsigned16;
          break;
        case 32:
          gxf_dtype.element_type = nvidia::gxf::PrimitiveType::kUnsigned32;
          break;
        case 64:
          gxf_dtype.element_type = nvidia::gxf::PrimitiveType::kUnsigned64;
          break;
        default:
          throw std::runtime_error(
              fmt::format("Unsupported DLPack data type (code: {}, bits: {}, lanes: {})",
                          dtype.code,
                          dtype.bits,
                          dtype.lanes));
      }
      break;
    case kDLFloat:
      switch (bits) {
        case 32:
          gxf_dtype.element_type = nvidia::gxf::PrimitiveType::kFloat32;
          break;
        case 64:
          gxf_dtype.element_type = nvidia::gxf::PrimitiveType::kFloat64;
          break;
        default:
          throw std::runtime_error(
              fmt::format("Unsupported DLPack data type (code: {}, bits: {}, lanes: {})",
                          dtype.code,
                          dtype.bits,
                          dtype.lanes));
      }
      break;
    case kDLComplex:
      switch (bits) {
        case 64:
          gxf_dtype.element_type = nvidia::gxf::PrimitiveType::kComplex64;
          break;
        case 128:
          gxf_dtype.element_type = nvidia::gxf::PrimitiveType::kComplex128;
          break;
        default:
          throw std::runtime_error(
              fmt::format("Unsupported DLPack data type (code: {}, bits: {}, lanes: {})",
                          dtype.code,
                          dtype.bits,
                          dtype.lanes));
      }
      break;
    default:
      throw std::runtime_error(
          fmt::format("Unsupported DLPack data type (code: {}, bits: {}, lanes: {})",
                      dtype.code,
                      dtype.bits,
                      dtype.lanes));
  }
  return gxf_dtype;
}

GXFTensor::GXFTensor(std::shared_ptr<DLManagedTensorCtx>& dl_ctx) : dl_ctx_(dl_ctx) {
  auto& dl_managed_tensor = dl_ctx_->tensor;
  auto& dl_tensor = dl_managed_tensor.dl_tensor;

  const uint32_t rank = dl_tensor.ndim;
  const auto shape = [&dl_tensor, &rank]() {
    std::array<int32_t, nvidia::gxf::Shape::kMaxRank> shape;
    for (uint32_t index = 0; index < rank; ++index) { shape[index] = dl_tensor.shape[index]; }
    return nvidia::gxf::Shape(shape, rank);
  }();

  const GXFDataType gxf_dtype = dtype2gxfdtype(dl_tensor.dtype);
  const nvidia::gxf::PrimitiveType element_type = gxf_dtype.element_type;
  const uint64_t bytes_per_element = gxf_dtype.bytes_per_element;

  const auto strides = [&dl_tensor, &rank, &shape, &bytes_per_element]() {
    nvidia::gxf::Tensor::stride_array_t strides;
    // If strides is not set, set it to the default strides
    if (dl_tensor.strides == nullptr) {
      strides = nvidia::gxf::ComputeTrivialStrides(shape, bytes_per_element);
    } else {
      for (uint32_t index = 0; index < rank; ++index) {
        // GXF Tensor's stride is in bytes, but DLPack's stride is in elements
        strides[index] = dl_tensor.strides[index] * bytes_per_element;
      }
    }
    return strides;
  }();

  nvidia::gxf::MemoryStorageType storage_type = nvidia::gxf::MemoryStorageType::kDevice;
  switch (dl_tensor.device.device_type) {
    case kDLCUDAHost:
      storage_type = nvidia::gxf::MemoryStorageType::kHost;
      break;
    case kDLCUDA:
      storage_type = nvidia::gxf::MemoryStorageType::kDevice;
      break;
    case kDLCPU:
      storage_type = nvidia::gxf::MemoryStorageType::kSystem;
      break;
    default:
      throw std::runtime_error(fmt::format("Unsupported DLPack device type (device_type: {})",
                                           dl_tensor.device.device_type));
  }

  this->wrapMemory(shape,
                   element_type,
                   bytes_per_element,
                   strides,
                   storage_type,
                   static_cast<char*>(dl_tensor.data) +
                       dl_tensor.byte_offset,  // shift the pointer by the byte offset
                   [dl_ctx = dl_ctx_](void*) mutable {
                     dl_ctx.reset();
                     return nvidia::gxf::Success;
                   });
}

// The number of mutexes to be used for GXFTensor::GXFTensor(nvidia::gxf::Tensor&, int64_t).
constexpr int HASH_MUTEX_COUNT = 257;  ///< A large-enough prime number

/**
 * @brief Generate a hash from the provided value using a straightforward index hash function.
 *
 * This function is primarily designed to select a mutex in the
 * `GXFTensor::GXFTensor(nvidia::gxf::Tensor&, int64_t)` method.
 * The hashing can be particularly beneficial when the 'value' represents a tensor's address.
 *
 * @param value The input value to hash.
 * @return The resulting hash value.
 */
static uint64_t simple_index_hash(uint64_t value) {
  value ^= (value >> 32);
  value ^= (value >> 16);
  return (value % HASH_MUTEX_COUNT) + 1;
}

GXFTensor::GXFTensor(nvidia::gxf::Tensor& tensor, int64_t id) {
  // Note:: Issue 4272363
  static std::mutex mutexes_[HASH_MUTEX_COUNT + 1];
  // Skip mutex usage if the user explicitly sets the 'id' to -1.
  std::optional<std::unique_lock<std::mutex>> lock;
  if (id != -1) {
    auto mutex_index = simple_index_hash(static_cast<uint64_t>(id));
    lock.emplace(mutexes_[mutex_index]);
  }

  // Get the tensor info
  const auto shape = tensor.shape();
  const auto element_type = tensor.element_type();
  const auto bytes_per_element = tensor.bytes_per_element();
  const auto storage_type = tensor.storage_type();
  const auto pointer = tensor.pointer();
  const auto shape_rank = shape.rank();

  // Move the memory buffer from 'tensor' to 'buffer' variable with a shared pointer
  auto buffer = std::make_shared<GXFMemoryBuffer>(std::move(tensor.move_buffer()));

  dl_ctx_ = std::make_shared<DLManagedTensorCtx>();
  dl_ctx_->memory_ref = buffer;
  auto& dl_managed_tensor = dl_ctx_->tensor;
  auto& dl_tensor = dl_managed_tensor.dl_tensor;

  auto& buffer_shape = buffer->dl_shape;
  auto& buffer_strides = buffer->dl_strides;

  stride_array_t strides;
  buffer_shape.reserve(shape_rank);
  buffer_strides.reserve(shape_rank);

  for (uint32_t index = 0; index < shape_rank; ++index) {
    const auto stride = tensor.stride(index);
    strides[index] = stride;

    buffer_shape.push_back(shape.dimension(index));
    // DLPack's stride (buffer_strides) is in elements but GXF Tensor's stride is in bytes
    buffer_strides.push_back(stride / bytes_per_element);
  }

  // Reinsert the MemoryBuffer into the 'tensor' the new deallocator (just holding
  // a shared pointer to the memory buffer so that releasing it would be handled by the shared
  // pointer's destructor).
  tensor.wrapMemory(shape,
                    element_type,
                    bytes_per_element,
                    strides,
                    storage_type,
                    pointer,
                    [buffer = buffer](void*) mutable {
                      buffer.reset();
                      return nvidia::gxf::Success;
                    });

  // Do the same for the 'this' object
  this->wrapMemory(shape,
                   element_type,
                   bytes_per_element,
                   strides,
                   storage_type,
                   pointer,
                   [buffer = buffer](void*) mutable {
                     buffer.reset();
                     return nvidia::gxf::Success;
                   });

  // Set the DLManagedTensorCtx
  dl_managed_tensor.manager_ctx = nullptr;  // not used
  dl_managed_tensor.deleter = nullptr;      // not used

  dl_tensor.data = this->pointer();
  dl_tensor.device = this->device();
  dl_tensor.ndim = this->shape().rank();
  dl_tensor.dtype = this->dtype();
  dl_tensor.shape = buffer_shape.data();
  dl_tensor.strides = buffer_strides.data();
  dl_tensor.byte_offset = 0;
}

DLDevice GXFTensor::device() const {
  switch (storage_type()) {
    case nvidia::gxf::MemoryStorageType::kSystem:
      return DLDevice{kDLCPU, 0};
    case nvidia::gxf::MemoryStorageType::kHost:
    case nvidia::gxf::MemoryStorageType::kDevice:
      return dldevice_from_pointer(pointer());
    default:
      throw std::runtime_error(fmt::format("Unsupported GXF storage type (storage_type: {})",
                                           static_cast<int>(storage_type())));
  }
}

DLDataType GXFTensor::dtype() const {
  DLDataType dtype;
  dtype.lanes = 1;
  dtype.bits = bytes_per_element() * 8;

  auto element_type = this->element_type();
  switch (element_type) {
    case nvidia::gxf::PrimitiveType::kInt8:
    case nvidia::gxf::PrimitiveType::kInt16:
    case nvidia::gxf::PrimitiveType::kInt32:
    case nvidia::gxf::PrimitiveType::kInt64:
      dtype.code = kDLInt;
      break;
    case nvidia::gxf::PrimitiveType::kUnsigned8:
    case nvidia::gxf::PrimitiveType::kUnsigned16:
    case nvidia::gxf::PrimitiveType::kUnsigned32:
    case nvidia::gxf::PrimitiveType::kUnsigned64:
      dtype.code = kDLUInt;
      break;
    case nvidia::gxf::PrimitiveType::kFloat32:
    case nvidia::gxf::PrimitiveType::kFloat64:
      dtype.code = kDLFloat;
      break;
    case nvidia::gxf::PrimitiveType::kComplex64:
    case nvidia::gxf::PrimitiveType::kComplex128:
      dtype.code = kDLComplex;
      break;
    default:
      throw std::runtime_error(
          fmt::format("Unsupported GXF element type: {}", static_cast<int>(element_type)));
  }
  return dtype;
}

std::shared_ptr<holoscan::Tensor> GXFTensor::as_tensor() {
  auto tensor = std::make_shared<holoscan::Tensor>(dl_ctx_);
  return tensor;
}

std::shared_ptr<GXFTensor> GXFTensor::from_tensor(std::shared_ptr<holoscan::Tensor> tensor) {
  auto gxf_tensor = std::make_shared<GXFTensor>(tensor->dl_ctx());
  return gxf_tensor;
}

}  // namespace holoscan::gxf
