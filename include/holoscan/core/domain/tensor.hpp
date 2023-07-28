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

#ifndef HOLOSCAN_CORE_DOMAIN_TENSOR_HPP
#define HOLOSCAN_CORE_DOMAIN_TENSOR_HPP

#include <dlpack/dlpack.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace holoscan {

/**
 * @brief Class that wraps a DLManagedTensor with a memory data reference.
 *
 * This class is used to wrap a DLManagedTensor (`tensor`) with a shared pointer to the memory
 * data (`memory_ref`). This is useful when the memory data is not owned by the DLManagedTensor, but
 * its lifetime is tied to the DLManagedTensor.
 *
 * In Holoscan SDK, this class is used to wrap a DLManagedTensor that is created by other libraries,
 * such as CuPy, with a shared pointer to the memory data so that the memory data is reference
 * counted and can be safely used/destroyed by the Holoscan SDK.
 */
struct DLManagedTensorCtx {
  DLManagedTensor tensor;            ///< The DLManagedTensor to wrap.
  std::shared_ptr<void> memory_ref;  ///< The memory data reference.
};

/**
 * @brief Class to wrap the deleter of a DLManagedTensor.
 *
 * This class is used with DLManagedTensorCtx class to wrap the DLManagedTensor.
 *
 * A shared pointer to this class in DLManagedTensorCtx class is used as the deleter of the
 * DLManagedTensorCtx::memory_ref
 *
 * When the last reference to the DLManagedTensorCtx object is released,
 * DLManagedTensorCtx::memory_ref will also be destroyed, which will call the deleter function
 * of the DLManagedTensor object.
 *
 */
class DLManagedMemoryBuffer {
 public:
  explicit DLManagedMemoryBuffer(DLManagedTensor* self);
  ~DLManagedMemoryBuffer();

 private:
  DLManagedTensor* self_ = nullptr;
};

/**
 * @brief Tensor class.
 *
 * A Tensor is a multi-dimensional array of elements of a single data type.
 *
 * The Tensor class is a wrapper around the DLManagedTensorCtx struct that holds the
 * DLManagedTensor object.
 * (https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv415DLManagedTensor).
 *
 * This class provides a primary interface to access Tensor data and is interoperable with other
 * frameworks that support DLManagedTensor.
 */
class Tensor {
 public:
  Tensor() = default;

  /**
   * @brief Construct a new Tensor from an existing DLManagedTensorCtx.
   *
   * @param ctx A shared pointer to the DLManagedTensorCtx to be used in Tensor construction.
   */
  explicit Tensor(std::shared_ptr<DLManagedTensorCtx>& ctx) : dl_ctx_(ctx) {}

  /**
   * @brief Construct a new Tensor from an existing DLManagedTensor pointer.
   *
   * @param ctx A pointer to the DLManagedTensor to be used in Tensor construction.
   */
  explicit Tensor(DLManagedTensor* dl_managed_tensor_ptr);

  virtual ~Tensor() = default;

  /**
   * @brief Get a pointer to the underlying data.
   *
   * @return The pointer to the Tensor's data.
   */
  void* data() const { return dl_ctx_->tensor.dl_tensor.data; }

  /**
   * @brief Get the device information of the Tensor.
   *
   * @return The device information of the Tensor.
   */
  DLDevice device() const { return dl_ctx_->tensor.dl_tensor.device; }

  /**
   * @brief Get the Tensor's data type information.
   *
   * For details of the DLDataType struct see the DLPack documentation:
   * https://dmlc.github.io/dlpack/latest/c_api.html#_CPPv410DLDataType
   *
   * @return The DLDataType struct containing DLPack dtype information for the tensor.
   */
  DLDataType dtype() const { return dl_ctx_->tensor.dl_tensor.dtype; }

  /**
   * @brief Get the shape of the Tensor data.
   *
   * @return The vector containing the Tensor's shape.
   */
  std::vector<int64_t> shape() const;

  /**
   * @brief Get the strides of the Tensor data.
   *
   * Note that, unlike `DLTensor.strides`, the strides this method returns are in number of bytes,
   * not elements (to be consistent with NumPy/CuPy's strides).
   *
   * @return The vector containing the Tensor's strides.
   */
  std::vector<int64_t> strides() const;

  /**
   * @brief Get the size (number of elements) in the Tensor.
   *
   * The size is defined as the number of elements, not the number of bytes. For the latter,
   * see ::nbytes.
   *
   * If the underlying DLDataType contains multiple lanes, all lanes are considered as a single
   * element. For example, a float4 vectorized type is counted as a single element, not four
   * elements.
   *
   * @return The size of the tensor in number of elements.
   */
  int64_t size() const;

  /**
   * @brief Get the number of dimensions of the Tensor.
   *
   * @return The number of dimensions.
   */
  int32_t ndim() const { return dl_ctx_->tensor.dl_tensor.ndim; }

  /**
   * @brief Get the itemsize of a single Tensor data element.
   *
   * If the underlying DLDataType contains multiple lanes, itemsize takes this into account.
   * For example, a Tensor containing (vectorized) float4 elements would have itemsize 16, not 4.
   *
   * @return The itemsize of the Tensor's data.
   */
  uint8_t itemsize() const {
    return (dl_ctx_->tensor.dl_tensor.dtype.bits * dl_ctx_->tensor.dl_tensor.dtype.lanes + 7) / 8;
  }

  /**
   * @brief Get the total number of bytes for the Tensor's data.
   *
   * @return The size of the Tensor's data in bytes.
   */
  int64_t nbytes() const { return size() * itemsize(); }

  /**
   * @brief Get a DLPack managed tensor pointer to the Tensor.
   *
   * @return A DLManagedTensor* pointer corresponding to the Tensor.
   */
  DLManagedTensor* to_dlpack();

  /**
   * @brief Get the internal DLManagedTensorCtx of the Tensor.
   *
   * @return A shared pointer to the Tensor's DLManagedTensorCtx.
   */
  std::shared_ptr<DLManagedTensorCtx>& dl_ctx() { return dl_ctx_; }

 protected:
  std::shared_ptr<DLManagedTensorCtx> dl_ctx_;  ///< The DLManagedTensorCtx object.
};

/**
 * @brief Detect the device information from the given pointer.
 *
 * @param ptr The pointer to the memory.
 * @return The device information.
 */
DLDevice dldevice_from_pointer(void* ptr);

/**
 * @brief Fill strides from the given DLTensor object.
 *
 * The following fields are used to fill strides:
 *
 * - ndim
 * - shape
 * - dtype
 *
 * If tensor's strides is nullptr, `strides` argument is filled with the calculated strides of the
 * given DLTensor object. Otherwise, `strides` argument is filled with the given DLTensor object's
 * strides. `strides` vector would be resized to the size of `ndim` field of the given DLTensor
 * object.
 *
 * @param tensor DLTensor object that holds information to fill strides.
 * @param[out] strides Strides to fill.
 * @param to_num_elments If true, the strides in `strides` argument are in number of elements, not
 * bytes (default: false).
 */
void calc_strides(const DLTensor& tensor, std::vector<int64_t>& strides,
                  bool to_num_elements = false);

/**
 * @brief Return DLDataType object from the NumPy type string.
 *
 * @param typestr The NumPy type string.
 * @return The DLDataType object.
 */
DLDataType dldatatype_from_typestr(const std::string& typestr);

/**
 * @brief Return a string providing the basic type of the homogeneous array in NumPy.
 *
 * Note: This method assumes little-endian for now.
 *
 * @return A const character pointer that represents a string
 */
const char* numpy_dtype(const DLDataType dtype);

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_DOMAIN_TENSOR_HPP */
