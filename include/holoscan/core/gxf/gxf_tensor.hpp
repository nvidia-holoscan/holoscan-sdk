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

#ifndef HOLOSCAN_CORE_GXF_GXF_TENSOR_HPP
#define HOLOSCAN_CORE_GXF_GXF_TENSOR_HPP

#include <memory>
#include <utility>
#include <vector>

#include "gxf/std/tensor.hpp"

#include "holoscan/core/common.hpp"
#include "holoscan/core/domain/tensor.hpp"

namespace holoscan::gxf {

class GXFMemoryBuffer;  // forward declaration

/**
 * @brief Class to wrap GXF Tensor holding DLPack tensor structure.
 *
 */
class GXFTensor : public nvidia::gxf::Tensor {
 public:
  GXFTensor() = default;

  /**
   * @brief Construct a new GXFTensor object.
   *
   * This constructor is used to wrap a GXF Tensor object.
   * The given nvidia::gxf::Tensor object is modified to point to the shared memory buffer so that
   * the memory buffer is shared between the GXF's Tensor object and the GXFTensor object.
   *
   * @param tensor Tensor to wrap.
   */
  explicit GXFTensor(nvidia::gxf::Tensor& tensor);

  /**
   * @brief Construct a new GXFTensor object.
   *
   * @param dl_ctx DLManagedTensorCtx object to wrap.
   */
  explicit GXFTensor(std::shared_ptr<DLManagedTensorCtx>& dl_ctx);

  /**
   * @brief Get DLDevice object from the GXF Tensor.
   *
   * @return DLDevice object.
   */
  DLDevice device() const;

  /**
   * @brief Get DLDataType object from the GXF Tensor.
   *
   * @return DLDataType object.
   */
  DLDataType dtype() const;

  /**
   * @brief Convert GXF Tensor to Holoscan Tensor.
   *
   * @return holoscan::Tensor object converted from GXF Tensor.
   */
  std::shared_ptr<holoscan::Tensor> as_tensor();

  /**
   * @brief Create GXF Tensor object from Holoscan Tensor.
   *
   * @param tensor Holoscan Tensor object to convert.
   * @return The shared pointer object to the GXFTensor object that is created from the given
   * Holoscan Tensor object.
   */
  static std::shared_ptr<GXFTensor> from_tensor(std::shared_ptr<holoscan::Tensor> tensor);

  /**
   * @brief Get the internal DLManagedTensorCtx of the GXFTensor.
   *
   * @return A shared pointer to the Tensor's DLManagedTensorCtx.
   */
  std::shared_ptr<DLManagedTensorCtx>& dl_ctx() { return dl_ctx_; }

 protected:
  std::shared_ptr<DLManagedTensorCtx> dl_ctx_;
};

/**
 * @brief Class to wrap the nvidia::gxf::MemoryBuffer object.
 *
 * This class inherits nvidia::gxf::MemoryBuffer and is used with DLManagedTensorCtx class to wrap
 * the GXF Tensor.
 *
 * A shared pointer to this class in DLManagedTensorCtx class is used as the deleter of the
 * DLManagedTensorCtx::memory_ref
 *
 * When the last reference to the DLManagedTensorCtx object is released,
 * DLManagedTensorCtx::memory_ref will also be destroyed, which will call the deleter function
 * of the DLManagedTensor object.
 *
 * This class holds shape and strides data of DLTensor object so that the data is released together
 * with the DLManagedTensor object.
 */
class GXFMemoryBuffer : public nvidia::gxf::MemoryBuffer {
 public:
  using nvidia::gxf::MemoryBuffer::MemoryBuffer;

  explicit GXFMemoryBuffer(nvidia::gxf::MemoryBuffer&& other)
      : nvidia::gxf::MemoryBuffer(std::forward<nvidia::gxf::MemoryBuffer>(other)) {}

  nvidia::gxf::Tensor::stride_array_t gxf_strides;  ///< Strides of the GXF Tensor.
  std::vector<int64_t> dl_shape;                    ///< Shape of the GXF Tensor.
  std::vector<int64_t> dl_strides;  ///< Strides of the GXF Tensor. This is used to calculate the
                                    ///< strides of the DLTensor.
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_GXF_GXF_TENSOR_HPP */
