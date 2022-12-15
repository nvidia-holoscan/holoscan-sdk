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
#ifndef _HOLOSCAN_INFER_CORE_H
#define _HOLOSCAN_INFER_CORE_H

#include <iostream>
#include <memory>
#include <vector>

#include <holoinfer_buffer.hpp>

namespace holoscan {
namespace inference {
/**
 * @brief Base Inference Class
 */
class InferBase {
 public:
  /**
   * @brief Does the Core inference
   * @param input_data Input DataBuffer
   * @param output_buffer Output DataBuffer, is populated with inferred results
   * @return InferStatus
   * */
  virtual InferStatus do_inference(std::shared_ptr<DataBuffer>& input_data,
                                   std::shared_ptr<DataBuffer>& output_buffer) {
    return InferStatus();
  }

  /**
   * @brief Get input data dimensions to the model
   * @return Vector of values as dimension
   * */
  virtual std::vector<int64_t> get_input_dims() const { return {}; }

  /**
   * @brief Get output data dimensions from the model
   * @return Vector of values as dimension
   * */
  virtual std::vector<int64_t> get_output_dims() const { return {}; }
  virtual void cleanup() {}
};

}  // namespace inference
}  // namespace holoscan
#endif
