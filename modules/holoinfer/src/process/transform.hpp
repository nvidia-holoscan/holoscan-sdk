/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef MODULES_HOLOINFER_PROCESS_TRANSFORM_HPP
#define MODULES_HOLOINFER_PROCESS_TRANSFORM_HPP

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <holoinfer_buffer.hpp>

namespace holoscan {
namespace inference {
/**
 * @brief Base Transform Class
 */
class TransformBase {
 public:
  /**
   * @brief Does the transform execution
   * @param indata Map with key as tensor name and value as raw data buffer
   * @param indim Map with key as tensor name and value as dimension of the input tensor
   * @param processed_data Output data map, that will be populated
   * @param processed_dims Dimension of the output tensor, is populated during the processing
   * @return InferStatus
   * */
  virtual InferStatus execute(const std::map<std::string, void*>& indata,
                              const std::map<std::string, std::vector<int>>& indim,
                              DataMap& processed_data, DimType& processed_dims) {
    return InferStatus();
  }

  virtual InferStatus initialize(const std::vector<std::string>& input_tensors) {
    return InferStatus();
  }
};

}  // namespace inference
}  // namespace holoscan

#endif /* MODULES_HOLOINFER_PROCESS_TRANSFORM_HPP */
