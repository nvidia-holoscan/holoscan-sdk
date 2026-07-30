/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
  virtual ~TransformBase() = default;

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
