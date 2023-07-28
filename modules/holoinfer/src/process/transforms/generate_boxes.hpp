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
#ifndef MODULES_HOLOINFER_TRANSFORMS_GENERATE_BOXES_HPP
#define MODULES_HOLOINFER_TRANSFORMS_GENERATE_BOXES_HPP

#include <bits/stdc++.h>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <holoinfer.hpp>
#include <holoinfer_constants.hpp>
#include <holoinfer_utils.hpp>

#include <process/transform.hpp>
namespace holoscan {
namespace inference {
/**
 * @brief Generate boxes Transform Class
 */
class GenerateBoxes : public TransformBase {
 public:
  /**
   * @brief Default Constructor
   */
  GenerateBoxes() {}
  explicit GenerateBoxes(const std::string& config_path) : config_path_(config_path) {}
  /**
   * @brief Initializer. Parses the config file and populates all required variables to be used in
   * the execution process
   *
   * @param input_tensors Input tensors from inference operator
   * @return InferStatus
   */
  InferStatus initialize(const std::vector<std::string>& input_tensors);

  /**
   * @brief Create a tensor mapping of pre-defined tensors of the class to incoming tensors from
   * inference operator
   *
   * @param input_tensors Input tensors from inference operator
   * @return InferStatus
   */
  InferStatus create_tensor_map(const std::vector<std::string>& input_tensors);

  /**
   * @brief Core execution. Ingests input data with tensor names as "scores", "labels" and "boxes".
   * Finds the valid boxes and text and populates the tensors and coordinates to be used in holoviz.
   * @param indata Map with key as tensor name as value as raw data buffer
   * @param indim Map with key as tensor name as value as dimension of the input tensor
   * @param processed_data Output data map, that will be populated
   * @param processed_dims Dimension of the output tensor, is populated during the processing
   * @return InferStatus
   * */
  InferStatus execute(const std::map<std::string, void*>& indata,
                      const std::map<std::string, std::vector<int>>& indim, DataMap& processed_data,
                      DimType& processed_dims);

  /**
   * @brief Ingests input data with tensor names as "scores", "labels" and "masks".
   * Finds the object masks and prepares it for rendering in holoviz.
   * @param indata Map with key as tensor name and value as raw data buffer
   * @param indim Map with key as tensor name and value as dimension of the input tensor
   * @param processed_data Output data map, that will be populated
   * @param processed_dims Dimension of the output tensor, is populated during the processing
   * @return InferStatus
   * */
  InferStatus execute_mask(const std::map<std::string, void*>& indata,
                           const std::map<std::string, std::vector<int>>& indim,
                           DataMap& processed_data, DimType& processed_dims);

 private:
  /// @brief  Path to the configuration file
  std::string config_path_;

  /// @brief Map containing key as object name and value as count of the objects to be stored for
  /// display. This is created based on data from configuration file. Object names in label_count
  /// must match the keywords in label_file.
  std::map<std::string, int> label_count;

  /// Threshold value of scores. All values above this threshold are selected as a valid score. This
  /// is configurable via configuration file
  float threshold = 0.75;

  /// Default display width. Used to generate coordinates of boxes and text for holoviz display.
  int width = 1920;

  /// Default display height. Used to generate coordinates of boxes and text for holoviz display.
  int height = 1080;

  /// Label file name. Path to label file name is derived from the configuration file path.
  std::string label_file = {};

  /// Label strings. It contains all the labels read from the label_file, and has a default value of
  /// object (if there is a mismatch)
  std::vector<std::string> label_strings = {"object"};

  /// Stores mapping of model tensors to Holoscan GXF tensors
  std::map<std::string, std::string> tensor_to_output_map;

  /// Color to be displayed for masks per object
  std::map<std::string, std::vector<float>> color_map;
};
}  // namespace inference
}  // namespace holoscan

#endif /* MODULES_HOLOINFER_TRANSFORMS_GENERATE_BOXES_HPP */
