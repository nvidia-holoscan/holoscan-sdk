/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "generate_boxes.hpp"

#include <yaml-cpp/yaml.h>

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace holoscan {
namespace inference {

InferStatus GenerateBoxes::create_tensor_map(const std::vector<std::string>& input_tensors) {
  for (auto& tensor_key : input_tensors) {
    if (tensor_key.find("scores") != std::string::npos) {
      if (tensor_to_output_map.find("scores") == tensor_to_output_map.end()) {
        tensor_to_output_map.insert({"scores", tensor_key});
      } else {
        return InferStatus(
            holoinfer_code::H_ERROR,
            "Generate boxes, Multiple tensors with scores keyword. It must be unique.");
      }
    }
    if (tensor_key.find("labels") != std::string::npos) {
      if (tensor_to_output_map.find("labels") == tensor_to_output_map.end()) {
        tensor_to_output_map.insert({"labels", tensor_key});
      } else {
        return InferStatus(
            holoinfer_code::H_ERROR,
            "Generate boxes, Multiple tensors with labels keyword. It must be unique.");
      }
    }
    if (tensor_key.find("masks") != std::string::npos) {
      if (tensor_to_output_map.find("masks") == tensor_to_output_map.end()) {
        tensor_to_output_map.insert({"masks", tensor_key});
      } else {
        return InferStatus(
            holoinfer_code::H_ERROR,
            "Generate boxes, Multiple tensors with masks keyword. It must be unique.");
      }
    }
    if (tensor_key.find("boxes") != std::string::npos) {
      if (tensor_to_output_map.find("boxes") == tensor_to_output_map.end()) {
        tensor_to_output_map.insert({"boxes", tensor_key});
      } else {
        return InferStatus(
            holoinfer_code::H_ERROR,
            "Generate boxes, Multiple tensors with boxes keyword. It must be unique.");
      }
    }
  }

  if (tensor_to_output_map.find("scores") == tensor_to_output_map.end() ||
      tensor_to_output_map.find("labels") == tensor_to_output_map.end()) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "Generate boxes, tensor_to_output_map must have the following mappings: "
                       "labels, scores.");
  }

  return InferStatus();
}

InferStatus GenerateBoxes::initialize(const std::vector<std::string>& input_tensors) {
  if (config_path_.length() == 0) {
    HOLOSCAN_LOG_ERROR("Config path cannot be empty");
    return InferStatus(holoinfer_code::H_ERROR, "Generate boxes, invalid config path.");
  }

  if (!std::filesystem::exists(config_path_)) {
    HOLOSCAN_LOG_ERROR("Config path not found: {}", config_path_);
    return InferStatus(holoinfer_code::H_ERROR, "Generate boxes, invalid config path.");
  }

  YAML::Node config = YAML::LoadFile(config_path_);
  if (!config["generate_boxes"]) {
    HOLOSCAN_LOG_ERROR("Generate boxes: generate_boxes key not present in config file {}",
                       config_path_);
    return InferStatus(holoinfer_code::H_ERROR, "Generate boxes, incorrect config file.");
  }
  auto configuration = config["generate_boxes"].as<node_type>();

  if (configuration.find("display") != configuration.end()) {
    if (configuration["display"].find("width") == configuration["display"].end()) {
      HOLOSCAN_LOG_WARN("Generate boxes: width missing. Using default width = {}.", width);
    } else {
      width = std::stoi(configuration["display"]["width"]);
    }
    if (configuration["display"].find("height") == configuration["display"].end()) {
      HOLOSCAN_LOG_WARN("Generate boxes: height missing. Using default height = {}.", height);
    } else {
      height = std::stoi(configuration["display"]["height"]);
    }
    HOLOSCAN_LOG_INFO("Updated width, height: {}, {}", width, height);
  } else {
    HOLOSCAN_LOG_WARN(
        "Generate boxes: display setting missing. Using default width = {} and height = {}.",
        width,
        height);
  }

  if (configuration.find("params") == configuration.end()) {
    HOLOSCAN_LOG_ERROR("Generate boxes: parames missing.");
    return InferStatus(holoinfer_code::H_ERROR, "Generate boxes, params missing in config.");
  }

  if (configuration["params"].find("label_file") == configuration["params"].end()) {
    HOLOSCAN_LOG_ERROR("Generate boxes: label_file missing in params.");
    return InferStatus(holoinfer_code::H_ERROR, "Generate boxes, Missing parameter.");
  }

  label_file = configuration["params"]["label_file"];
  std::string label_file_path =
      std::filesystem::path(config_path_).parent_path() / std::filesystem::path(label_file);
  HOLOSCAN_LOG_INFO("Label file path: {}", label_file_path);

  if (!std::filesystem::exists(label_file_path)) {
    HOLOSCAN_LOG_ERROR("Label file path incorrect {}", label_file_path);
    return InferStatus(holoinfer_code::H_ERROR, "Generate boxes, label path does not exist.");
  }

  // label file exists and found
  std::ifstream infile(label_file_path);
  std::string label;
  if (infile) {  // read all labels and store them
    while (std::getline(infile, label)) {
      label_strings.push_back(label);
    }
    HOLOSCAN_LOG_INFO("Count of labels: {}", label_strings.size());
  } else {
    HOLOSCAN_LOG_ERROR("Error opening label file {}", label_file_path);
    return InferStatus(holoinfer_code::H_ERROR, "Generate boxes, error opening label file.");
  }
  infile.close();

  if (configuration["params"].find("threshold") == configuration["params"].end()) {
    HOLOSCAN_LOG_INFO(
        "Generate boxes: threshold parameter for scores missing in params. Using default value "
        "of 0.75");
  } else {
    threshold = std::stof(configuration["params"]["threshold"]);
    HOLOSCAN_LOG_INFO("Updated threshold value: {}", threshold);
  }

  if (configuration.find("objects") != configuration.end()) {
    for (const auto& [current_object, count] : configuration["objects"]) {
      label_count.insert({current_object, std::stoi(count)});
    }
  } else {
    HOLOSCAN_LOG_WARN("Generate boxes: no object settings found, nothing will be displayed.");
  }

  if (configuration.find("color") != configuration.end()) {
    for (const auto& [current_object, current_color] : configuration["color"]) {
      std::vector<std::string> tokens;
      std::vector<float> col;

      if (current_color.length() != 0) {
        holoscan::inference::string_split(current_color, tokens, ' ');
        switch (tokens.size()) {
          case 4:
            for (const auto& t : tokens) {
              col.push_back(std::stof(t));
            }
            break;
          case 3:
            for (const auto& t : tokens) {
              col.push_back(std::stof(t));
            }
            col.push_back(1.0);
            break;
          default:
            color_map.insert({current_object, {0, 0, 0, 1}});
            break;
        }
      } else {
        color_map.insert({current_object, {0, 0, 0, 1}});
      }
      color_map.insert({current_object, col});
    }
  }

  return create_tensor_map(input_tensors);
}

InferStatus GenerateBoxes::execute_mask(const std::map<std::string, void*>& indata,
                                        const std::map<std::string, std::vector<int>>& indim,
                                        DataMap& processed_data, DimType& processed_dims) {
  auto key = "holoviz_masks";
  if (processed_data.find(key) == processed_data.end()) {
    processed_data.insert({key, std::make_shared<DataBuffer>()});
    processed_dims.insert({key, {{height, width, 4}}});
  }
  processed_data.at(key)->host_buffer_->resize(height * width * 4);

  float* scores = static_cast<float*>(indata.at(tensor_to_output_map.at("scores")));
  float* masks = static_cast<float*>(indata.at(tensor_to_output_map.at("masks")));
  int64_t* labels = static_cast<int64_t*>(indata.at(tensor_to_output_map.at("labels")));

  auto dims_scores = indim.at(tensor_to_output_map.at("scores"));

  size_t size_scores =
      accumulate(dims_scores.begin(), dims_scores.end(), 1, std::multiplies<size_t>());
  auto buffer = reinterpret_cast<float*>(processed_data.at(key)->host_buffer_->data());

  for (int i = 0; i < size_scores; i++) {
    if (scores[i] > threshold) {
      std::string key_mask = "object";
      // if the label is not found in config, object is used as label
      if (labels[i] < label_strings.size()) {  // it has to be a valid label
        if (label_count.find(label_strings[labels[i]]) != label_count.end()) {
          key_mask = label_strings[labels[i]];
        }
      }

      float red = 1, blue = 1, green = 1, alpha = 1;
      if (color_map.find(key_mask) != color_map.end()) {
        red = color_map.at(key_mask)[0];
        green = color_map.at(key_mask)[1];
        blue = color_map.at(key_mask)[2];
        alpha = color_map.at(key_mask)[3];
      }

      for (int a = 0; a < height; a++) {
        for (int b = 0; b < width; b++) {
          auto index = i * width * height + a * width + b;
          auto hindex = (a * width + b) * 4;

          if (masks[index] > threshold) {
            buffer[hindex] = red;
            buffer[hindex + 1] = green;
            buffer[hindex + 2] = blue;
            buffer[hindex + 3] = alpha;
          }
        }
      }
    }
  }

  return InferStatus();
}

InferStatus GenerateBoxes::execute(const std::map<std::string, void*>& indata,
                                   const std::map<std::string, std::vector<int>>& indim,
                                   DataMap& processed_data, DimType& processed_dims) {
  if (indata.find(tensor_to_output_map.at("scores")) == indata.end() ||
      indata.find(tensor_to_output_map.at("labels")) == indata.end() ||
      indim.find(tensor_to_output_map.at("scores")) == indim.end() ||
      indim.find(tensor_to_output_map.at("labels")) == indim.end()) {
    return InferStatus(holoinfer_code::H_ERROR,
                       "Generate boxes, Input data and dimension map must have the following "
                       "tensors: labels, scores.");
  }

  if (tensor_to_output_map.find("masks") != tensor_to_output_map.end()) {
    if (indata.find(tensor_to_output_map.at("masks")) != indata.end()) {
      return execute_mask(indata, indim, processed_data, processed_dims);
    }
    return InferStatus(
        holoinfer_code::H_ERROR,
        "Generate boxes, Input data must have a 'masks' tensor when the dimension map does.");
  }

  if (indata.find(tensor_to_output_map.at("boxes")) == indata.end() ||
      indim.find(tensor_to_output_map.at("boxes")) == indim.end()) {
    return InferStatus(
        holoinfer_code::H_ERROR,
        "Generate boxes, Input data must have a 'boxes' tensor when the dimension map does.");
  }
  // reset all tensors to be displayed in holoviz
  for (const auto& [object_name, max_objects] : label_count) {
    for (int i = 0; i < max_objects; i++) {
      auto key = fmt::format("{}{}", object_name, i);
      auto key_text = fmt::format("{}text{}", object_name, i);

      if (processed_data.find(key) == processed_data.end()) {
        processed_data.insert({key, std::make_shared<DataBuffer>()});
        processed_data.at(key)->host_buffer_->resize(4);
        processed_dims.insert({key, {{1, 2, 2}}});

        processed_data.insert({key_text, std::make_shared<DataBuffer>()});
        processed_data.at(key_text)->host_buffer_->resize(3);
        processed_dims.insert({key_text, {{1, 1, 3}}});
      }

      auto current_data = static_cast<float*>(processed_data.at(key)->host_buffer_->data());
      current_data[0] = 0;
      current_data[1] = 0;
      current_data[2] = 0;
      current_data[3] = 0;

      auto current_data_text =
          static_cast<float*>(processed_data.at(key_text)->host_buffer_->data());
      current_data_text[0] = 1.1;
      current_data_text[1] = 1.1;
      current_data_text[2] = 0.05;
    }
  }

  // Populate only the tensors that are required for the current frame
  if (indata.size() > 0) {
    float* boxes = reinterpret_cast<float*>(indata.at(tensor_to_output_map.at("boxes")));
    float* scores = static_cast<float*>(indata.at(tensor_to_output_map.at("scores")));
    int64_t* labels = static_cast<int64_t*>(indata.at(tensor_to_output_map.at("labels")));

    auto dims_scores = indim.at(tensor_to_output_map.at("scores"));

    size_t size_scores =
        accumulate(dims_scores.begin(), dims_scores.end(), 1, std::multiplies<size_t>());
    std::map<std::string, std::vector<std::vector<float>>> valid_boxes;

    // later update with a priority queue
    // Currently the first configured count of the object above threshold is selected
    // Later it will be configured count of the object above threshold with highest score.
    for (int i = 0; i < size_scores; i++) {
      if (scores[i] > threshold) {
        std::string key = "object";  // if the label is not found in config, object is used as label
        if (labels[i] < label_strings.size()) {  // it has to be a valid label
          if (label_count.find(label_strings[labels[i]]) != label_count.end()) {
            key = label_strings[labels[i]];
          }
        }

        valid_boxes[key].push_back({boxes[4 * i] / width,
                                    boxes[4 * i + 1] / height,
                                    boxes[4 * i + 2] / width,
                                    boxes[4 * i + 3] / height});
      }
    }

    for (const auto& [obj_name, obj_boxes] : valid_boxes) {
      int size_of_item = obj_boxes.size();
      HOLOSCAN_LOG_INFO("Valid box count for object {} = {}", obj_name, size_of_item);

      if (label_count.find(obj_name) != label_count.end()) {
        if (size_of_item > label_count.at(obj_name)) {
          HOLOSCAN_LOG_INFO("Valid box count more than maximum display limit of {}",
                            label_count.at(obj_name));
          size_of_item = label_count.at(obj_name);
        }
      } else {
        size_of_item = 0;
      }

      for (int i = 0; i < size_of_item; i++) {
        auto key = fmt::format("{}{}", obj_name, i);
        std::memcpy(processed_data.at(key)->host_buffer_->data(),
                    obj_boxes[i].data(),
                    obj_boxes[i].size() * sizeof(float));

        key = fmt::format("{}text{}", obj_name, i);
        auto current_data_text = static_cast<float*>(processed_data.at(key)->host_buffer_->data());
        current_data_text[0] = obj_boxes[i][2];
        current_data_text[1] = obj_boxes[i][1];
      }
    }
  }
  return InferStatus();
}

}  // namespace inference
}  // namespace holoscan
