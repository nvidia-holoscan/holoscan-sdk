/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/operators/inference/inference.hpp>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

#include <holoscan/cuda_execution/cuda_green_context_pool.hpp>
#include <holoscan/cuda_execution/cuda_stream_pool.hpp>
#include <holoscan/logger/logger.hpp>

namespace holoscan::ops {

namespace {

// Fills `out` from `node[key]` when present, leaving `out` at its current (default) value
// otherwise.
template <typename T>
void assign_if_present(const YAML::Node& node, const char* key, T& out) {
  if (const auto value = node[key]) {
    out = value.as<T>();
  }
}

InferenceDataMap to_data_map(const YAML::Node& node) {
  InferenceDataMap result;
  if (!node) {
    return result;
  }
  for (const auto& entry : node) {
    result[entry.first.as<std::string>()] = entry.second.as<std::string>();
  }
  return result;
}

InferenceDataVecMap to_data_vec_map(const YAML::Node& node) {
  InferenceDataVecMap result;
  if (!node) {
    return result;
  }
  for (const auto& entry : node) {
    std::vector<std::string> values;
    for (const auto& value : entry.second) {
      values.push_back(value.as<std::string>());
    }
    result[entry.first.as<std::string>()] = std::move(values);
  }
  return result;
}

void require_key(const YAML::Node& node, const char* key) {
  if (!node[key]) {
    throw std::runtime_error(std::string("InferenceOpParams::from_yaml_node: missing required "
                                         "key '") +
                             key + "'");
  }
}

}  // namespace

InferenceOpParams InferenceOpParams::from_yaml_node(const YAML::Node& node,
                                                    const std::filesystem::path& config_dir) {
  require_key(node, "model_path_map");
  require_key(node, "input_map");
  require_key(node, "output_map");

  InferenceOpParams params;

  params.model_path_map = to_data_map(node["model_path_map"]);
  // Relative model paths are resolved against `config_dir` (the standard "config-relative
  // paths" convention). Absolute paths pass through unchanged.
  if (!config_dir.empty()) {
    for (auto& entry : params.model_path_map) {
      const std::filesystem::path resolved(entry.second);
      if (resolved.is_relative()) {
        entry.second = (config_dir / resolved).lexically_normal().string();
      }
    }
  }
  params.input_map = to_data_vec_map(node["input_map"]);
  params.output_map = to_data_vec_map(node["output_map"]);
  params.trt_opt_profile = to_data_vec_map(node["trt_opt_profile"]);
  params.backend_map = to_data_map(node["backend_map"]);
  params.dla_core_map = to_data_map(node["dla_core_map"]);

  assign_if_present(node, "backend", params.backend);
  assign_if_present(node, "is_engine_path", params.is_engine_path);
  assign_if_present(node, "input_on_cuda", params.input_on_cuda);
  assign_if_present(node, "output_on_cuda", params.output_on_cuda);
  assign_if_present(node, "transmit_on_cuda", params.transmit_on_cuda);
  assign_if_present(node, "dynamic_input_dims", params.dynamic_input_dims);
  assign_if_present(node, "output_dynamic_dims", params.output_dynamic_dims);
  assign_if_present(node, "enable_fp16", params.enable_fp16);
  assign_if_present(node, "enable_cuda_graphs", params.enable_cuda_graphs);
  assign_if_present(node, "dla_core", params.dla_core);
  assign_if_present(node, "dla_gpu_fallback", params.dla_gpu_fallback);
  assign_if_present(node, "parallel_inference", params.parallel_inference);
  assign_if_present(node, "max_batch_size", params.max_batch_size);
  assign_if_present(node, "max_memory", params.max_memory);

  if (const auto stream_pool_node = node["cuda_stream_pool"]) {
    CudaStreamPool::Config config;
    assign_if_present(stream_pool_node, "device_id", config.device_id);
    assign_if_present(stream_pool_node, "max_size", config.max_size);
    assign_if_present(stream_pool_node, "reserved_size", config.reserved_size);
    params.cuda_stream_pool = std::make_shared<CudaStreamPool>(config);
    HOLOSCAN_LOG_DEBUG(
        "InferenceOpParams::from_yaml_node: constructed cuda_stream_pool "
        "(max_size={}, device_id={})",
        config.max_size,
        config.device_id);
  }

  if (const auto green_context_node = node["cuda_green_context_pool"]) {
    CudaGreenContextPool::Config config;
    assign_if_present(green_context_node, "device_id", config.device_id);
    assign_if_present(green_context_node, "num_partitions", config.num_partitions);
    assign_if_present(green_context_node, "sms_per_partition", config.sms_per_partition);
    params.cuda_green_context_pool = std::make_shared<CudaGreenContextPool>(config);
    HOLOSCAN_LOG_DEBUG(
        "InferenceOpParams::from_yaml_node: constructed cuda_green_context_pool "
        "(num_partitions={}, device_id={}, sms_per_partition size={})",
        config.num_partitions,
        config.device_id,
        config.sms_per_partition.size());
  }

  return params;
}

InferenceOpParams InferenceOpParams::from_yaml(std::string_view path) {
  const std::string path_str(path);
  YAML::Node root;
  try {
    root = YAML::LoadFile(path_str);
  } catch (const YAML::Exception& exception) {
    throw std::runtime_error("InferenceOpParams::from_yaml(" + path_str +
                             "): failed to parse YAML: " + exception.what());
  }
  return from_yaml_node(root, std::filesystem::path(path_str).parent_path());
}

}  // namespace holoscan::ops
