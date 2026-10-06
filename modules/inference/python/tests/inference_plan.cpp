// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cstdint>
#include <iostream>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/external_topic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/inference/operator_catalog.hpp>
#include <holoscan/time/manual_clock.hpp>

namespace {

using holoscan::authoring::ParameterAssignment;
using holoscan::authoring::ParameterValue;

bool registered_parameters_are_preserved() {
  const std::array model_paths{
      ParameterAssignment{"model", ParameterValue::from(std::string_view{"/tmp/model.onnx"})}};
  const std::array input_names{ParameterValue::from(std::string_view{"input"})};
  const std::array inputs{
      ParameterAssignment{"model", ParameterValue::from(std::span{input_names})}};
  const std::array output_names{ParameterValue::from(std::string_view{"result"})};
  const std::array outputs{
      ParameterAssignment{"model", ParameterValue::from(std::span{output_names})}};
  const std::array backend_map{
      ParameterAssignment{"model", ParameterValue::from(std::string_view{"trt"})}};
  const std::array profile_values{ParameterValue::from(std::string_view{"1,2,4"})};
  const std::array profiles{
      ParameterAssignment{"model", ParameterValue::from(std::span{profile_values})}};
  const std::array dla_map{
      ParameterAssignment{"model", ParameterValue::from(std::string_view{"3"})}};
  const std::array stream_fields{
      ParameterAssignment{"device_id", ParameterValue::from(std::int64_t{1})},
      ParameterAssignment{"stream_flags", ParameterValue::from(std::uint64_t{2})},
      ParameterAssignment{"stream_priority", ParameterValue::from(std::int64_t{-2})},
      ParameterAssignment{"reserved_size", ParameterValue::from(std::uint64_t{2})},
      ParameterAssignment{"max_size", ParameterValue::from(std::uint64_t{4})},
      ParameterAssignment{"nvtx_identifier",
                          ParameterValue::from(std::string_view{"registered-streams"})},
  };
  const std::array sms_values{ParameterValue::from(std::int64_t{8}),
                              ParameterValue::from(std::int64_t{8})};
  const std::array green_fields{
      ParameterAssignment{"device_id", ParameterValue::from(std::int64_t{1})},
      ParameterAssignment{"num_partitions", ParameterValue::from(std::uint64_t{2})},
      ParameterAssignment{"sms_per_partition", ParameterValue::from(std::span{sms_values})},
      ParameterAssignment{"default_context", ParameterValue::from(std::int64_t{1})},
      ParameterAssignment{"nvtx_identifier",
                          ParameterValue::from(std::string_view{"registered-contexts"})},
  };
  const std::array assignments{
      ParameterAssignment{"model_path_map", ParameterValue::from(std::span{model_paths})},
      ParameterAssignment{"input_map", ParameterValue::from(std::span{inputs})},
      ParameterAssignment{"output_map", ParameterValue::from(std::span{outputs})},
      ParameterAssignment{"backend", ParameterValue::from(std::string_view{"trt"})},
      ParameterAssignment{"backend_map", ParameterValue::from(std::span{backend_map})},
      ParameterAssignment{"trt_opt_profile", ParameterValue::from(std::span{profiles})},
      ParameterAssignment{"is_engine_path", ParameterValue::from(true)},
      ParameterAssignment{"input_on_cuda", ParameterValue::from(false)},
      ParameterAssignment{"output_on_cuda", ParameterValue::from(false)},
      ParameterAssignment{"transmit_on_cuda", ParameterValue::from(false)},
      ParameterAssignment{"dynamic_input_dims", ParameterValue::from(true)},
      ParameterAssignment{"output_dynamic_dims", ParameterValue::from(true)},
      ParameterAssignment{"enable_fp16", ParameterValue::from(true)},
      ParameterAssignment{"enable_cuda_graphs", ParameterValue::from(false)},
      ParameterAssignment{"dla_core", ParameterValue::from(std::int64_t{2})},
      ParameterAssignment{"dla_core_map", ParameterValue::from(std::span{dla_map})},
      ParameterAssignment{"dla_gpu_fallback", ParameterValue::from(false)},
      ParameterAssignment{"parallel_inference", ParameterValue::from(false)},
      ParameterAssignment{"max_batch_size", ParameterValue::from(std::int64_t{8})},
      ParameterAssignment{"max_memory", ParameterValue::from(std::uint64_t{4096})},
      ParameterAssignment{"cuda_stream_pool", ParameterValue::from(std::span{stream_fields})},
      ParameterAssignment{"cuda_green_context_pool", ParameterValue::from(std::span{green_fields})},
  };

  holoscan::Graph graph{"inference-registered-parameter-readback"};
  const auto row =
      holoscan::authoring::authoring_registry().operator_type("holoscan.inference.InferenceOp");
  if (!row) {
    std::cerr << row.error().message << '\n';
    return false;
  }
  const auto added = row->get().add_to_graph_from_assignments(graph, "inference", assignments);
  if (!added) {
    std::cerr << added.error().message << '\n';
    return false;
  }

  const auto handle = graph.handle<holoscan::ops::InferenceOp>("inference");
  const auto& params = handle->params();
  const holoscan::ops::InferenceDataMap expected_model_paths{{"model", "/tmp/model.onnx"}};
  const holoscan::ops::InferenceDataVecMap expected_inputs{{"model", {"input"}}};
  const holoscan::ops::InferenceDataVecMap expected_outputs{{"model", {"result"}}};
  const holoscan::ops::InferenceDataMap expected_backend_map{{"model", "trt"}};
  const holoscan::ops::InferenceDataVecMap expected_profiles{{"model", {"1,2,4"}}};
  const holoscan::ops::InferenceDataMap expected_dla_map{{"model", "3"}};
  if (params.model_path_map != expected_model_paths || params.input_map != expected_inputs ||
      params.output_map != expected_outputs || params.backend != "trt" ||
      params.backend_map != expected_backend_map || params.trt_opt_profile != expected_profiles ||
      !params.is_engine_path || params.input_on_cuda || params.output_on_cuda ||
      params.transmit_on_cuda || !params.dynamic_input_dims || !params.output_dynamic_dims ||
      !params.enable_fp16 || params.enable_cuda_graphs || params.dla_core != 2 ||
      params.dla_core_map != expected_dla_map || params.dla_gpu_fallback ||
      params.parallel_inference || params.max_batch_size != 8 || params.max_memory != 4096U ||
      params.cuda_stream_pool == nullptr || params.cuda_green_context_pool == nullptr) {
    return false;
  }
  const auto& stream = params.cuda_stream_pool->config();
  const auto& green = params.cuda_green_context_pool->config();
  return stream.device_id == 1 && stream.stream_flags == 2U && stream.stream_priority == -2 &&
         stream.reserved_size == 2U && stream.max_size == 4U &&
         stream.nvtx_identifier == "registered-streams" && green.device_id == 1 &&
         green.num_partitions == 2U &&
         green.sms_per_partition == std::vector<std::int32_t>({8, 8}) &&
         green.default_context == 1 && green.nvtx_identifier == "registered-contexts";
}

}  // namespace

int main() {
  if (!holoscan::ops::register_inference_operator_catalog()) {
    std::cerr << "failed to register Inference authoring catalog\n";
    return 1;
  }
  if (!registered_parameters_are_preserved()) {
    std::cerr << "registered Inference parameter translation mismatch\n";
    return 1;
  }

  holoscan::ops::InferenceOpParams params;
  params.model_path_map = {{"model", "/tmp/model.onnx"}};
  params.input_map = {{"model", {"input"}}};
  params.output_map = {{"model", {"result"}}};
  params.input_on_cuda = false;
  params.output_on_cuda = false;
  params.transmit_on_cuda = false;
  params.enable_cuda_graphs = false;
  params.parallel_inference = false;

  holoscan::Graph graph{"inference-python-parity"};
  const auto clock = graph.add_clock<holoscan::ManualClock>("compile-clock");
  graph.set_default_clock(clock);
  constexpr std::array<std::int64_t, 1U> max_shape{1};
  holoscan::TopicOptions topic_options;
  topic_options.endpoint_name = "inference-input";
  topic_options.tensor = holoscan::TensorBoundaryContract{
      .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                         .dtype = DLDataType{kDLUInt, 8U, 1U},
                         .rank = 1U,
                         .strides = holoscan::TensorStridePolicy::kContiguousOnly},
      .bounds = holoscan::tensor_bounds(max_shape),
  };
  const auto input =
      graph.import_topic<holoscan::Tensor>("holomq://inference-input", std::move(topic_options));
  const auto inference = graph.op<holoscan::ops::InferenceOp>("inference", std::move(params));
  graph.add_flow(input.output(), *inference->input_port("input"));

  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  if (!plan.ok()) {
    std::cerr << plan.json();
    return 1;
  }
  std::cout << plan.json();
  return 0;
}
