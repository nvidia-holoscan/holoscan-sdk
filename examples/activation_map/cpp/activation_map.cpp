/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <unistd.h>

#include <climits>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gxf/core/entity.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/utils/cuda_macros.hpp>

constexpr int TENSOR_SIZE = 16;
namespace holoscan::ops {
class MakeTensorsOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(MakeTensorsOp)

  MakeTensorsOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<std::vector<ops::InferenceOp::ActivationSpec>>("models");
    spec.output<holoscan::gxf::Entity>("tensors");
    spec.param(allocator_, "allocator", "Allocator", "Allocator");
  }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    std::vector<ops::InferenceOp::ActivationSpec> models;
    auto tensors_entity = holoscan::gxf::Entity::New(&context);
    auto index_value = static_cast<float>(index_);
    add_tensor(tensors_entity, "first_preprocessed", index_value, context);
    add_tensor(tensors_entity, "second_preprocessed", index_value, context);
    add_tensor(tensors_entity, "third_preprocessed", index_value, context);
    models.emplace_back("first", 0);
    models.emplace_back("second", 0);
    models.emplace_back("third", 0);
    switch (index_) {
      case 0:
        // first model
        models[0].set_active();
        break;
      case 1:
        // second model
        models[1].set_active();
        break;
      case 2:
        // third model
        models[2].set_active();
        break;
      case 3:
        // first + second models
        models[0].set_active();
        models[1].set_active();
        break;
      case 4:
        // second + third models
        models[1].set_active();
        models[2].set_active();
        break;
      case 5:
        // third + first models
        models[0].set_active();
        models[2].set_active();
        break;
      default:
        // all models
        models[0].set_active();
        models[1].set_active();
        models[2].set_active();
        break;
    }
    op_output.emit(models, "models");
    op_output.emit(tensors_entity, "tensors");
    index_ = (++index_) % 7;
  }

 private:
  void add_tensor(holoscan::gxf::Entity& entity, const char* name, float value,
                  ExecutionContext& context) {
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                         allocator_->gxf_cid());
    auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>(name).value();
    tensor->reshape<float>(nvidia::gxf::Shape({TENSOR_SIZE}),
                           nvidia::gxf::MemoryStorageType::kDevice,
                           allocator.value());
    std::vector<float> hdata;
    hdata.resize(TENSOR_SIZE);
    for (auto i = 0; i < TENSOR_SIZE; i++) {
      hdata[i] = value;
    }
    HOLOSCAN_CUDA_CALL(cudaMemcpy(
        tensor->pointer(), hdata.data(), TENSOR_SIZE * sizeof(float), cudaMemcpyHostToDevice));
  }

  Parameter<std::shared_ptr<Allocator>> allocator_;
  int index_ = 0;
};

class PrintInferResultOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PrintInferResultOp)

  PrintInferResultOp() = default;
  void setup(OperatorSpec& spec) override {
    spec.input<std::vector<ops::InferenceOp::ActivationSpec>>("models");
    spec.input<holoscan::gxf::Entity>("models_result");
  }

  static void read_tensor(holoscan::gxf::Entity& entity, const std::string& name,
                          std::vector<float>& out) {
    out.resize(TENSOR_SIZE, 0.F);
    auto ret_tensor = entity.get<holoscan::Tensor>(name.c_str());
    if (!ret_tensor) {
      throw std::runtime_error(fmt::format("Tensor '{}' not found in message", name));
    }
    HOLOSCAN_CUDA_CALL(cudaMemcpy(
        out.data(), ret_tensor->data(), TENSOR_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  }
  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto models_message = op_input.receive<std::vector<ops::InferenceOp::ActivationSpec>>("models");
    if (!models_message) {
      HOLOSCAN_LOG_INFO("List of models message is null");
      return;
    }
    auto models = models_message.value();
    std::vector<std::string> selected_model;

    for (const auto& m : models) {
      if (m.is_active()) {
        selected_model.push_back(m.model());
      }
    }

    HOLOSCAN_LOG_INFO(
        fmt::format("Model {} was selected to infer", fmt::join(selected_model, ", ")));

    auto models_result_entity = op_input.receive<gxf::Entity>("models_result").value();
    std::vector<float> first_ret;
    std::vector<float> second_ret;
    std::vector<float> third_ret;
    read_tensor(models_result_entity, "first_output", first_ret);
    read_tensor(models_result_entity, "second_output", second_ret);
    read_tensor(models_result_entity, "third_output", third_ret);

    HOLOSCAN_LOG_INFO(fmt::format("  First model result: {}", fmt::join(first_ret, ", ")));
    HOLOSCAN_LOG_INFO(fmt::format("  Second model result: {}", fmt::join(second_ret, ", ")));
    HOLOSCAN_LOG_INFO(fmt::format("  Third model result: {}", fmt::join(third_ret, ", ")));
  }
};
}  // namespace holoscan::ops
class ActivationMapDemoApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto allocator = make_resource<UnboundedAllocator>("pool");

    // For model paths, need when debugging
    std::array<char, PATH_MAX> result{};
    ssize_t count = readlink("/proc/self/exe", result.data(), result.size());
    auto prog_path = std::string(result.data(), (count > 0) ? count : 0);
    size_t pos = prog_path.find_last_of("\\/");
    auto prog_dir = (std::string::npos == pos) ? "" : prog_path.substr(0, pos);

    ops::InferenceOp::DataMap model_path_map;
    model_path_map.insert("first", prog_dir + "/../models/dummy_addition_model_1.onnx");
    model_path_map.insert("second", prog_dir + "/../models/dummy_addition_model_2.onnx");
    model_path_map.insert("third", prog_dir + "/../models/dummy_addition_model_3.onnx");

    auto make_tensor_op = make_operator<ops::MakeTensorsOp>(
        "make_tensors", Arg("allocator") = allocator, make_condition<CountCondition>(7));
    auto infer_op = make_operator<ops::InferenceOp>("infer",
                                                    from_config("infers"),
                                                    Arg("model_path_map") = model_path_map,
                                                    Arg("allocator") = allocator);
    auto print_result_op = make_operator<ops::PrintInferResultOp>("print");
    add_flow(
        make_tensor_op, infer_op, {{"models", "model_activation_specs"}, {"tensors", "receivers"}});
    add_flow(make_tensor_op, print_result_op, {{"models", "models"}});
    add_flow(infer_op, print_result_op, {{"transmitter", "models_result"}});
  }
};

int main([[maybe_unused]] int argc, char** argv) {
  auto app = holoscan::make_application<ActivationMapDemoApp>();
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path /= std::filesystem::path("activation_map.yaml");
  if (argc > 1) {
    config_path = std::filesystem::path(argv[1]);
  }
  app->config(config_path);
  app->run();
  return 0;
}
