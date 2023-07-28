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

#include "test_core.hpp"

#include <functional>
#include <memory>
#include <string>
#include <utility>

void HoloInferTests::holoinfer_assert(const HoloInfer::InferStatus& status,
                                      const std::string& module, unsigned int current_test,
                                      const std::string& test_name,
                                      HoloInfer::holoinfer_code assert_type) {
  total_test_count++;
  if (status.get_code() != assert_type) {
    status.display_message();
    std::cerr << "Test " << current_test << ": " << test_name << " in " << module << " -> FAIL.\n";
    fail_test_count++;
  } else {
    std::cout << "Test " << current_test << ": " << test_name << " in " << module << " -> PASS.\n";
    pass_test_count++;
  }
}

void HoloInferTests::print_summary() {
  std::cout << "\nInference sub module test summary.\n\n";
  std::cout << "Tests executed   :\t" << total_test_count << "\n";
  std::cout << "Tests passed     :\t" << pass_test_count << " ("
            << 100.0 * (float(pass_test_count) / float(total_test_count)) << "%)\n";
  std::cout << "Tests failed     :\t" << fail_test_count << "\n\n";
}

void HoloInferTests::clear_specs() {
  inference_specs_.reset();
  holoscan_infer_context_.reset();
}

HoloInfer::InferStatus HoloInferTests::setup_inference() {
  holoscan_infer_context_ = std::make_unique<HoloInfer::InferContext>();
  auto status = holoscan_infer_context_->set_inference_params(inference_specs_);
  return status;
}

void HoloInferTests::setup_specifications() {
  // Create inference specification structure
  inference_specs_ = std::make_shared<HoloInfer::InferenceSpecs>(backend,
                                                                 backend_map,
                                                                 model_path_map,
                                                                 pre_processor_map,
                                                                 inference_map,
                                                                 device_map,
                                                                 is_engine_path,
                                                                 infer_on_cpu,
                                                                 parallel_inference,
                                                                 enable_fp16,
                                                                 input_on_cuda,
                                                                 output_on_cuda);
}

HoloInfer::InferStatus HoloInferTests::create_specifications() {
  setup_specifications();
  return setup_inference();
}

HoloInfer::InferStatus HoloInferTests::call_parameter_check_inference() {
  return HoloInfer::inference_validity_check(
      model_path_map, pre_processor_map, inference_map, in_tensor_names, out_tensor_names);
}

HoloInfer::InferStatus HoloInferTests::prepare_for_inference() {
  clear_specs();

  auto status = create_specifications();

  for (const auto td : in_tensor_dimensions) {
    auto db = std::make_shared<HoloInfer::DataBuffer>();
    size_t buffer_size =
        std::accumulate(td.second.begin(), td.second.end(), 1, std::multiplies<size_t>());

    db->device_buffer->resize(buffer_size);
    db->host_buffer.resize(buffer_size);
    inference_specs_->data_per_tensor_.insert({td.first, std::move(db)});
  }

  return status;
}

HoloInfer::InferStatus HoloInferTests::do_inference() {
  return holoscan_infer_context_->execute_inference(inference_specs_->data_per_tensor_,
                                                    inference_specs_->output_per_model_);
}
