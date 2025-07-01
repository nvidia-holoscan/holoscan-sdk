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

#include "test_core.hpp"

#include <memory>
#include <string>

void ProcessingTests::processing_assert(const HoloInfer::InferStatus& status,
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

void ProcessingTests::print_summary() {
  std::cout << "\nProcessing sub module test summary.\n\n";
  std::cout << "Tests executed   :\t" << total_test_count << "\n";
  std::cout << "Tests passed     :\t" << pass_test_count << " ("
            << 100.0 * (float(pass_test_count) / float(total_test_count)) << "%)\n";
  std::cout << "Tests failed     :\t" << fail_test_count << "\n\n";
}

int ProcessingTests::get_status() {
  if (fail_test_count > 0) return 1;
  return 0;
}

void ProcessingTests::clear_processor() {
  holoscan_processor_context_.reset();
}

HoloInfer::InferStatus ProcessingTests::call_parameter_check_processing() {
  return HoloInfer::processor_validity_check(
      processed_map, in_tensor_processing, out_tensor_processing);
}

HoloInfer::InferStatus ProcessingTests::setup_processor(bool use_cuda_graphs) {
  holoscan_processor_context_ = std::make_unique<HoloInfer::ProcessorContext>();
  auto status = holoscan_processor_context_->initialize(
      process_operations, custom_kernels, use_cuda_graphs, config_path);
  return status;
}

HoloInfer::InferStatus ProcessingTests::execute_processor() {
  auto status = holoscan_processor_context_->process(process_operations,
                                                     processed_map,
                                                     data_per_tensor,
                                                     dims_per_tensor,
                                                     process_with_cuda,
                                                     cuda_stream);
  return status;
}
