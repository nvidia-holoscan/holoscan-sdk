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

#include <memory>

#include "inference/test_core.hpp"
#include "processing/test_core.hpp"

int main() {
  try {
    std::unique_ptr<HoloInferTests> holoinfer_tests = std::make_unique<HoloInferTests>();
    holoinfer_tests->parameter_test_inference();
    holoinfer_tests->parameter_setup_test();
    holoinfer_tests->inference_tests();
    holoinfer_tests->clear_specs();

    std::unique_ptr<ProcessingTests> processor_tests = std::make_unique<ProcessingTests>();
    processor_tests->parameter_test();
    processor_tests->parameter_setup_test();

    holoinfer_tests->print_summary();
    processor_tests->print_summary();
  } catch (...) {
    std::cout << "Exception in executing tests.\n";
    std::exit(1);
  }
  // summary of tests
  return 0;
}
