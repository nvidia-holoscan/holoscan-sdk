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
#ifndef _TEST_HOLOSCAN_INFER_SETTINGS_H
#define _TEST_HOLOSCAN_INFER_SETTINGS_H

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <holoinfer.hpp>
#include <holoinfer_utils.hpp>

namespace HoloInfer = holoscan::inference;

unsigned int test_count = 0;
bool is_x86_64 = !HoloInfer::is_platform_aarch64();

/// Default parameters for inference
std::string backend = "trt";  // NOLINT
std::vector<std::string> in_tensor_names = {
    "plax_cham_pre_proc", "aortic_pre_proc", "bmode_pre_proc"};
std::vector<std::string> out_tensor_names = {"plax_cham_infer", "aortic_infer", "bmode_infer"};

std::map<std::string, std::string> model_path_map = {
    {"plax_chamber", "../data/multiai_ultrasound/models/plax_chamber.onnx"},
    {"aortic_stenosis", "../data/multiai_ultrasound/models/aortic_stenosis.onnx"},
    {"bmode_perspective", "../data/multiai_ultrasound/models/bmode_perspective.onnx"}};

std::map<std::string, std::vector<std::string>> pre_processor_map = {
    {"plax_chamber", {"plax_cham_pre_proc"}},
    {"aortic_stenosis", {"aortic_pre_proc"}},
    {"bmode_perspective", {"bmode_pre_proc"}}};

std::map<std::string, std::string> inference_map = {{"plax_chamber", "plax_cham_infer"},
                                                    {"aortic_stenosis", "aortic_infer"},
                                                    {"bmode_perspective", "bmode_infer"}};

bool parallel_inference = true;
bool infer_on_cpu = false;
bool enable_fp16 = false;
bool input_on_cuda = true;
bool output_on_cuda = true;
bool is_engine_path = false;

const std::map<std::string, std::vector<int>> in_tensor_dimensions = {
    {"plax_cham_pre_proc", {320, 320, 3}},
    {"aortic_pre_proc", {300, 300, 3}},
    {"bmode_pre_proc", {320, 240, 3}},
};

/// Pointer to inference context.
std::unique_ptr<HoloInfer::InferContext> holoscan_infer_context_;

/// Pointer to multi ai inference specifications
std::shared_ptr<HoloInfer::MultiAISpecs> multiai_specs_;

#endif
