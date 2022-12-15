/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#ifndef _TEST_HOLOSCAN_MULTIAI_H
#define _TEST_HOLOSCAN_MULTIAI_H

#include <string>

#include "test_infer_settings.hpp"

void holoinfer_assert(const HoloInfer::InferStatus& status, const std::string& module,
                      const std::string& test_name, HoloInfer::holoinfer_code assert_type);

void clear_specs();
HoloInfer::InferStatus create_specifications();
HoloInfer::InferStatus call_parameter_check();
void parameter_test();
void parameter_setup_test();
HoloInfer::InferStatus prepare_for_inference();
HoloInfer::InferStatus do_mapping();
HoloInfer::InferStatus do_inference();
void inference_tests();

#endif
