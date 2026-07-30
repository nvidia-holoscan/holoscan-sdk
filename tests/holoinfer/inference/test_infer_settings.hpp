/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef HOLOINFER_INFERENCE_TEST_SETTINGS_HPP
#define HOLOINFER_INFERENCE_TEST_SETTINGS_HPP

#include <holoinfer.hpp>
#include <holoinfer_utils.hpp>

namespace HoloInfer = holoscan::inference;

static const bool is_x86_64 = !HoloInfer::is_platform_aarch64();

#endif
