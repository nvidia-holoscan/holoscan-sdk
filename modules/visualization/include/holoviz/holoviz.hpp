/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Umbrella header for HoloViz. Pulls in the full public core API (text and imgui are separate).
// Typical bring-up order: create_window -> window.create_surface() ->
// create_device_for_surface -> Device::create_swapchain -> resources/pipelines
// -> frame loop (acquire / record / submit / present).

#pragma once

#include <holoviz/adapter.hpp>
#include <holoviz/buffer.hpp>
#include <holoviz/command_list.hpp>
#include <holoviz/common.hpp>
#include <holoviz/descriptor.hpp>
#include <holoviz/device.hpp>
#include <holoviz/display.hpp>
#include <holoviz/external_semaphore.hpp>
#include <holoviz/image.hpp>
#include <holoviz/pipeline.hpp>
#include <holoviz/query.hpp>
#include <holoviz/render_pass.hpp>
#include <holoviz/slang_compiler.hpp>
#include <holoviz/surface.hpp>
#include <holoviz/swapchain.hpp>
#include <holoviz/window.hpp>
