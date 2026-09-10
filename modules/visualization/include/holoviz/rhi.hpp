/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Umbrella header for the holoviz RHI. Pulls in the full public API.
// Typical bring-up order: create_window -> window.create_surface() ->
// create_device_for_surface -> Device::create_swapchain -> resources/pipelines
// -> frame loop (acquire / record / submit / present).

#pragma once

#include <holoviz/rhi/adapter.hpp>
#include <holoviz/rhi/buffer.hpp>
#include <holoviz/rhi/command_list.hpp>
#include <holoviz/rhi/common.hpp>
#include <holoviz/rhi/descriptor.hpp>
#include <holoviz/rhi/device.hpp>
#include <holoviz/rhi/display.hpp>
#include <holoviz/rhi/external_semaphore.hpp>
#include <holoviz/rhi/image.hpp>
#include <holoviz/rhi/pipeline.hpp>
#include <holoviz/rhi/query.hpp>
#include <holoviz/rhi/render_pass.hpp>
#include <holoviz/rhi/slang_compiler.hpp>
#include <holoviz/rhi/surface.hpp>
#include <holoviz/rhi/swapchain.hpp>
#include <holoviz/rhi/window.hpp>
