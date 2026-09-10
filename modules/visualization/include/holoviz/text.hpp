/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Umbrella header for the holoviz text module. Deliberately excludes
// imgui_ui.hpp, which is opt-in because it pulls the bundled Dear ImGui API
// into the consumer's context (only render_ui() callbacks need it).

#pragma once

#include <holoviz/text/imgui_backend.hpp>
#include <holoviz/text/monospace_text_renderer.hpp>
#include <holoviz/text/text_renderer.hpp>
