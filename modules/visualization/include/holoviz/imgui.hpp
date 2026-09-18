/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Umbrella header for the holoviz imgui module. Deliberately excludes
// imgui/ui.hpp, which is opt-in because it pulls the bundled Dear ImGui API
// into the consumer's context (only render_ui() callbacks need it).

#pragma once

#include <holoviz/imgui/renderer.hpp>
