/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Opt-in header for Renderer::render_ui() callbacks: pulls in the bundled
// Dear ImGui API so a callback can build arbitrary debug UI. This is the only
// holoviz header that includes ImGui — the rest of the imgui and text APIs
// only forward-declare its types, so ordinary rendering never adds ImGui to
// a consumer's context. The bundled headers under <holoviz/imgui/vendor/> are the
// exact version and configuration libholoviz was compiled with, and the ImGui
// entry points are exported from libholoviz, so callbacks need no ImGui
// sources or libraries of their own:
//
//   backend.render_ui(extent, [&] {
//     ImGui::Begin("Debug");
//     ImGui::SliderFloat("speed", &speed, 0.0F, 10.0F);
//     ImGui::End();
//   });

#pragma once

#include <holoviz/imgui/vendor/imgui.h>

#include <holoviz/imgui/renderer.hpp>
