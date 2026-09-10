/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Opt-in header for ImGuiBackend::render_ui() callbacks: pulls in the bundled
// Dear ImGui API so a callback can build arbitrary debug UI. This is the only
// holoviz header that includes ImGui — the rest of the text API only
// forward-declares its types, so ordinary text rendering never adds ImGui to
// a consumer's context. The bundled headers under <holoviz/imgui/> are the
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

#include <holoviz/imgui/imgui.h>

#include <holoviz/text/imgui_backend.hpp>
