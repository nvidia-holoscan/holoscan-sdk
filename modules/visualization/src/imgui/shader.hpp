/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Embedded Slang source for the ImGui pipeline. A string constant rather than
// a file on disk: library code has no resource path to load from (only
// examples ship files next to their binaries).

#pragma once

namespace holoviz::imgui::detail {

extern const char* const kSlangSource;

inline constexpr const char* kVertexEntryPoint = "vertex_main";
inline constexpr const char* kFragmentEntryPoint = "fragment_main";

}  // namespace holoviz::imgui::detail
