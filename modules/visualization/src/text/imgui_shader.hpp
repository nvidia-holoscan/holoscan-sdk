/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Embedded Slang source for the ImGui pipeline. A string constant rather than
// a file on disk: library code has no resource path to load from (only
// examples ship files next to their binaries).

#pragma once

namespace holoviz::text::detail {

extern const char* const kImGuiSlangSource;

inline constexpr const char* kImGuiVertexEntryPoint = "vertex_main";
inline constexpr const char* kImGuiFragmentEntryPoint = "fragment_main";

}  // namespace holoviz::text::detail
