/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Embedded Slang source for the push-constant monospace text pipeline. A
// string constant rather than a file on disk: library code has no resource
// path to load from (only examples ship files next to their binaries).

#pragma once

namespace holoviz::text::detail {

extern const char* const kMonospaceSlangSource;

inline constexpr const char* kMonospaceVertexEntryPoint = "vertex_main";
inline constexpr const char* kMonospaceFragmentEntryPoint = "fragment_main";

}  // namespace holoviz::text::detail
