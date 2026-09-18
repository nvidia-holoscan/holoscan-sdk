/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Runtime Slang -> SPIR-V compilation. Shaders are authored in Slang only;
// this is the development-time path that compiles source in-process and
// reflects its interface.

#pragma once

#include <holoviz/holoviz_export.h>

#include "common.hpp"
#include "pipeline.hpp"

namespace holoviz {

/// Compile one Slang source string containing both entry points and return
/// the bytecode plus reflected attributes, bindings (descriptor space 0
/// only), and push-constant size. User vertex inputs must be wrapped in a
/// struct parameter; a direct non-system-value varying input (e.g.
/// `float2 position : POSITION` as a bare parameter) is rejected with an
/// error. `SV_*` inputs — direct parameters or struct fields — are ignored
/// for attribute reflection. Sampled-image bindings must be plain
/// `Texture2D`; 1D/3D/cube and arrayed/multisampled/combined variants are
/// rejected because HoloViz `Image` is 2D-only. Array resource declarations (e.g.
/// `Texture2D x[2]`, sized or unsized) are rejected with an error for now —
/// Descriptor has no array bind operations, so such a layout could not be
/// populated. The first call initializes a process-wide
/// Slang global session that lives for the remainder of the process;
/// initialization is thread-safe and a failed initialization is sticky for
/// the process. ALL Slang access is serialized by one internal process-wide
/// mutex (Slang sessions are not thread-safe), so this is callable from any
/// thread — compiles are expected at setup time, not on the frame path. The
/// immortal session caches one module per compile for the life of the
/// process; that growth is bounded by the declared pipeline set and accepted.
[[nodiscard]] HOLOVIZ_EXPORT expected<ShaderBlob> compile_slang_shader(
    Backend backend, const char* vertex_entry_point, const char* fragment_entry_point,
    const char* source);

}  // namespace holoviz
