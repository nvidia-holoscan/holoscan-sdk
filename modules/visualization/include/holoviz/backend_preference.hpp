// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace holoviz {

/// The caller-facing knob for selecting the render backend. This is the user's *preference*;
/// the module resolves it to a concrete `RenderBackend` implementation (see
/// render_backend.hpp) — the two compose: `BackendPreference` chooses, `RenderBackend`
/// executes.
///
/// The built-in Vulkan backend ships first. Future ICD-level backends (e.g. corVK) are added here
/// as first-class options so selecting them never requires touching module internals.
enum class BackendPreference : uint8_t {
  kAuto,           ///< let the module pick the best available backend for the platform
  kBuiltinVulkan,  ///< the built-in native Vulkan backend (the initial default)
};

}  // namespace holoviz
