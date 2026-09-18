/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoviz/imgui/vendor/imgui.h>

namespace holoviz::imgui::detail {

/// Restores the previously current ImGui context on scope exit. Every imgui-module
/// entry point that switches the process-global current context takes one of these
/// as its first statement, so the caller's own ImGui context (if any) is unchanged
/// on return. Guards nest safely (LIFO restore). Do NOT use around
/// ImGui::DestroyContext of the guarded context — restoring a destroyed context
/// would leave a dangling current pointer; destroy() handles that case explicitly.
class ContextGuard {
 public:
  ContextGuard() : previous_(ImGui::GetCurrentContext()) {}
  ContextGuard(const ContextGuard&) = delete;
  ContextGuard& operator=(const ContextGuard&) = delete;
  ContextGuard(ContextGuard&&) = delete;
  ContextGuard& operator=(ContextGuard&&) = delete;
  ~ContextGuard() { ImGui::SetCurrentContext(previous_); }

 private:
  ImGuiContext* previous_;
};

}  // namespace holoviz::imgui::detail
