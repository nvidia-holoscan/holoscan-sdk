/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <holoviz/imgui/imgui.h>

namespace holoviz::text::detail {

/// Restores the previously current ImGui context on scope exit. Every text-module
/// entry point that switches the process-global current context takes one of these
/// as its first statement, so the caller's own ImGui context (if any) is unchanged
/// on return. Guards nest safely (LIFO restore). Do NOT use around
/// ImGui::DestroyContext of the guarded context — restoring a destroyed context
/// would leave a dangling current pointer; destroy() handles that case explicitly.
class ImGuiContextGuard {
 public:
  ImGuiContextGuard() : previous_(ImGui::GetCurrentContext()) {}
  ImGuiContextGuard(const ImGuiContextGuard&) = delete;
  ImGuiContextGuard& operator=(const ImGuiContextGuard&) = delete;
  ImGuiContextGuard(ImGuiContextGuard&&) = delete;
  ImGuiContextGuard& operator=(ImGuiContextGuard&&) = delete;
  ~ImGuiContextGuard() { ImGui::SetCurrentContext(previous_); }

 private:
  ImGuiContext* previous_;
};

}  // namespace holoviz::text::detail
