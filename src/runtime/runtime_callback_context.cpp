/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "runtime_callback_context.hpp"

namespace holoscan::detail {
namespace {

thread_local RuntimeCallbackIdentity runtime_callback_identity_state{};

}  // namespace

RuntimeCallbackContextScope::RuntimeCallbackContextScope(RuntimeCallbackContext context) noexcept
    : RuntimeCallbackContextScope(RuntimeCallbackIdentity{.context = context}) {}

RuntimeCallbackContextScope::RuntimeCallbackContextScope(RuntimeCallbackIdentity identity) noexcept
    : previous_(runtime_callback_identity_state) {
  runtime_callback_identity_state = identity;
}

RuntimeCallbackContextScope::~RuntimeCallbackContextScope() noexcept {
  runtime_callback_identity_state = previous_;
}

bool runtime_callback_context_restricted() noexcept {
  return runtime_callback_identity_state.context != RuntimeCallbackContext::kNone;
}

RuntimeCallbackIdentity runtime_callback_identity() noexcept {
  return runtime_callback_identity_state;
}

}  // namespace holoscan::detail
