/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/authoring/constructor_handle.hpp>

#include <cstdint>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

namespace holoscan::authoring {

void* box_constructor_handle(const ConstructorHandle& handle) {
  return new ConstructorHandle(handle);
}

void release_constructor_handle_box(void* box) noexcept {
  delete static_cast<ConstructorHandle*>(box);
}

const void* constructor_handle_box_identity() noexcept {
  // The address of this object is unique to one loaded copy of Holoscan Core.
  static const char identity{};
  return &identity;
}

const ConstructorHandleType<const ConstructorCallback>& callback_handle_type() noexcept {
  static constexpr ConstructorHandleDescriptor kDescriptor{
      .name = "holoscan.VoidCallback",
      .description = "Callback invoked with no arguments, such as a device start or stop action."};
  static const ConstructorHandleType<const ConstructorCallback> type{kDescriptor};
  return type;
}

namespace detail {

std::int64_t constructor_handle_process_id() noexcept {
#if defined(_WIN32)
  return static_cast<std::int64_t>(_getpid());
#else
  return static_cast<std::int64_t>(getpid());
#endif
}

}  // namespace detail
}  // namespace holoscan::authoring
