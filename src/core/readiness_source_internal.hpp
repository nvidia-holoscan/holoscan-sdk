/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <holoscan/core/readiness_source.hpp>

namespace holoscan::detail {

// Private Runtime/test construction seam. Installed headers expose only an incomplete friend so
// applications cannot invoke binding operations or manufacture sender lifecycle authority.
struct ReadinessSourceAccess {
  template <ReadinessSourceKind Kind>
  [[nodiscard]] static constexpr BasicSourceRef<Kind> make_ref(
      SourceId id, std::uint64_t plan_provenance) noexcept {
    return BasicSourceRef<Kind>(id, plan_provenance);
  }

  static void bind(ReadableSource& source, ReadableSourceRef ref) noexcept { source.ref_ = ref; }
  static void bind(NotificationSource& source, NotificationSourceRef ref) noexcept {
    source.ref_ = ref;
  }
  static void bind(DoorbellSource& source, DoorbellSourceRef ref) noexcept { source.ref_ = ref; }
  static void bind(SpinSource& source, SpinSourceRef ref) noexcept { source.ref_ = ref; }

  [[nodiscard]] static ReadableSourceRef ref(const ReadableSource& source) noexcept {
    return source.ref_;
  }
  [[nodiscard]] static NotificationSourceRef ref(const NotificationSource& source) noexcept {
    return source.ref_;
  }
  [[nodiscard]] static DoorbellSourceRef ref(const DoorbellSource& source) noexcept {
    return source.ref_;
  }
  [[nodiscard]] static SpinSourceRef ref(const SpinSource& source) noexcept { return source.ref_; }

  [[nodiscard]] static NotificationSender make_sender(
      void* state, void (*retain)(void*) noexcept, void (*release)(void*) noexcept,
      NotificationSender::PostFunction post) noexcept {
    const bool complete_lifecycle = state != nullptr && retain != nullptr && release != nullptr;
    return NotificationSender(
        complete_lifecycle ? ValueLifecycle(state, retain, release) : ValueLifecycle{}, post);
  }
};

}  // namespace holoscan::detail
