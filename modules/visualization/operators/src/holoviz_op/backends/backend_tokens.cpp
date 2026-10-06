// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "holoviz_op/backends/backend_tokens.hpp"

#include <memory>
#include <utility>

#include "holoviz_op/types.hpp"

namespace holoscan::visualization::backends {

void TokenRegistry::insert(const void* ptr, Kind kind, uintptr_t device) {
  const std::lock_guard<std::mutex> lock(mutex_);
  entries_.insert_or_assign(ptr, Entry{kind, device});
}

void TokenRegistry::erase(const void* ptr) {
  const std::lock_guard<std::mutex> lock(mutex_);
  entries_.erase(ptr);
}

TokenRegistry::Check TokenRegistry::check(const void* ptr, Kind kind, uintptr_t device) const {
  const std::lock_guard<std::mutex> lock(mutex_);
  const auto entry = entries_.find(ptr);
  if (entry == entries_.end()) {
    return Check::kUnknown;
  }
  if (entry->second.kind != kind) {
    return Check::kWrongKind;
  }
  if (entry->second.device != device) {
    return Check::kWrongDevice;
  }
  return Check::kOk;
}

TokenRegistry& token_registry() {
  // Leaky singleton: never destroyed, so it survives any static-destruction ordering — a
  // payload retained in a static that dies after main() can still run its destructor's
  // erase() against a live registry.
  static auto* registry = new TokenRegistry();
  return *registry;
}

SurfaceHandle wrap_surface_handle(holoviz::Surface&& surface) {
  auto* raw = new holoviz::Surface(std::move(surface));
  token_registry().insert(raw, TokenRegistry::Kind::kSurface, 0);
  return SurfaceHandle{std::shared_ptr<holoviz::Surface>(raw, [](holoviz::Surface* ptr) {
    token_registry().erase(ptr);
    delete ptr;
  })};
}

}  // namespace holoscan::visualization::backends
