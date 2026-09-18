// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The concrete payloads behind the seam's opaque tokens (render_backend.hpp: FrameCommands /
// SyncToken / SurfaceHandle are shared_ptr<void> so nothing above the RenderBackend /
// PresentBackend seam depends on a concrete backend type). This header is the private
// contract between the module's HoloViz-based backends: WsiPresentBackend mints AcquirePayload,
// HolovizRenderBackend consumes its gates and mints SubmitPayload, WsiPresentBackend redeems
// both at present(). It is deliberately not installed — an out-of-tree backend pairing
// defines its own payload types.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <holoviz/common.hpp>
#include <holoviz/device.hpp>
#include <holoviz/image.hpp>
#include <holoviz/surface.hpp>
#include <holoviz/swapchain.hpp>
#include "holoviz_op/backends/render_backend.hpp"
#include "holoviz_op/types.hpp"

namespace holoscan::visualization::backends {

/// Process-wide provenance registry for the seam's type-erased payloads. The only UB-free
/// predicate over an arbitrary shared_ptr<void> from an untrusted caller is comparison
/// against a set of addresses this module itself minted — any scheme that inspects the
/// pointee (a magic field, an envelope header, a vtable) dereferences unknown memory first.
/// Payloads register their own address in their constructor and unregister in their
/// destructor, so an entry can never dangle and an address can never be reused while its
/// old entry exists. Entries also carry the producing device's identity, so a genuine
/// payload minted for one device is rejected before its gates/images reach another.
///
/// Inserts happen on setup paths only (payload pools, per-slot submit payloads, surface
/// handles); the frame path performs lock + hash lookups, allocation-free.
///
/// Hot-path cost, weighed deliberately: a steady-state frame does three check() calls
/// (begin_frame, submit, present), each one uncontended-mutex round-trip plus one find()
/// over ~a dozen entries — roughly +100 ns per call versus the magic-field probe it
/// replaced, i.e. noise next to the queue submit/present in the same frame. The single
/// process-wide mutex is shared by all backend instances, but each backend is
/// single-threaded and the critical sections are nanoseconds, so cross-device contention
/// is negligible at frame rates. If a profile ever disagrees, the cheap escapes are
/// caching the validated payload pointer per frame (drops submit()'s re-lookup) or a
/// read-optimized structure — checks vastly outnumber inserts.
class TokenRegistry {
 public:
  enum class Kind : uint8_t { kAcquire, kSubmit, kSurface, kTargetSet };

  /// Outcome of check(), ordered from success to most-foreign.
  enum class Check : uint8_t { kOk, kWrongDevice, kWrongKind, kUnknown };

  void insert(const void* ptr, Kind kind, uintptr_t device);
  void erase(const void* ptr);
  [[nodiscard]] Check check(const void* ptr, Kind kind, uintptr_t device) const;

 private:
  struct Entry {
    Kind kind;
    uintptr_t device;
  };

  // Backends are single-threaded, but the registry is shared by every backend instance in
  // the process, and independent device/backend pairs on different threads are legal.
  mutable std::mutex mutex_;
  std::unordered_map<const void*, Entry> entries_;
};

/// The single process-wide registry (a leaky singleton: never destroyed, so a payload
/// retained past static teardown can still unregister safely). Defined out of line in
/// backend_tokens.cpp rather than as a header-inline static: every consumer (the operator DSO,
/// the unit tests) links exactly one copy of the backend object library, so one definition
/// means one registry — a header-inline static compiled into several TUs could split.
[[nodiscard]] TokenRegistry& token_registry();

/// Stable identity of a device for provenance checks: the pImpl address survives handle
/// moves, and storing it as an integer keeps comparisons well-defined without ever
/// dereferencing it. Cross-backend tokens are validated by device identity, not producer
/// identity — a RenderBackend and a PresentBackend on the same device is the normal pairing.
[[nodiscard]] inline uintptr_t device_identity(const holoviz::Device& device) {
  return reinterpret_cast<uintptr_t>(&device.impl());
}

/// Producer-owned liveness cell published to token consumers. The present backend owns
/// one shared instance, advances current_generation on every swapchain (re)creation, and
/// releases it on destroy (expiring consumers' weak references). Consumers lock
/// AcquirePayload::source and compare generations BEFORE dereferencing
/// AcquirePayload::image — the borrowed image pointer and extent are only valid for the
/// generation that minted them.
struct AcquireSource {
  uint64_t current_generation = 0;

  /// One borrowed swapchain image of the current generation; index in `targets` ==
  /// swapchain image index.
  struct Target {
    const holoviz::Image* image = nullptr;
    holoviz::Extent2D extent{};
  };

  /// The current generation's swapchain images, for RenderBackend::register_targets() and
  /// the render backend's post-recreation re-warm. Rebuilt alongside current_generation on
  /// every (re)creation and cleared before the producer releases the cell at destroy();
  /// consumers lock and compare generations before dereferencing any entry — the same
  /// discipline as AcquirePayload::image.
  std::vector<Target> targets;
};

/// Payload behind AcquireToken::image_ready (and FrameTarget::acquire). Holds the WHOLE
/// move-only holoviz::AcquireSyncToken: Device::submit() consumes the gates in place (their
/// consumed flags live in swapchain-recycled slots; nothing is moved out), so the intact
/// token can later be moved into holoviz::Swapchain::present(). One shared_ptr is shared
/// between the present backend's pending table and the caller. Address-pinned (registered
/// with the token registry for its whole lifetime), so it is neither copyable nor movable;
/// pool reuse re-stamps the mutable fields in place.
struct AcquirePayload {
  explicit AcquirePayload(uintptr_t device) : device_key(device) {
    token_registry().insert(this, TokenRegistry::Kind::kAcquire, device);
  }
  ~AcquirePayload() { token_registry().erase(this); }
  AcquirePayload(const AcquirePayload&) = delete;
  AcquirePayload& operator=(const AcquirePayload&) = delete;
  AcquirePayload(AcquirePayload&&) = delete;
  AcquirePayload& operator=(AcquirePayload&&) = delete;

  /// Identity of the device this payload was minted for; immutable for the lifetime.
  const uintptr_t device_key;
  /// The swapchain image this frame renders into; borrowed from the swapchain
  /// (identity-stable per image index), filled by the present backend.
  const holoviz::Image* image = nullptr;
  holoviz::Extent2D extent{};
  holoviz::AcquireSyncToken token;
  /// Swapchain generation this payload was minted under; stale after resize().
  uint64_t generation = 0;
  /// Identity of the exact acquire() call that minted this payload (nonzero, strictly
  /// increasing per producer). Present() validates PresentArgs::acquire_issuance against
  /// it, so a retained stale acquire can never redeem a fresh acquire that reused its
  /// image index after a recreation (the index alone is recyclable; this is not).
  uint64_t issuance = 0;
  /// Liveness/generation of the producing present backend; lock and compare against
  /// `generation` before touching `image`/`extent`. Expired = the producer (or its
  /// swapchain) is gone.
  std::weak_ptr<const AcquireSource> source;
};

/// Payload behind the render_done SyncToken a RenderBackend's submit() returns.
/// Address-pinned like AcquirePayload.
struct SubmitPayload {
  explicit SubmitPayload(uintptr_t device) : device_key(device) {
    token_registry().insert(this, TokenRegistry::Kind::kSubmit, device);
  }
  ~SubmitPayload() { token_registry().erase(this); }
  SubmitPayload(const SubmitPayload&) = delete;
  SubmitPayload& operator=(const SubmitPayload&) = delete;
  SubmitPayload(SubmitPayload&&) = delete;
  SubmitPayload& operator=(SubmitPayload&&) = delete;

  /// Identity of the device this payload was minted for; immutable for the lifetime.
  const uintptr_t device_key;
  holoviz::SubmitSyncToken token;
};

/// Payload behind the SyncToken PresentBackend::targets() returns: a weak reference to
/// the producer's liveness cell, so a retained handle can never pin dead image pointers —
/// consumers lock and read AcquireSource::targets under the generation discipline.
/// Address-pinned like the other payloads.
struct TargetSetPayload {
  explicit TargetSetPayload(uintptr_t device) : device_key(device) {
    token_registry().insert(this, TokenRegistry::Kind::kTargetSet, device);
  }
  ~TargetSetPayload() { token_registry().erase(this); }
  TargetSetPayload(const TargetSetPayload&) = delete;
  TargetSetPayload& operator=(const TargetSetPayload&) = delete;
  TargetSetPayload(TargetSetPayload&&) = delete;
  TargetSetPayload& operator=(TargetSetPayload&&) = delete;

  /// Identity of the device this payload was minted for; immutable for the lifetime.
  const uintptr_t device_key;
  std::weak_ptr<const AcquireSource> source;
};

/// Result of the typed accessors: `payload` is non-null only when `check == kOk`, and the
/// cast behind it is safe — the registry hit proves the pointee registered its own address
/// from its own constructor, so the cast is to the object's true type. `check` lets call
/// sites distinguish a wrong-device genuine token from a foreign/mistyped one.
struct AcquireLookup {
  AcquirePayload* payload = nullptr;
  TokenRegistry::Check check = TokenRegistry::Check::kUnknown;
};

struct SubmitLookup {
  SubmitPayload* payload = nullptr;
  TokenRegistry::Check check = TokenRegistry::Check::kUnknown;
};

struct TargetSetLookup {
  TargetSetPayload* payload = nullptr;
  TokenRegistry::Check check = TokenRegistry::Check::kUnknown;
};

/// Typed accessors; a null token, a token minted by a different backend pairing, a token
/// of the other kind, or a token minted for a different device all yield a null payload —
/// callers turn that into kInvalidArgument. The registry lookup strictly precedes any
/// cast, so foreign memory is never dereferenced.
[[nodiscard]] inline AcquireLookup acquire_payload(const SyncToken& token, uintptr_t device) {
  void* raw = token.impl.get();
  if (raw == nullptr) {
    return {};
  }
  const TokenRegistry::Check check =
      token_registry().check(raw, TokenRegistry::Kind::kAcquire, device);
  if (check != TokenRegistry::Check::kOk) {
    return {nullptr, check};
  }
  return {static_cast<AcquirePayload*>(raw), check};
}

[[nodiscard]] inline SubmitLookup submit_payload(const SyncToken& token, uintptr_t device) {
  void* raw = token.impl.get();
  if (raw == nullptr) {
    return {};
  }
  const TokenRegistry::Check check =
      token_registry().check(raw, TokenRegistry::Kind::kSubmit, device);
  if (check != TokenRegistry::Check::kOk) {
    return {nullptr, check};
  }
  return {static_cast<SubmitPayload*>(raw), check};
}

[[nodiscard]] inline TargetSetLookup target_set_payload(const SyncToken& token, uintptr_t device) {
  void* raw = token.impl.get();
  if (raw == nullptr) {
    return {};
  }
  const TokenRegistry::Check check =
      token_registry().check(raw, TokenRegistry::Kind::kTargetSet, device);
  if (check != TokenRegistry::Check::kOk) {
    return {nullptr, check};
  }
  return {static_cast<TargetSetPayload*>(raw), check};
}

/// Recover the holoviz::Surface behind a SurfaceHandle minted by make_surface_handle();
/// nullptr for anything else. Surfaces are instance-level (minted before any device
/// exists), so they register with device identity 0 and no device check applies —
/// device/surface compatibility is validated downstream by surface_capabilities().
[[nodiscard]] inline std::shared_ptr<holoviz::Surface> surface_from_handle(
    const SurfaceHandle& handle) {
  const void* raw = handle.impl.get();
  if (raw == nullptr ||
      token_registry().check(raw, TokenRegistry::Kind::kSurface, 0) != TokenRegistry::Check::kOk) {
    return nullptr;
  }
  return std::static_pointer_cast<holoviz::Surface>(handle.impl);
}

}  // namespace holoscan::visualization::backends
