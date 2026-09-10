// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include <holoviz/backends/wsi_present_backend.hpp>
#include <holoviz/present_backend.hpp>
#include <holoviz/rhi/device.hpp>
#include <holoviz/rhi/surface.hpp>
#include <holoviz/rhi/swapchain.hpp>

#include "backends/backend_tokens.hpp"
#include "backends/wsi_frame_table.hpp"
#include "backends/wsi_present_tracker.hpp"

namespace holoviz::backends {

/// Resolve a requested present mode against what the surface supports. Pure policy, kept
/// header-inline so it is unit-testable without linking module-private symbols:
///  - kAuto walks the priority list kFifoLatestReady -> kFifo -> kImmediate and returns the
///    first supported mode.
///  - An explicit FIFO-family request (kFifo / kFifoRelaxed / kFifoLatestReady) that is
///    unsupported falls back down the same list — tear-free degradation, never a surprise
///    tearing mode (kImmediate in the list only matters for surfaces with no FIFO at all,
///    which the Vulkan spec does not permit but other substrates might).
///  - Explicit kMailbox / kImmediate are opt-ins with visible behavior; when unsupported
///    they fail with kInvalidArgument instead of silently changing tearing/latency.
[[nodiscard]] inline expected<PresentMode> resolve_present_mode(PresentMode preferred,
                                                                rhi::PresentModeSet supported) {
  const auto first_supported = [&supported]() -> expected<PresentMode> {
    // The kAuto priority order (present_mode.hpp): tear-free with stale-frame dropping when
    // available, plain vsync otherwise; kImmediate only when the surface has no FIFO at all.
    for (const PresentMode mode :
         {PresentMode::kFIFOLatestReady, PresentMode::kFIFO, PresentMode::kImmediate}) {
      if (supported.contains(mode)) {
        return mode;
      }
    }
    return make_unexpected(Error{.code = ErrorCode::kFailure,
                                 .message = "[holoviz] Surface reports no usable present mode."});
  };

  switch (preferred) {
    case PresentMode::kAuto:
      return first_supported();
    case PresentMode::kFIFO:
    // case PresentMode::kFIFORelaxed:
    case PresentMode::kFIFOLatestReady:
      if (supported.contains(preferred)) {
        return preferred;
      }
      // Tear-free requests degrade tear-free; the caller sees the substitution through
      // SwapchainInfo::active_mode (and the log line emitted at create_swapchain()).
      return first_supported();
    case PresentMode::kMailbox:
    case PresentMode::kImmediate:
      if (supported.contains(preferred)) {
        return preferred;
      }
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] The requested present mode is an explicit opt-in and "
                           "this surface does not support it."});
  }
  return make_unexpected(Error{.code = ErrorCode::kInvalidArgument,
                               .message = "[holoviz] Unknown present mode requested."});
}

/// The WSI implementation of the PresentBackend seam, on rhi::Swapchain. Single-threaded,
/// like every RHI handle; all calls belong on the render thread.
class WsiPresentBackend final : public PresentBackend {
 public:
  explicit WsiPresentBackend(rhi::Device& device) : device_(&device) {}
  ~WsiPresentBackend() override;

  [[nodiscard]] expected<CreatedSwapchainInfo> create_swapchain(const SwapchainDesc& desc) override;
  [[nodiscard]] expected<void> destroy(std::chrono::nanoseconds timeout) override;
  [[nodiscard]] expected<void> resize(Extent2D new_extent) override;
  [[nodiscard]] expected<AcquireToken> acquire(std::chrono::nanoseconds timeout) override;
  [[nodiscard]] expected<void> present(const PresentArgs& args) override;
  [[nodiscard]] expected<WaitResult> wait_for_present(PresentId id,
                                                      std::chrono::nanoseconds timeout) override;
  [[nodiscard]] expected<SwapchainInfo> swapchain_info() const override;
  [[nodiscard]] expected<SyncToken> targets() const override;
  PresentInfo info() const override;

  /// Sanctioned fault-injection access (tests only): the internally owned RHI swapchain,
  /// for rhi::vulkan::swapchain_fault_injection().
  rhi::Swapchain& swapchain_for_test() { return swapchain_; }

 private:
  /// The creation path of create_swapchain(): resolve the stored mode preference against
  /// fresh surface capabilities, clamp the stored image-count request, create the RHI
  /// swapchain, and derive all per-swapchain state (info, frame table, outcome tracker,
  /// payload pool).
  [[nodiscard]] expected<void> create_swapchain_internal();

  /// Reconcile cached state after any swapchain recreation — an explicit resize() or the
  /// RHI's automatic recovery inside acquire()/present() — detected through the RHI
  /// generation counter. Refreshes swapchain_info_, resizes the frame table / outcome
  /// tracker / payload pool when the image count changed, and retires pre-recreation
  /// present ids. No-op when the generation is unchanged.
  [[nodiscard]] expected<void> refresh_swapchain_state();

  /// Lease a payload for acquire(): reuse a pool entry only when the pool holds the sole
  /// reference (no frame-table entry, caller AcquireToken, or render-backend keep-alive
  /// can alias it). Single-threaded backend, so use_count is exact. When every pool entry
  /// is still referenced (the caller retains more than image_count + 2 tokens) this
  /// returns kBackpressured instead of allocating — the acquire path stays allocation-free
  /// and, called before the RHI acquire, exhaustion consumes nothing.
  [[nodiscard]] expected<std::shared_ptr<AcquirePayload>> lease_payload();

  /// Best-effort reconcile on frame-loop error paths: publishes any generation change
  /// (refresh_swapchain_state() is transactional and idempotent) but never masks the error
  /// it follows — a refresh failure here is logged, and the next acquire()/resize()
  /// retries the reconciliation.
  void reconcile_after_frame_error(const char* where);

  /// Give an acquired-but-unusable image back via an empty RHI present (the module's only
  /// release path; one stale frame may flash — an error-path cosmetic). Best-effort:
  /// logged on failure, at which point resize()/destroy() remain the recovery. Keeps the
  /// RHI's single outstanding acquire recoverable when acquire() fails after the image was
  /// already handed out.
  void release_unpresented_acquire(rhi::AcquireSyncToken&& token);

  /// Rebuild acquire_source_->targets for the current generation (image wrappers are
  /// per-generation), best-effort per index: an unresolvable image leaves a null entry
  /// that consumers skip. No-op without a liveness cell.
  void publish_targets(uint32_t image_count, rhi::Extent2D extent);

  rhi::Device* device_ = nullptr;
  std::shared_ptr<rhi::Surface> surface_;
  rhi::Swapchain swapchain_;
  SwapchainInfo swapchain_info_{};
  PresentMode preferred_mode_ = PresentMode::kAuto;
  uint32_t requested_image_count_ = 0;
  /// rhi::Swapchain::generation() as last reconciled; stamped into AcquirePayloads so a
  /// frame that straddled a recreation is detected (kFrameDropped) without touching its
  /// dead gates.
  uint64_t last_seen_generation_ = 0;
  /// Device reports present-id/present-wait: wait_for_present() uses the native display-done
  /// path for live queued ids instead of the render-done fallback.
  bool has_present_feedback_ = false;

  WsiFrameTable frame_table_;

  /// Issuance tombstones for pending acquires dropped wholesale by a changed-count
  /// recreation's frame-table resize: presenting one must answer the consumed-frame
  /// contract (kFrameDropped, id burned, wait_for_present() == kDropped), never a
  /// retryable "not acquired"/"out of range" for a frame that cannot be redeemed.
  WsiRetiredAcquireLog retired_acquires_;

  /// Exact per-id outcomes (queued/dropped/retired/never-issued/evicted) behind
  /// wait_for_present(); replaces the old high-water-mark + ring heuristics that defaulted
  /// unknown ids to "displayed".
  WsiPresentTracker tracker_;

  /// Producer liveness/generation cell stamped into every AcquirePayload (weak), so the
  /// render backend can reject a stale token before dereferencing its borrowed image.
  std::shared_ptr<AcquireSource> acquire_source_;

  /// Preallocated acquire payloads (image_count + 2), leased by exclusivity so the acquire
  /// path performs no steady-state allocation.
  std::vector<std::shared_ptr<AcquirePayload>> payload_pool_;
  size_t payload_pool_next_ = 0;

  /// Issuance counter behind AcquireToken::issuance / AcquirePayload::issuance: one
  /// nonzero, strictly increasing identity per acquire() for the backend's lifetime, so
  /// present() redeems the exact acquire and never a fresh one that recycled its image
  /// index. Never reset — recreations must not make an old issuance collide with a new one.
  uint64_t next_acquire_issuance_ = 0;
};

}  // namespace holoviz::backends
