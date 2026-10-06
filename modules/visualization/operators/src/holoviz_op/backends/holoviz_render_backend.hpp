// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The HoloViz render backend: the graphics-substrate implementation of the RenderBackend seam,
// built on holoviz (pools and images at allocate(), pipelines at build(), record and
// submit per frame). Deliberately named for HoloViz and not for any graphics API — HoloViz
// hides which API drives the graphics hardware, and a future pure-CUDA backend compositing
// with kernels would be this backend's sibling behind the same seam.

#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <variant>
#include <vector>

#include <cuda.h>

#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/command_list.hpp>
#include <holoviz/device.hpp>
#include <holoviz/expected.hpp>
#include <holoviz/image.hpp>
#include <holoviz/render_pass.hpp>

#include "holoviz_op/backends/backend_tokens.hpp"
#include "holoviz_op/backends/holoviz_geometry_layer.hpp"
#include "holoviz_op/backends/holoviz_image_layer.hpp"
#include "holoviz_op/backends/holoviz_lut_layer.hpp"
#include "holoviz_op/backends/holoviz_text_layer.hpp"
#include "holoviz_op/backends/render_backend.hpp"
#include "holoviz_op/types.hpp"

namespace holoscan::visualization::backends {

struct HolovizRenderBackendDesc {
  /// Stream the backend queues its per-frame CUDA work on (frame-data uploads, producer
  /// ordering). nullptr = the backend creates a private DEFAULT-flag stream at allocate()
  /// time if any declared layer needs one. Default-flag deliberately, not non-blocking: on
  /// at least one driver, async copies into Vulkan-imported mipmapped arrays land only
  /// partially when enqueued on a CU_STREAM_NON_BLOCKING stream (see the workaround note
  /// at the creation site). A default-flag stream serializes with the legacy null stream —
  /// applications that need concurrency with legacy-null-stream work must pass their own
  /// stream here instead of relying on the fallback.
  CUstream stream = nullptr;
  const char* debug_name = nullptr;
};

/// Create the HoloViz render backend on `device`. The device is borrowed and must outlive the
/// returned backend; destroy the backend before the device.
[[nodiscard]] expected<std::unique_ptr<RenderBackend>> create_holoviz_render_backend(
    holoviz::Device& device, const HolovizRenderBackendDesc& desc);

/// The HoloViz implementation of the RenderBackend seam. Single-threaded, like every HoloViz
/// handle; all calls belong on the render thread.
class HolovizRenderBackend final : public RenderBackend {
 public:
  HolovizRenderBackend(holoviz::Device& device, const HolovizRenderBackendDesc& desc)
      : device_(&device), desc_(desc) {}
  ~HolovizRenderBackend() override;

  [[nodiscard]] expected<void> allocate(const LayerStack& layers,
                                        const RenderConfig& config) override;
  [[nodiscard]] expected<void> build(ImageFormat output_format) override;
  [[nodiscard]] expected<void> register_targets(const SyncToken& targets) override;
  [[nodiscard]] expected<FrameCommands> begin_frame(const AcquireToken& acquired,
                                                    uint32_t frame_index) override;
  [[nodiscard]] expected<void> record(FrameCommands& cmds, const RenderList& layers,
                                      const CompositeSpec& composite) override;
  [[nodiscard]] expected<SyncToken> submit(const FrameCommands& cmds,
                                           const SubmitContext& ctx) override;
  [[nodiscard]] expected<void> abort_frame(FrameCommands& cmds) override;
  [[nodiscard]] expected<BackendInfo> info() const override;
  /// \return current pool high-water marks, for debug/diagnostic use only.
  [[nodiscard]] expected<AllocationStats> stats() const override;

  /// Tear down all owned HoloViz resources (children-before-device rule). The destructor calls
  /// this and logs failures.
  [[nodiscard]] expected<void> destroy(std::chrono::nanoseconds timeout) override;

  /// Test-only readback: cumulative native framebuffer creations in the underlying pass
  /// (see holoviz::detail::RenderPassImpl::render_targets_created()) — the instrumentation
  /// the allocation-free-frame tests freeze-check.
  [[nodiscard]] uint64_t render_targets_created_for_test() const;

  /// Headless-mode test access: the offscreen target image a frame slot renders into (for
  /// Device::download_from_image readback after wait_on_submit on the frame's token).
  /// Header-inline so tests can call it without linking module-private symbols.
  [[nodiscard]] expected<std::reference_wrapper<const holoviz::Image>> headless_target(
      uint32_t frame_index) const {
    if (!config_.headless || frame_index >= slots_.size() || stage_ == Stage::kCreated) {
      return make_unexpected(
          Error{.code = ErrorCode::kInvalidArgument,
                .message = "[holoviz] headless_target() needs an allocated headless backend "
                           "and a valid slot index."});
    }
    return std::cref(slots_[frame_index].headless_target);
  }

 private:
  enum class Stage : uint8_t { kCreated, kAllocated, kBuilt, kDestroyed };

  /// Payload behind FrameCommands: pre-created per slot at allocate() and recycled every
  /// time the slot cycles, so the frame path performs no allocation.
  struct FrameState {
    uint32_t slot = 0;
    const holoviz::Image* target = nullptr;
    holoviz::Extent2D extent{};
    /// Keep-alive for the AcquirePayload whose gates ride this frame's submit.
    std::shared_ptr<void> acquire;
    /// External tokens collected during record() (text uploads, CUDA producers); capacity
    /// reserved at allocate().
    std::vector<holoviz::ExternalSyncToken> waits;
    bool open = false;  ///< between begin_frame() and submit()
  };

  /// Immutable-per-issuance envelope behind FrameCommands. find_ticket() validates a
  /// handle by pointer identity against the slots' ticket registries (a foreign
  /// shared_ptr<void> payload is compared by address, never dereferenced) and by epoch
  /// (a handle from a previous issuance of the slot is stale). A ticket is only
  /// re-stamped when no outstanding handle references it (use_count() == 1), so a stale
  /// handle always reads the epoch of the issuance that minted it; when both of a slot's
  /// tickets are still retained, begin_frame() refuses with kBackpressured instead of
  /// allocating a replacement — there is no frame-path allocation.
  struct FrameTicket {
    uint32_t slot = 0;
    uint64_t epoch = 0;
    std::shared_ptr<FrameState> state;  ///< keep-alive; dereferenced only after a registry hit
  };

  /// Payload behind the SyncToken submit() returns; pre-created per slot and updated in
  /// place. The token value only ever moves forward (it is the slot's latest submit), so a
  /// stale holder waits on a newer submit — later than its frame, never dangling.
  struct FrameSlot {
    holoviz::CommandList cmd;
    holoviz::SubmitSyncToken last_submit{};
    holoviz::Image headless_target;  ///< empty unless RenderConfig::headless
    std::shared_ptr<FrameState> state;
    std::shared_ptr<SubmitPayload> submit_payload;
    uint64_t current_epoch = 0;  ///< epoch of the latest begin_frame() issuance
    /// Per-slot ticket pool: the retention bound is the handles from the current and the
    /// immediately previous issuance. Reuse is gated on exclusivity; with both tickets
    /// still retained, begin_frame() fails kBackpressured before touching the slot.
    std::array<std::shared_ptr<FrameTicket>, 2> tickets;
  };

  /// Per-declared-layer backend state, index-aligned with the LayerBundle. One alternative
  /// per implemented layer kind; the remaining kinds land in their stages, and adding an
  /// alternative turns every non-exhaustive visit into a compile error.
  using LayerState = std::variant<HolovizTextLayerState, HolovizGeometryLayerState,
                                  HolovizImageLayerState, HolovizLUTLayerState>;

  /// Registry-scan lookup behind every FrameCommands consumer: kInvalidArgument for a
  /// null/foreign handle; a registry hit returns the ticket plus whether a later
  /// begin_frame() superseded it (stale). Callers pick their stale policy — record()/
  /// submit() reject (via frame_state()), abort_frame() treats it as a no-op.
  struct TicketLookup {
    const FrameTicket* ticket = nullptr;
    bool stale = false;
  };
  [[nodiscard]] expected<TicketLookup> find_ticket(const FrameCommands& cmds) const;

  /// The mutating body of allocate(), after its precondition checks. It appends to the
  /// owned-state members as it goes, so a failure leaves partial state behind — allocate()
  /// then releases that state and makes the failure terminal (stage kDestroyed), because a
  /// retry over the leftovers would misalign layer_states_ with the declared bundle and
  /// re-create already-created per-slot resources.
  [[nodiscard]] expected<void> allocate_impl(const LayerStack& layers, const RenderConfig& config);

  /// find_ticket() with the strict policy: stale handles fail with kInvalidState.
  [[nodiscard]] expected<FrameState*> frame_state(const FrameCommands& cmds) const;

  /// Terminal quarantine bundle for a destructor whose bounded drain timed out: the owned
  /// handles are moved in and DELIBERATELY never destroyed (the process-lifetime list
  /// holding these bundles is never deleted), because the pending GPU/CUDA work still
  /// references their resources and every per-handle destructor frees unconditionally —
  /// freeing would be a use-after-free, not a leak.
  struct QuarantinedState {
    std::vector<FrameSlot> slots;
    holoviz::RenderPass render_pass;
    std::vector<LayerState> layer_states;
    holoviz::ExternalSemaphore upload_semaphore;
    CUevent drain_event = nullptr;
    CUstream stream = nullptr;  ///< only when the backend owned it; borrowed stays out
  };

  /// Move every owned member into a quarantine bundle and poison the device
  /// (report_leaked_resources()), so neither member destruction nor device teardown frees
  /// resources the timed-out drain proved are still referenced. Destructor-only.
  void quarantine_owned_state() noexcept;

  holoviz::Device* device_ = nullptr;
  HolovizRenderBackendDesc desc_{};
  Stage stage_ = Stage::kCreated;
  RenderConfig config_{};
  uint32_t declared_layer_count_ = 0;
  std::vector<uint8_t> composite_seen_;  // per-record duplicate-index scratch (reused)
  std::vector<LayerState> layer_states_;

  holoviz::RenderPass render_pass_;
  std::vector<FrameSlot> slots_;

  /// The AcquireSource generation whose targets have pre-created framebuffers (via
  /// register_targets() or the begin_frame() re-warm). A payload from a newer generation
  /// triggers a one-frame re-warm of every current target — the documented allocation
  /// exception on the recreation frame; steady state is a single integer compare.
  uint64_t warmed_generation_ = 0;

  /// CUDA upload path, created at allocate() when any declared layer stages frame data
  /// through device memory (geometry/image/LUT). The semaphore orders the async copies
  /// against the device timeline; the context is pushed around every CUDA call the backend
  /// makes so callers need no ambient context.
  CUcontext cuda_context_ = nullptr;
  CUstream stream_ = nullptr;
  bool owns_stream_ = false;
  holoviz::ExternalSemaphore upload_semaphore_;
  /// Bounded-drain marker for abort/destroy: recorded on stream_ at the wait site and
  /// polled under a timeout, replacing unbounded cuStreamSynchronize. Created at
  /// allocate() alongside the stream; covers borrowed streams too.
  CUevent drain_event_ = nullptr;

  AllocationStats stats_{};
};

}  // namespace holoscan::visualization::backends
