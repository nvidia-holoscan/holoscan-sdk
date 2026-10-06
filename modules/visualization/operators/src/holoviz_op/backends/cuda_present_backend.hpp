// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// The CUDA capture PresentBackend: the implementation of the PresentBackend seam behind
// CudaSinkDesc. There is no swapchain and no display — the backend owns a small ring of
// CUDA-exported render targets, and "presenting" a frame means resolving its pixels into
// caller-provided pitch-linear device memory (one copy) on the backend's CUDA stream.
//
// A zero-copy sibling that handed out the render target's CUarray directly was considered and
// removed for now to keep a single CUDA delivery path (see CudaSinkDesc).

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda.h>

#include <holoscan/visualization/holoviz_op.hpp>
#include <holoviz/device.hpp>
#include <holoviz/expected.hpp>
#include <holoviz/external_semaphore.hpp>
#include <holoviz/image.hpp>

#include "holoviz_op/backends/backend_tokens.hpp"
#include "holoviz_op/backends/present_backend.hpp"
#include "holoviz_op/backends/present_frame_table.hpp"
#include "holoviz_op/backends/present_tracker.hpp"
#include "holoviz_op/types.hpp"

namespace holoscan::visualization::backends {

/// Creation parameters of the CUDA capture backend.
struct CudaPresentBackendDesc {
  /// Pixel format of every output frame. Must be a packed single-plane format with a HoloViz
  /// render-target lowering (see common::to_holoviz_format and common::bytes_per_pixel) whose
  /// render target CUDA can import. The BGRA kinds have no CUDA array equivalent: they render
  /// into the RGBA sibling (reported by swapchain_info().format) and the resolve swaps the red
  /// and blue bytes in place, one extra pass over the destination.
  ImageFormat format = ImageFormat::kR8G8B8A8Unorm;
};

/// Destination of one pitch-linear resolve.
struct CudaResolveTarget {
  CUdeviceptr device_ptr = 0;  ///< device memory of at least row_pitch_bytes * height bytes
  size_t row_pitch_bytes = 0;  ///< at least width * bytes_per_pixel(format)
};

/// PresentBackend over a ring of CUDA-exported render targets.
///
/// Semantics of the generic protocol on this backend: create_swapchain() allocates the ring
/// (SwapchainDesc::preferred_extent must be nonzero; image_count defaults to 2; `surface` and
/// `framebuffer_srgb` are ignored — the format alone decides the sRGB question;
/// swapchain_info().format is the render-target format the renderer must build for, which is
/// the RGBA sibling of a BGRA output format); acquire() hands
/// out the first free ring image from a round-robin cursor and never blocks (kBackpressured only
/// when every image is acquired and unpresented); present() orders the resolve after the frame's
/// rendering on the backend stream and records the frame's completion event; wait_for_present()
/// reports \c kDisplayed once that resolve has completed (there is no display — "displayed" means
/// "consumable") and keeps answering \c kDisplayed for a completed id as long as the id is
/// tracked, so info() reports has_present_id / has_present_wait = true. Empty presents release
/// the image without output. resize() builds the replacement ring before releasing the current
/// one: a failed recreation leaves the old ring (and every token into it) intact and usable.
///
/// GPU ordering: one holoviz::ExternalSemaphore bound to the backend stream carries both
/// directions. acquire() signals it and stores the token in AcquirePayload::render_wait, so the
/// render into a reused image waits for the previous frame's resolve (and anything the consumer
/// ordered on the stream after it); present() makes the stream wait on the render's
/// SubmitSyncToken before the copy. set_resolve_target() additionally orders the backend stream
/// after everything already queued on the destination's writer stream, so a destination the
/// pool recycles under a consumer read that is still queued there cannot be overwritten early.
/// The backend owns its stream because the operator's activation stream is only borrowed per
/// activation and unavailable when the backend is created; callers join their own stream to
/// ready_event() before reading the destination.
///
/// Teardown: destroy() releases nothing until both GPU users of the ring are proven done — the
/// output side by drain() (the backend stream idle, which covers every resolve, every
/// render-completion wait including one a failed present() left queued without a copy, and the
/// writer-stream edges) and the render side by the external semaphore's own destroy(), which
/// waits for every submit that waited on an acquire token. A resolve whose completion event
/// could not be recorded (present() failed after the copy was enqueued) is unprovable to the
/// destination's owner: drain() then fails with kFailure until a later present of the same
/// image records an event behind it. When the destructor's bounded proof fails, the
/// ring, the semaphore import, and the stream are moved into a process-lifetime quarantine and
/// the device is poisoned (report_leaked_resources()), matching HolovizRenderBackend — a leak
/// beats freeing memory an in-flight copy still targets.
///
/// Single-threaded, like every HoloViz handle.
class CudaPresentBackend final : public PresentBackend {
 public:
  CudaPresentBackend(holoviz::Device& device, const CudaPresentBackendDesc& desc)
      : device_(&device), desc_(desc) {}
  ~CudaPresentBackend() override;

  CudaPresentBackend(const CudaPresentBackend&) = delete;
  CudaPresentBackend& operator=(const CudaPresentBackend&) = delete;
  CudaPresentBackend(CudaPresentBackend&&) = delete;
  CudaPresentBackend& operator=(CudaPresentBackend&&) = delete;

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

  // ---- typed extension beyond the seam ------------------------------------------------------

  /// Return the CUDA stream every resolve is ordered on (kInvalidState before create_swapchain()).
  [[nodiscard]] expected<CUstream> stream() const;

  /// Register the destination of the next present() of \p image_index. Call between acquire()
  /// and present() of that image (kInvalidState otherwise); a present() without a target fails
  /// with \c kInvalidArgument and leaves the frame pending (set a target and retry, or release
  /// the image via an empty present). The target is consumed by one resolve.
  ///
  /// \p writer_stream is the stream the destination's owner declared as its writer (for a pooled
  /// Tensor, the stream the write guard was opened on). The backend stream is made to wait for
  /// everything queued on it at this call, so the resolve lands after any consumer read of a
  /// recycled destination that is still pending there. Must be a stream of the backend device's
  /// CUDA context (the primary context in the default device mode); kInvalidArgument when null.
  [[nodiscard]] expected<void> set_resolve_target(uint32_t image_index, CudaResolveTarget target,
                                                  CUstream writer_stream);

  /// Return the event recorded on stream() after the most recent non-empty present() of
  /// \p image_index (kNotFound before the first such present; kFailure while that present's
  /// resolve is unprovable, see resolve_may_be_in_flight()). The same event is re-recorded by
  /// the next present() of that image, so wait on it before the image is presented again.
  [[nodiscard]] expected<CUevent> ready_event(uint32_t image_index) const;

  /// True when the most recent present() of \p image_index enqueued its resolve, whether or not
  /// that present() then succeeded: a failure after the copy was queued (an event record) leaves
  /// the destination's owner responsible for keeping it alive until ready_event() proves the
  /// copy complete — or, if no event covers the copy, until the backend is destroyed.
  [[nodiscard]] bool resolve_may_be_in_flight(uint32_t image_index) const;

  /// Test-only fault injection for the fallible steps of a non-empty present().
  struct FaultInjection {
    bool fail_resolve_enqueue = false;  ///< fail the resolve copy once, before it is queued
    bool fail_ready_record = false;     ///< fail the slot ready-event record once
    bool fail_present_record = false;   ///< fail the per-id completion record once
  };
  void set_fault_injection_for_tests(FaultInjection faults) { faults_ = faults; }

 private:
  /// One ring entry: the exported render target, its CUDA view, and the resolve bookkeeping.
  struct Slot {
    holoviz::Image image;
    CUarray array = nullptr;
    CUevent ready = nullptr;         ///< recorded on stream_ after each resolve of this image
    CUevent target_ready = nullptr;  ///< recorded on the writer stream by set_resolve_target()
    bool has_frame = false;
    bool resolve_enqueued = false;  ///< the latest present() of this image queued its copy
    bool resolve_unproven = false;  ///< ... and no event was recorded after it
    CudaResolveTarget target{};
    bool has_target = false;
  };

  /// Completion state of one tagged present, independent of the reusable image slots: the
  /// event is recorded right after the frame's resolve and is never re-recorded until the record
  /// itself is recycled (image_count + 2 records, the tracker's eviction bound), and a completed
  /// outcome is latched so a later frame's pending resolve cannot revoke it.
  struct PresentRecord {
    PresentId id = 0;
    CUevent done = nullptr;
    bool completed = false;
  };

  /// Everything a ring generation owns, built as a unit so a failed allocation releases the
  /// partial result without touching the live ring.
  struct Ring {
    std::vector<Slot> slots;
    std::vector<PresentRecord> records;
  };

  /// Owned GPU state moved out of a backend whose destructor could not prove quiescence; kept
  /// for the process lifetime, never destroyed (see HolovizRenderBackend::QuarantinedState).
  struct QuarantinedState {
    Ring ring;
    holoviz::ExternalSemaphore semaphore;
    CUstream stream = nullptr;
  };

  /// Allocate a ring for \p extent without publishing it (the fallible part of
  /// create_swapchain()/resize()); a failure releases whatever was built.
  [[nodiscard]] expected<Ring> build_ring(Extent2D extent, uint32_t image_count);
  /// Publish \p ring as the new generation: resets the cursor, tables, and targets.
  void commit_ring(Ring&& ring, Extent2D extent);
  /// Release \p ring's images and events (after the caller drained the stream).
  [[nodiscard]] expected<void> release_ring(Ring& ring, std::chrono::nanoseconds timeout);
  /// Move the live ring out and release it.
  [[nodiscard]] expected<void> destroy_ring(std::chrono::nanoseconds timeout);
  /// Block until everything queued on stream_ completed (bounded by \p timeout via stream
  /// polling; a zero timeout fails immediately with kBackpressured when busy). kFailure when a
  /// slot holds a resolve no event was recorded behind.
  [[nodiscard]] expected<void> drain(std::chrono::nanoseconds timeout);
  /// Enqueue the copy of \p slot's render target into its registered destination (and, for a
  /// BGRA output format, the in-place red/blue swap behind it). Marks the slot's resolve
  /// enqueued and unproven as soon as the copy is queued.
  [[nodiscard]] expected<void> resolve(Slot& slot);
  [[nodiscard]] expected<std::shared_ptr<AcquirePayload>> lease_payload();
  void publish_targets();
  /// Claim the next PresentRecord for \p id and record its event on stream_ (no-op for id 0).
  [[nodiscard]] expected<void> record_present(PresentId id);
  [[nodiscard]] PresentRecord* record_for_id(PresentId id);
  /// Destructor-only: move every owned GPU object into the quarantine and poison the device.
  void quarantine_owned_state() noexcept;

  static constexpr const char* kName = "cuda_sink";

  holoviz::Device* device_ = nullptr;
  CudaPresentBackendDesc desc_{};
  CUcontext cuda_context_ = nullptr;
  CUstream stream_ = nullptr;
  holoviz::ExternalSemaphore semaphore_;

  std::vector<Slot> slots_;
  uint32_t next_slot_ = 0;
  SwapchainInfo info_{};
  uint64_t generation_ = 0;

  PresentFrameTable frame_table_;
  RetiredAcquireLog retired_acquires_;
  PresentTracker tracker_;
  /// Completion records of the recent tagged presents (same eviction bound as the tracker).
  std::vector<PresentRecord> records_;
  size_t records_next_ = 0;

  std::shared_ptr<AcquireSource> acquire_source_;
  std::vector<std::shared_ptr<AcquirePayload>> payload_pool_;
  size_t payload_pool_next_ = 0;
  uint64_t next_acquire_issuance_ = 0;
  FaultInjection faults_{};
};

/// Create the CUDA capture backend on \p device (borrowed; must outlive the returned backend).
/// Fails with kInvalidArgument for an invalid device.
[[nodiscard]] expected<std::unique_ptr<CudaPresentBackend>> create_cuda_present_backend(
    holoviz::Device& device, const CudaPresentBackendDesc& desc);

}  // namespace holoscan::visualization::backends
