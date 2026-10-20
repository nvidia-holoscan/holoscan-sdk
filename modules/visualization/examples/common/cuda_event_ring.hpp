/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CudaEventSpanRing: a ring of CUDA event pairs for measuring per-frame stream
// spans without host-serializing the frame loop. Each frame records a
// start/end pair into slot (frame_index % depth); the span is collected a full
// ring lap later, when the events have long retired. The ring enforces the one
// ordering rule that makes the measurement valid: a slot must be collected
// BEFORE the current frame re-records it — collecting afterwards would
// host-sync on the just-enqueued work and serialize the loop, which is exactly
// the artifact the ring exists to avoid. Misuse throws instead of silently
// measuring the wrong frame.

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>

namespace holoviz::examples {

class CudaEventSpanRing {
 public:
  /// CUDA driver entry points the ring creates and destroys events through;
  /// tests inject failing fakes, production uses the defaults.
  using EventCreateFn = CUresult (*)(CUevent*, unsigned int);
  using EventDestroyFn = CUresult (*)(CUevent);

  /// Create \p depth event pairs. Depth must cover the frames a pair can stay
  /// in flight; swapchain image count + 1 keeps a slot's previous pair fully
  /// recorded before it is recycled. Strong guarantee: if any create fails,
  /// every event created so far is destroyed before the exception propagates.
  explicit CudaEventSpanRing(uint32_t depth, EventCreateFn event_create = &cuEventCreate,
                             EventDestroyFn event_destroy = &cuEventDestroy)
      : slots_(depth), event_destroy_(event_destroy) {
    if (depth == 0) {
      throw std::invalid_argument("CudaEventSpanRing: depth must be at least 1");
    }
    try {
      for (Slot& slot : slots_) {
        check(event_create(&slot.start, CU_EVENT_DEFAULT), "cuEventCreate");
        check(event_create(&slot.end, CU_EVENT_DEFAULT), "cuEventCreate");
      }
    } catch (...) {
      destroy_events();
      throw;
    }
  }

  CudaEventSpanRing(const CudaEventSpanRing&) = delete;
  CudaEventSpanRing& operator=(const CudaEventSpanRing&) = delete;
  CudaEventSpanRing(CudaEventSpanRing&&) = delete;
  CudaEventSpanRing& operator=(CudaEventSpanRing&&) = delete;

  ~CudaEventSpanRing() {
    // Callers drain the stream (cuStreamSynchronize / wait_idle) before
    // destruction, mirroring the previous open-coded rings.
    destroy_events();
  }

  /// Collect the span this frame's slot still holds from depth frames ago.
  /// Empty until the slot holds a completed pair (first ring lap, or an end
  /// event recorded on a later frame that has not arrived yet). Must run
  /// before record_start() for the same frame; collecting a slot the current
  /// frame already re-recorded throws.
  std::optional<float> collect(uint64_t frame_index) {
    const Slot& slot = slots_[slot_of(frame_index)];
    if (!slot.pair_complete) {
      return std::nullopt;
    }
    if (slot.start_frame == frame_index) {
      throw std::logic_error(
          "CudaEventSpanRing::collect called after the slot was re-recorded this frame; "
          "collect must precede record_start to avoid host-syncing on the current frame");
    }
    check(cuEventSynchronize(slot.end), "cuEventSynchronize");
    float span_ms = 0.0F;
    check(cuEventElapsedTime(&span_ms, slot.start, slot.end), "cuEventElapsedTime");
    return span_ms;
  }

  /// Record the span's start into this frame's slot.
  void record_start(uint64_t frame_index, CUstream stream) {
    Slot& slot = slots_[slot_of(frame_index)];
    check(cuEventRecord(slot.start, stream), "cuEventRecord");
    slot.start_frame = frame_index;
    slot.start_epoch = epoch_;
    slot.pair_complete = false;
  }

  /// Record the span's end for \p frame_index's slot. May run on a later
  /// frame's iteration (spans that close when control returns to CUDA), as
  /// long as it stays within one ring lap of record_start. An end whose start
  /// was recorded before the last reset() is dropped: it legitimately closes a
  /// pre-boundary span (e.g. the last warmup frame's), and completing the pair
  /// would resurrect it past the reset.
  void record_end(uint64_t frame_index, CUstream stream) {
    Slot& slot = slots_[slot_of(frame_index)];
    if (slot.start_epoch != epoch_) {
      return;  // delayed close of a span reset() already excluded
    }
    if (slot.start_frame != frame_index) {
      throw std::logic_error(
          "CudaEventSpanRing::record_end: the slot no longer holds this frame's start event "
          "(ring depth too small for how late the span closes)");
    }
    check(cuEventRecord(slot.end, stream), "cuEventRecord");
    slot.pair_complete = true;
  }

  /// Forget all recorded pairs (e.g. at a warmup/measurement boundary) so
  /// later collects only ever return spans recorded after the reset — even a
  /// pre-reset start whose end arrives afterwards stays excluded (the epoch
  /// bump orphans it). The events themselves stay; only the pair bookkeeping
  /// is cleared.
  void reset() {
    ++epoch_;
    for (Slot& slot : slots_) {
      slot.pair_complete = false;
    }
  }

  uint32_t depth() const { return static_cast<uint32_t>(slots_.size()); }

 private:
  struct Slot {
    CUevent start = nullptr;
    CUevent end = nullptr;
    // Frame that last recorded `start`, and whether `end` completed the pair;
    // together they detect both collect-after-re-record and a too-small ring.
    // start_epoch ties the start to a measurement window so a reset() between
    // start and end orphans the pair instead of letting the late end revive it.
    uint64_t start_frame = 0;
    uint64_t start_epoch = 0;
    bool pair_complete = false;
  };

  uint32_t slot_of(uint64_t frame_index) const {
    return static_cast<uint32_t>(frame_index % slots_.size());
  }

  void destroy_events() {
    for (Slot& slot : slots_) {
      if (slot.start != nullptr) {
        event_destroy_(slot.start);
        slot.start = nullptr;
      }
      if (slot.end != nullptr) {
        event_destroy_(slot.end);
        slot.end = nullptr;
      }
    }
    slots_.clear();
  }

  static void check(CUresult result, const char* what) {
    if (result != CUDA_SUCCESS) {
      const char* error_name = nullptr;
      cuGetErrorName(result, &error_name);
      throw std::runtime_error(std::string("CudaEventSpanRing: ") + what + " failed: " +
                               (error_name != nullptr ? error_name : "unknown CUDA error"));
    }
  }

  std::vector<Slot> slots_;
  EventDestroyFn event_destroy_ = &cuEventDestroy;
  // Measurement epoch, bumped by reset(); pairs only complete within one epoch.
  uint64_t epoch_ = 0;
};

}  // namespace holoviz::examples
