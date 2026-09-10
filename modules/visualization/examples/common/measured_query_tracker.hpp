/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// MeasuredQueryTracker: per-swapchain-image bookkeeping for benchmark GPU
// timestamp queries. Each image's query range is rewritten every time the
// image is resubmitted, so a resolved range is only meaningful while it still
// belongs to the frame that wrote it. The tracker tags each image's pending
// range with whether its submit happened inside the measured window: the
// acquire-time read then never samples a warmup frame's leftover range, and
// the end-of-run drain flushes exactly the measured ranges the loop left
// pending — each measured submit is collected exactly once.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace holoviz::examples {

class MeasuredQueryTracker {
 public:
  explicit MeasuredQueryTracker(uint32_t image_count) : pending_measured_(image_count, false) {
    if (image_count == 0) {
      throw std::invalid_argument("MeasuredQueryTracker: image_count must be at least 1");
    }
  }

  /// True when \p image holds an uncollected query range that a measured-window
  /// submit wrote — the only ranges that belong in the reported statistics.
  bool should_collect(uint32_t image) const { return pending_measured_.at(image); }

  /// The caller read \p image's range; forget it so it is never sampled twice.
  void on_collect(uint32_t image) { pending_measured_.at(image) = false; }

  /// A submit rewrote \p image's query range; \p measured says whether it
  /// happened inside the measured window (warmup submits overwrite and untag).
  void on_submit(uint32_t image, bool measured) { pending_measured_.at(image) = measured; }

  uint32_t image_count() const { return static_cast<uint32_t>(pending_measured_.size()); }

 private:
  std::vector<bool> pending_measured_;
};

}  // namespace holoviz::examples
