// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// GPU-free bookkeeping regression for the examples' MeasuredQueryTracker
// (examples/common/), the per-swapchain-image GPU-query ownership the minimal text
// benchmark uses. Simulates a three-image warmup -> measured benchmark loop and proves the
// two failure modes the tracker exists to prevent never happen: an image's first measured
// acquire must not read the warmup-owned query range it still holds, and the end-of-run
// drain must surface exactly the last measured range per image — every measured submit
// sampled exactly once, no warmup submit ever.

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "measured_query_tracker.hpp"

#include <gtest/gtest.h>

namespace {

using holoviz::examples::MeasuredQueryTracker;

TEST(MeasuredQueryTracker, WarmupRangesAreNeverCollectedAndMeasuredRangesExactlyOnce) {
  constexpr uint32_t kImageCount = 3;
  constexpr uint64_t kWarmupFrames = 7;  // deliberately not a multiple of the image count
  constexpr uint64_t kMeasuredFrames = 8;
  MeasuredQueryTracker tracker(kImageCount);
  EXPECT_EQ(tracker.image_count(), kImageCount);

  // Each image's pending query range is tagged with the frame that submitted it; the
  // benchmark loop reads a range on the image's NEXT acquire, so the oracle records which
  // frame's range each collect actually returned.
  std::vector<uint64_t> range_owner(kImageCount, 0);
  std::vector<uint64_t> collected_frames;

  const auto run_frame = [&](uint64_t frame, bool collecting) {
    const uint32_t image = static_cast<uint32_t>(frame % kImageCount);  // rotating acquires
    if (tracker.should_collect(image)) {
      collected_frames.push_back(range_owner[image]);
      tracker.on_collect(image);
    }
    range_owner[image] = frame;  // the resubmit below rewrites the image's range
    tracker.on_submit(image, collecting);
  };

  for (uint64_t frame = 0; frame < kWarmupFrames; ++frame) {
    run_frame(frame, /*collecting=*/false);
  }
  EXPECT_TRUE(collected_frames.empty()) << "nothing may be collected during warmup";

  for (uint64_t frame = kWarmupFrames; frame < kWarmupFrames + kMeasuredFrames; ++frame) {
    run_frame(frame, /*collecting=*/true);
  }

  // End-of-run drain: flush each image's still-pending measured range.
  for (uint32_t image = 0; image < kImageCount; ++image) {
    if (tracker.should_collect(image)) {
      collected_frames.push_back(range_owner[image]);
      tracker.on_collect(image);
    }
  }

  // Exactly the measured frames, each exactly once, in whatever order — no warmup frame's
  // range (the pre-fix bug: each image's first measured acquire read its warmup range) and
  // no missing measured tail (the pre-fix bug: the last measured range per image was never
  // read).
  ASSERT_EQ(collected_frames.size(), kMeasuredFrames);
  std::vector<bool> seen(kWarmupFrames + kMeasuredFrames, false);
  for (const uint64_t frame : collected_frames) {
    EXPECT_GE(frame, kWarmupFrames) << "a warmup-owned query range was collected";
    EXPECT_FALSE(seen[frame]) << "frame " << frame << " sampled twice";
    seen[frame] = true;
  }

  // After the drain nothing is pending, and a fresh pass starts clean.
  for (uint32_t image = 0; image < kImageCount; ++image) {
    EXPECT_FALSE(tracker.should_collect(image));
  }
}

TEST(MeasuredQueryTracker, RejectsZeroImages) {
  EXPECT_THROW(MeasuredQueryTracker(0), std::invalid_argument);
}

}  // namespace
