// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Bookkeeping tests for the examples' CudaEventSpanRing (examples/common/): steady-state
// collection stays one ring lap behind recording, incomplete pairs and first laps return
// nothing, the collect-after-re-record misuse — the bug the ring exists to prevent, which
// silently turns a benchmark loop into per-frame host synchronization — throws, reset()
// fences a warmup/measurement boundary so every measured frame is sampled exactly once,
// and a partially failed construction destroys the events it already created (GPU-free,
// via the injectable create/destroy seam).

#include <cstdint>
#include <optional>
#include <stdexcept>

#include <cuda.h>

#include "cuda_event_ring.hpp"
#include "image_compare_test_util.hpp"

#include <gtest/gtest.h>

#include "holoviz_test_skip.hpp"

namespace {

using holoviz::examples::CudaEventSpanRing;

class CudaEventRingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = holoviz::test_util::try_create_device();
    if (!device_) {
      HOLOVIZ_ENV_SKIP() << "no usable GPU/adapter on this machine";
    }
    const auto context = device_.cuda_context();
    ASSERT_TRUE(context.has_value()) << context.error().message;
    if (cuCtxSetCurrent(*context) != CUDA_SUCCESS ||
        cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING) != CUDA_SUCCESS) {
      HOLOVIZ_ENV_SKIP() << "cannot create a CUDA stream on the device context";
    }
  }

  void TearDown() override {
    if (stream_ != nullptr) {
      cuStreamSynchronize(stream_);
      cuStreamDestroy(stream_);
    }
    if (device_) {
      EXPECT_TRUE(device_.destroy().has_value());
    }
  }

  holoviz::Device device_;
  CUstream stream_ = nullptr;
};

TEST_F(CudaEventRingTest, CollectsOneLapBehindAndRejectsMisuse) {
  constexpr uint32_t kDepth = 3;
  CudaEventSpanRing ring(kDepth);
  EXPECT_EQ(ring.depth(), kDepth);

  for (uint64_t frame = 0; frame < 10; ++frame) {
    const std::optional<float> collected = ring.collect(frame);
    if (frame < kDepth) {
      EXPECT_FALSE(collected.has_value()) << "frame " << frame << ": slot never recorded";
    } else {
      if (!collected.has_value()) {
        FAIL() << "frame " << frame;
      }
      EXPECT_GE(*collected, 0.0F);
    }
    ring.record_start(frame, stream_);
    ring.record_end(frame, stream_);
  }

  // The ordering bug this ring prevents: re-record first, collect after — the collect
  // would host-sync on the current frame's work and measure the wrong span.
  ring.record_start(10, stream_);
  ring.record_end(10, stream_);
  EXPECT_THROW(static_cast<void>(ring.collect(10)), std::logic_error);
}

TEST_F(CudaEventRingTest, IncompletePairsAndLateEndsAreHandled) {
  CudaEventSpanRing ring(2);

  // start without end: the pair is incomplete a lap later, not a bogus sample.
  ring.record_start(0, stream_);
  EXPECT_FALSE(ring.collect(2).has_value());

  // end recorded on a later frame (a span closing when control returns), still collectable
  // one lap after the start.
  ring.record_start(2, stream_);
  ring.record_end(2, stream_);
  ring.record_start(3, stream_);
  ring.record_end(3, stream_);
  EXPECT_TRUE(ring.collect(4).has_value());

  // a ring too shallow for how late the end lands throws instead of timing garbage.
  ring.record_start(4, stream_);
  ring.record_start(5, stream_);
  ring.record_start(6, stream_);  // recycles frame 4's slot
  EXPECT_THROW(ring.record_end(4, stream_), std::logic_error);
}

// The warmup/measurement boundary contract benchmarks rely on: reset() excludes every span
// recorded before it, and the end-of-run drain (collect(N)..collect(N + depth - 1) after the
// loop leaves frame N) flushes exactly the last ring lap — so each measured frame is sampled
// exactly once and no warmup frame ever is.
TEST_F(CudaEventRingTest, ResetExcludesPriorSamplesAndDrainFlushesTheWindow) {
  constexpr uint32_t kDepth = 3;
  constexpr uint64_t kWarmupFrames = 5;
  constexpr uint64_t kFrames = 15;  // frames [5, 15) are the measured window
  CudaEventSpanRing ring(kDepth);

  for (uint64_t frame = 0; frame < kWarmupFrames; ++frame) {
    ring.record_start(frame, stream_);
    ring.record_end(frame, stream_);
  }
  ring.reset();

  uint64_t collected = 0;
  for (uint64_t frame = kWarmupFrames; frame < kFrames; ++frame) {
    const std::optional<float> sample = ring.collect(frame);
    if (frame < kWarmupFrames + kDepth) {
      EXPECT_FALSE(sample.has_value())
          << "frame " << frame << ": collected a pre-reset (warmup) span";
    } else {
      EXPECT_TRUE(sample.has_value()) << "frame " << frame << ": measured span missing";
    }
    collected += sample.has_value() ? 1 : 0;
    ring.record_start(frame, stream_);
    ring.record_end(frame, stream_);
  }

  // Drain the last ring lap: each of the depth pending slots is visited exactly once.
  for (uint64_t i = 0; i < kDepth; ++i) {
    const std::optional<float> sample = ring.collect(kFrames + i);
    EXPECT_TRUE(sample.has_value()) << "drain " << i << ": pending measured span missing";
    collected += sample.has_value() ? 1 : 0;
  }
  EXPECT_EQ(collected, kFrames - kWarmupFrames)
      << "every measured frame must be sampled exactly once";

  // An incomplete pair (start without end) stays excluded through the drain.
  ring.record_start(kFrames, stream_);
  EXPECT_FALSE(ring.collect(kFrames + kDepth).has_value());
}

// A span straddling the boundary — start(last warmup frame), reset(), then the delayed
// record_end on the first measured iteration (exactly how the benchmarks close their
// Vulkan/frame spans) — must stay excluded: pre-fix, the retained start_frame matched, the
// late end re-completed the warmup pair, and it was collected a lap later in place of a
// measured sample. Post-reset frames must still each be collected exactly once.
TEST_F(CudaEventRingTest, LateEndAcrossResetStaysExcluded) {
  constexpr uint32_t kDepth = 3;
  constexpr uint64_t kLastWarmup = 4;
  CudaEventSpanRing ring(kDepth);

  for (uint64_t frame = 0; frame <= kLastWarmup; ++frame) {
    ring.record_start(frame, stream_);
    if (frame < kLastWarmup) {
      ring.record_end(frame, stream_);  // earlier warmup spans close normally
    }
  }
  ring.reset();
  // The delayed close arrives after the boundary; it must not resurrect the warmup pair
  // (and must not throw — it is legitimate benchmark behavior, not ring misuse).
  ring.record_end(kLastWarmup, stream_);

  uint64_t collected = 0;
  constexpr uint64_t kFrames = kLastWarmup + 1 + 2 * kDepth;
  for (uint64_t frame = kLastWarmup + 1; frame < kFrames; ++frame) {
    const std::optional<float> sample = ring.collect(frame);
    if (frame < kLastWarmup + 1 + kDepth) {
      EXPECT_FALSE(sample.has_value())
          << "frame " << frame << ": a pre-reset span leaked past the boundary";
    } else {
      EXPECT_TRUE(sample.has_value()) << "frame " << frame << ": measured span missing";
    }
    collected += sample.has_value() ? 1 : 0;
    ring.record_start(frame, stream_);
    ring.record_end(frame, stream_);
  }
  for (uint64_t i = 0; i < kDepth; ++i) {
    const std::optional<float> sample = ring.collect(kFrames + i);
    EXPECT_TRUE(sample.has_value()) << "drain " << i << ": pending measured span missing";
    collected += sample.has_value() ? 1 : 0;
  }
  EXPECT_EQ(collected, kFrames - (kLastWarmup + 1))
      << "every post-reset frame must be sampled exactly once";
}

// GPU-free: the injectable create/destroy seam proves a partially failed construction
// destroys every event it already created before rethrowing (the pre-fix constructor threw
// with the created handles stranded in a never-destructed object).
namespace fake_events {
uint32_t create_calls = 0;
uint32_t fail_at = 0;  // 1-based create call to fail; 0 = never
uint32_t destroy_calls = 0;

CUresult counting_create(CUevent* event, unsigned int /*flags*/) {
  ++create_calls;
  if (fail_at != 0 && create_calls == fail_at) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  *event = reinterpret_cast<CUevent>(static_cast<uintptr_t>(create_calls));
  return CUDA_SUCCESS;
}

CUresult counting_destroy(CUevent /*event*/) {
  ++destroy_calls;
  return CUDA_SUCCESS;
}
}  // namespace fake_events

TEST(CudaEventRingCtor, PartialCreateFailureDestroysCreatedEvents) {
  constexpr uint32_t kDepth = 3;  // 6 creates per full construction
  // Fail at the first, a middle, and the last create call.
  for (const uint32_t fail_at : {1u, 4u, 6u}) {
    fake_events::create_calls = 0;
    fake_events::destroy_calls = 0;
    fake_events::fail_at = fail_at;
    EXPECT_THROW(
        CudaEventSpanRing(kDepth, &fake_events::counting_create, &fake_events::counting_destroy),
        std::runtime_error);
    EXPECT_EQ(fake_events::create_calls, fail_at);
    EXPECT_EQ(fake_events::destroy_calls, fail_at - 1)
        << "every event created before the failure must be destroyed";
  }

  // Success path: construction and destruction balance.
  fake_events::create_calls = 0;
  fake_events::destroy_calls = 0;
  fake_events::fail_at = 0;
  {
    const CudaEventSpanRing ring(
        kDepth, &fake_events::counting_create, &fake_events::counting_destroy);
    EXPECT_EQ(fake_events::create_calls, 2 * kDepth);
  }
  EXPECT_EQ(fake_events::destroy_calls, 2 * kDepth);
}

}  // namespace
