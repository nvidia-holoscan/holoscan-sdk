/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include <holoscan/core/lifecycle.hpp>
#include <holoscan/sensor_io/lifecycle_guards.hpp>

namespace holoscan::sensor_io {
namespace {

/// Records whether each stage body actually ran, so a test can distinguish "the body ran and
/// returned kSkipped" from "the guard withheld the body".
class BodyRecorder {
 public:
  [[nodiscard]] auto body(LifecycleStage stage, LifecycleStatus status) {
    return [this, stage, status]() noexcept {
      invoked_.push_back(stage);
      return status;
    };
  }

  [[nodiscard]] bool ran(LifecycleStage stage) const {
    for (const LifecycleStage seen : invoked_) {
      if (seen == stage) {
        return true;
      }
    }
    return false;
  }

  [[nodiscard]] std::size_t count() const noexcept { return invoked_.size(); }

 private:
  std::vector<LifecycleStage> invoked_;
};

constexpr LifecycleStatus kOk = LifecycleStatus::kOk;
constexpr LifecycleStatus kSkipped = LifecycleStatus::kSkipped;
constexpr LifecycleStatus kFatal = LifecycleStatus::kFatalFailure;

// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, RunsEveryStageOnTheHappyPath) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kDiscover>(recorder.body(LifecycleStage::kDiscover, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kWarmUp>(recorder.body(LifecycleStage::kWarmUp, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kArm>(recorder.body(LifecycleStage::kArm, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStart>(recorder.body(LifecycleStage::kStart, kOk)), kOk);

  EXPECT_EQ(recorder.count(), 6U);
  EXPECT_TRUE(guards.allocated());
  EXPECT_TRUE(guards.armed());
  EXPECT_TRUE(guards.started());
  EXPECT_FALSE(guards.any_guard_skipped());
}

// The lifecycle guide's recommended shape: register a stage only where the operator genuinely has
// work, commonly just a matched kAllocate and kRelease pair. Core records a successful skipped
// checkpoint for every barrier the operator left empty, so the guard is never asked about them.
// Gating a forward body on a record an unregistered earlier stage would have written withholds
// every body such an operator has, and Core reads the result as a clean run in which nothing ran.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, SparseRegistrationRunsBodiesWithNoEarlierStage) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_TRUE(recorder.ran(LifecycleStage::kAllocate));
  EXPECT_TRUE(guards.allocated());

  EXPECT_EQ(guards.run<LifecycleStage::kRelease>(recorder.body(LifecycleStage::kRelease, kOk)),
            kOk);
  EXPECT_TRUE(recorder.ran(LifecycleStage::kRelease));
  EXPECT_FALSE(guards.allocated());
  EXPECT_FALSE(guards.any_guard_skipped());
}

// The same property for an operator whose only work is at kArm and kStart, with no kConfigure and
// no kAllocate to establish anything on its behalf.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, SparseArmAndStartRunWithoutAnAllocateStage) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kArm>(recorder.body(LifecycleStage::kArm, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStart>(recorder.body(LifecycleStage::kStart, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStop>(recorder.body(LifecycleStage::kStop, kOk)), kOk);

  EXPECT_TRUE(recorder.ran(LifecycleStage::kArm));
  EXPECT_TRUE(recorder.ran(LifecycleStage::kStart));
  EXPECT_TRUE(recorder.ran(LifecycleStage::kStop));
  EXPECT_FALSE(guards.any_guard_skipped());

  // Nothing was allocated, so the release Core still schedules must not reach its body.
  EXPECT_EQ(guards.run<LifecycleStage::kRelease>(recorder.body(LifecycleStage::kRelease, kOk)),
            kSkipped);
  EXPECT_FALSE(recorder.ran(LifecycleStage::kRelease));
}

// A failed allocate took nothing, so the rollback release Core schedules must not reach its body.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, FailedAllocateOwnsNothingToRelease) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kFatal)),
            kFatal);
  EXPECT_FALSE(guards.allocated());

  EXPECT_EQ(guards.run<LifecycleStage::kRelease>(recorder.body(LifecycleStage::kRelease, kOk)),
            kSkipped);
  EXPECT_FALSE(recorder.ran(LifecycleStage::kRelease));
  EXPECT_TRUE(guards.guard_skipped(LifecycleStage::kRelease));
}

// The case this type exists for. An allocate body with nothing to reserve reports kSkipped, which
// Core records as a completed kAllocate because callback_succeeded admits kOk and kSkipped alike.
// run_planned_teardown then gates kRelease on that recorded completion, so Core schedules a release
// for an operator that reserved nothing, and LifecycleContext gives the operator no way to discover
// which of the two its own allocate body reported. Suppressing the body here is what keeps that
// release from handing back buffers that were never taken. Arming must still be permitted: having
// nothing to allocate is a success, not a failure.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, SkippedAllocatePermitsArmButOwnsNothing) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(
      guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kSkipped)),
      kSkipped);

  EXPECT_FALSE(guards.allocated());

  EXPECT_EQ(guards.run<LifecycleStage::kArm>(recorder.body(LifecycleStage::kArm, kOk)), kOk);
  EXPECT_TRUE(recorder.ran(LifecycleStage::kArm));

  EXPECT_EQ(guards.run<LifecycleStage::kRelease>(recorder.body(LifecycleStage::kRelease, kOk)),
            kSkipped);
  EXPECT_FALSE(recorder.ran(LifecycleStage::kRelease));
}

// A successful release gives the resources back, so a second release must not run again. This is
// the double-release the ownership record exists to prevent.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, ReleaseClosesOwnershipSoASecondReleaseIsWithheld) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kRelease>(recorder.body(LifecycleStage::kRelease, kOk)),
            kOk);

  EXPECT_TRUE(recorder.ran(LifecycleStage::kRelease));
  EXPECT_FALSE(guards.allocated());

  BodyRecorder after_release;
  EXPECT_EQ(guards.run<LifecycleStage::kRelease>(after_release.body(LifecycleStage::kRelease, kOk)),
            kSkipped);
  EXPECT_EQ(after_release.count(), 0U);
}

// Core keeps an advanced kStop eligible after a completed kArm, so a sensor that armed but never
// started still owes its device a quiesce.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, StopRunsAfterArmEvenWhenStartNeverRan) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kArm>(recorder.body(LifecycleStage::kArm, kOk)), kOk);

  EXPECT_EQ(guards.run<LifecycleStage::kStop>(recorder.body(LifecycleStage::kStop, kOk)), kOk);
  EXPECT_TRUE(recorder.ran(LifecycleStage::kStop));
  // The arm flag survives the stop; see StartOnlyRestartRunsTheStartBodyBecauseArmIsRetained.
  EXPECT_TRUE(guards.armed());
}

// The forward half of the arm conflation. An arm body with nothing to queue reports kSkipped, which
// Core records as a completed kArm exactly as it records kOk, and then publishes steady state.
// Withholding the start body here on the grounds that nothing was armed would skip the stream while
// the run reported success: the device never produces a frame and nothing reports a failure.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, SkippedArmStillPermitsTheStartBody) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kArm>(recorder.body(LifecycleStage::kArm, kSkipped)),
            kSkipped);

  // The arm owned nothing, and the start body still has to run.
  EXPECT_FALSE(guards.armed());

  EXPECT_EQ(guards.run<LifecycleStage::kStart>(recorder.body(LifecycleStage::kStart, kOk)), kOk);
  EXPECT_TRUE(recorder.ran(LifecycleStage::kStart));
  EXPECT_TRUE(guards.started());
  EXPECT_FALSE(guards.guard_skipped(LifecycleStage::kStart));
}

// The same conflation on the teardown side. run_planned_teardown gates kStop on a recorded kArm,
// and an kArm body that reported kSkipped is recorded just as a kOk one is, so Core schedules a
// stop for a device that was never armed. Nothing was turned on, so nothing may be turned off.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, SkippedArmOwnsNothingToStop) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kArm>(recorder.body(LifecycleStage::kArm, kSkipped)),
            kSkipped);
  EXPECT_TRUE(recorder.ran(LifecycleStage::kArm));
  EXPECT_FALSE(guards.armed());

  EXPECT_EQ(guards.run<LifecycleStage::kStop>(recorder.body(LifecycleStage::kStop, kOk)), kSkipped);
  EXPECT_FALSE(recorder.ran(LifecycleStage::kStop));
  EXPECT_TRUE(guards.guard_skipped(LifecycleStage::kStop));

  // Allocation still succeeded, so the buffers are real and release must reach its body.
  EXPECT_EQ(guards.run<LifecycleStage::kRelease>(recorder.body(LifecycleStage::kRelease, kOk)),
            kOk);
  EXPECT_TRUE(recorder.ran(LifecycleStage::kRelease));
}

// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, StopIsWithheldWhenNothingWasArmed) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStop>(recorder.body(LifecycleStage::kStop, kOk)), kSkipped);
  EXPECT_FALSE(recorder.ran(LifecycleStage::kStop));
}

// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, ConfigureResetsProgressSoRestartBeginsClean) {
  StageGuards guards;
  BodyRecorder first;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(first.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(first.body(LifecycleStage::kAllocate, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kArm>(first.body(LifecycleStage::kArm, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStart>(first.body(LifecycleStage::kStart, kOk)), kOk);
  ASSERT_TRUE(guards.started());

  BodyRecorder restart;
  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(restart.body(LifecycleStage::kConfigure, kOk)),
            kOk);

  EXPECT_FALSE(guards.allocated());
  EXPECT_FALSE(guards.armed());
  EXPECT_FALSE(guards.started());
  EXPECT_FALSE(guards.any_guard_skipped());

  // The reset disclaimed the previous epoch's resources, so teardown for them must be withheld
  // until this epoch's forward bodies take something of their own.
  EXPECT_EQ(guards.run<LifecycleStage::kStop>(restart.body(LifecycleStage::kStop, kOk)), kSkipped);
  EXPECT_EQ(guards.run<LifecycleStage::kRelease>(restart.body(LifecycleStage::kRelease, kOk)),
            kSkipped);
  EXPECT_EQ(restart.count(), 1U);
}

// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, FailedStopRetainsProgressSoTeardownCanBeRetried) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kArm>(recorder.body(LifecycleStage::kArm, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStart>(recorder.body(LifecycleStage::kStart, kOk)), kOk);

  EXPECT_EQ(guards.run<LifecycleStage::kStop>(recorder.body(LifecycleStage::kStop, kFatal)),
            kFatal);
  EXPECT_TRUE(guards.armed());
  EXPECT_TRUE(guards.started());
}

// Regression: a failed kStop leaves the arm and start records set, and kRelease used to clear only
// the allocation ones. That left a freed device still describing itself as armed, so a later kStop
// reached its body and quiesced buffers that had already been handed back.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, ReleaseAfterFailedStopClearsProgressSoStopCannotRunOnFreedBuffers) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kArm>(recorder.body(LifecycleStage::kArm, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStart>(recorder.body(LifecycleStage::kStart, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStop>(recorder.body(LifecycleStage::kStop, kFatal)),
            kFatal);

  // Release must still run, or the buffers a failed stop was quiescing are stranded for good.
  EXPECT_EQ(guards.run<LifecycleStage::kRelease>(recorder.body(LifecycleStage::kRelease, kOk)),
            kOk);
  EXPECT_TRUE(recorder.ran(LifecycleStage::kRelease));

  EXPECT_FALSE(guards.allocated());
  EXPECT_FALSE(guards.armed());
  EXPECT_FALSE(guards.started());

  BodyRecorder after_release;
  EXPECT_EQ(guards.run<LifecycleStage::kStop>(after_release.body(LifecycleStage::kStop, kOk)),
            kSkipped);
  EXPECT_FALSE(after_release.ran(LifecycleStage::kStop));
}

// A release body runs before the flags are cleared, so it can tell whether it is tearing down a
// device that stopped cleanly or one that never did.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, ReleaseBodyObservesPreReleaseProgress) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kArm>(recorder.body(LifecycleStage::kArm, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStart>(recorder.body(LifecycleStage::kStart, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStop>(recorder.body(LifecycleStage::kStop, kFatal)),
            kFatal);

  bool saw_started = false;
  EXPECT_EQ(guards.run<LifecycleStage::kRelease>([&guards, &saw_started]() noexcept {
    saw_started = guards.started();
    return kOk;
  }),
            kOk);
  EXPECT_TRUE(saw_started);
  EXPECT_FALSE(guards.started());
}

// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, FailedReleaseRetainsOwnership) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kRelease>(recorder.body(LifecycleStage::kRelease, kFatal)),
            kFatal);
  EXPECT_TRUE(guards.allocated());
}

// Regression: Core runs the suffix kStop, kStart for a restart requested at kStart and retains the
// completed kArm checkpoint rather than revisiting it
// (public/tests/core/test_lifecycle_executor.cpp, RestartSuffixesMatchExactRequestedStageTable).
// Clearing armed_ at kStop made this kStart skip, and Core reads kSkipped as success, so the
// restart was published with a sensor that never started.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, StartOnlyRestartRunsTheStartBodyBecauseArmIsRetained) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kArm>(recorder.body(LifecycleStage::kArm, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStart>(recorder.body(LifecycleStage::kStart, kOk)), kOk);

  // The restart suffix, with no kArm between the two.
  BodyRecorder restart;
  EXPECT_EQ(guards.run<LifecycleStage::kStop>(restart.body(LifecycleStage::kStop, kOk)), kOk);
  EXPECT_FALSE(guards.started());
  EXPECT_TRUE(guards.armed());

  EXPECT_EQ(guards.run<LifecycleStage::kStart>(restart.body(LifecycleStage::kStart, kOk)), kOk);
  EXPECT_TRUE(restart.ran(LifecycleStage::kStart));
  EXPECT_TRUE(guards.started());
  EXPECT_FALSE(guards.guard_skipped(LifecycleStage::kStart));
}

// The shutdown teardown that follows a restart's stop runs kStop again against a retained arm flag.
// It must reach the body, because the intervening kStart may have resumed the device.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, StopRunsAgainAfterAStartOnlyRestart) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kArm>(recorder.body(LifecycleStage::kArm, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStart>(recorder.body(LifecycleStage::kStart, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStop>(recorder.body(LifecycleStage::kStop, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStart>(recorder.body(LifecycleStage::kStart, kOk)), kOk);

  BodyRecorder shutdown;
  EXPECT_EQ(guards.run<LifecycleStage::kStop>(shutdown.body(LifecycleStage::kStop, kOk)), kOk);
  EXPECT_TRUE(shutdown.ran(LifecycleStage::kStop));
}

// A failed kArm armed nothing, so the rollback kStop Core schedules against the operator's recorded
// kArm checkpoint must not quiesce a device that was never armed. The kRelease in the same rollback
// still runs, because the kAllocate before it did take real buffers.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, FailedArmOwnsNothingToStopButAllocationIsStillReleased) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kArm>(recorder.body(LifecycleStage::kArm, kFatal)), kFatal);
  EXPECT_FALSE(guards.armed());

  EXPECT_EQ(guards.run<LifecycleStage::kStop>(recorder.body(LifecycleStage::kStop, kOk)), kSkipped);
  EXPECT_FALSE(recorder.ran(LifecycleStage::kStop));

  // Release must still run: allocate succeeded, so the buffers are real.
  EXPECT_EQ(guards.run<LifecycleStage::kRelease>(recorder.body(LifecycleStage::kRelease, kOk)),
            kOk);
  EXPECT_TRUE(recorder.ran(LifecycleStage::kRelease));
}

// LIFECYCLE-006's restart entry at kArm: the suffix is kStop then kArm, kStart, with no kConfigure
// and no kRelease. Nothing in that suffix hands the device back, so a re-arm reporting kSkipped has
// relinquished nothing and the previous epoch's acquisition is still outstanding. It has to stay
// recorded, while the kStart that follows still reaches its body.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, ArmRestartRetainsOwnershipAcrossASkippedRearm) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kArm>(recorder.body(LifecycleStage::kArm, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStart>(recorder.body(LifecycleStage::kStart, kOk)), kOk);

  BodyRecorder restart;
  EXPECT_EQ(guards.run<LifecycleStage::kStop>(restart.body(LifecycleStage::kStop, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kArm>(restart.body(LifecycleStage::kArm, kSkipped)),
            kSkipped);
  EXPECT_TRUE(restart.ran(LifecycleStage::kArm));
  EXPECT_TRUE(guards.armed());

  EXPECT_EQ(guards.run<LifecycleStage::kStart>(restart.body(LifecycleStage::kStart, kOk)), kOk);
  EXPECT_TRUE(restart.ran(LifecycleStage::kStart));
  EXPECT_FALSE(guards.guard_skipped(LifecycleStage::kStart));
}

// The defect the retention above exists to prevent. A restart whose re-entered kArm and kStart
// bodies both report kSkipped used to clear both records, and the shutdown kStop then found nothing
// to quiesce and withheld its body -- even though the first epoch's kArm acquired a device and no
// kRelease or kConfigure ever ran. Core reads that withheld stop as a success, so the operator
// completes teardown still holding the device.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, SkippedRearmAndSkippedRestartedStartStillReachTheShutdownStop) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kArm>(recorder.body(LifecycleStage::kArm, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStart>(recorder.body(LifecycleStage::kStart, kOk)), kOk);

  // The restart suffix: stop, then a re-arm and a restarted start that both have no work.
  BodyRecorder restart;
  EXPECT_EQ(guards.run<LifecycleStage::kStop>(restart.body(LifecycleStage::kStop, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kArm>(restart.body(LifecycleStage::kArm, kSkipped)),
            kSkipped);
  EXPECT_EQ(guards.run<LifecycleStage::kStart>(restart.body(LifecycleStage::kStart, kSkipped)),
            kSkipped);

  // Neither skip relinquished anything, so the device armed in the first epoch is still owed a
  // quiesce even though no body in this suffix took anything of its own.
  EXPECT_TRUE(guards.armed());
  EXPECT_FALSE(guards.started());

  BodyRecorder shutdown;
  EXPECT_EQ(guards.run<LifecycleStage::kStop>(shutdown.body(LifecycleStage::kStop, kOk)), kOk);
  EXPECT_TRUE(shutdown.ran(LifecycleStage::kStop));
  EXPECT_FALSE(guards.guard_skipped(LifecycleStage::kStop));

  EXPECT_EQ(guards.run<LifecycleStage::kRelease>(shutdown.body(LifecycleStage::kRelease, kOk)),
            kOk);
  EXPECT_TRUE(shutdown.ran(LifecycleStage::kRelease));
}

// The same retention on the allocation record, which the kArm restart suffix never releases either.
// A failed kRelease keeps its record so teardown can be retried, and a re-entered kAllocate that
// reports kSkipped must not be what discards it: the buffers the first epoch reserved were never
// handed back, so kRelease has to stay reachable.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, SkippedReallocateRetainsTheAllocationOwedARelease) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kRelease>(recorder.body(LifecycleStage::kRelease, kFatal)),
            kFatal);
  ASSERT_TRUE(guards.allocated());

  BodyRecorder retry;
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(retry.body(LifecycleStage::kAllocate, kSkipped)),
            kSkipped);
  EXPECT_TRUE(guards.allocated());

  EXPECT_EQ(guards.run<LifecycleStage::kRelease>(retry.body(LifecycleStage::kRelease, kOk)), kOk);
  EXPECT_TRUE(retry.ran(LifecycleStage::kRelease));
  EXPECT_FALSE(guards.allocated());
}

// The same rule on the start record, which completes it across every forward body that holds one.
// Core runs the kStop that does relinquish this before it re-enters kStart, so the record is
// already clear by then and no sequence the runtime produces tells the two readings apart. Pinned
// anyway, because the rule is what SENSOR-IO-001 states rather than the ordering that makes it
// unobservable: a guard whose correctness rests on a suffix always containing a particular stage
// acquires a silent dependency on that suffix never changing.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, SkippedRestartedStartRetainsAnEarlierStartRecord) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kArm>(recorder.body(LifecycleStage::kArm, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStart>(recorder.body(LifecycleStage::kStart, kOk)), kOk);
  ASSERT_TRUE(guards.started());

  // A second start body with no work of its own. It handed nothing back, so what the first one
  // published is still published.
  BodyRecorder restart;
  EXPECT_EQ(guards.run<LifecycleStage::kStart>(restart.body(LifecycleStage::kStart, kSkipped)),
            kSkipped);
  EXPECT_TRUE(restart.ran(LifecycleStage::kStart));
  EXPECT_TRUE(guards.started());

  // The stage that does relinquish it still does, and clears only that record.
  EXPECT_EQ(guards.run<LifecycleStage::kStop>(restart.body(LifecycleStage::kStop, kOk)), kOk);
  EXPECT_FALSE(guards.started());
  EXPECT_TRUE(guards.armed());
}

// The public diagnostic query takes a LifecycleStage and documents no valid-enum precondition, so a
// value outside the closed vocabulary has to answer rather than shift by itself. 255 exceeds the
// width of the promoted shift operand, which would be undefined behavior in a noexcept query.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, GuardSkippedRejectsAStageOutsideTheVocabulary) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  // A real withheld stage, so the query is answering from a non-empty record.
  EXPECT_EQ(guards.run<LifecycleStage::kStop>(recorder.body(LifecycleStage::kStop, kOk)), kSkipped);
  ASSERT_TRUE(guards.guard_skipped(LifecycleStage::kStop));

  // Unknown stage values must not alias any retained lifecycle guard bit.
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_FALSE(guards.guard_skipped(static_cast<LifecycleStage>(255)));
  // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
  EXPECT_FALSE(guards.guard_skipped(static_cast<LifecycleStage>(8)));

  // The widest stage the vocabulary does name still reports through the record.
  EXPECT_FALSE(guards.guard_skipped(LifecycleStage::kRelease));
  EXPECT_EQ(guards.run<LifecycleStage::kRelease>(recorder.body(LifecycleStage::kRelease, kOk)),
            kSkipped);
  EXPECT_TRUE(guards.guard_skipped(LifecycleStage::kRelease));
}

// LIFECYCLE-006's restart entry at kAllocate: the suffix is kStop, kRelease then kAllocate through
// kStart, again with no kConfigure. The kRelease disclaims the whole ownership record, so the
// forward suffix has to rebuild it from the re-entered allocate rather than inheriting any of it.
// REQ: SENSOR-IO-001
TEST(SensorIoStageGuards, AllocateRestartRebuildsProgressWithoutAConfigure) {
  StageGuards guards;
  BodyRecorder recorder;

  EXPECT_EQ(guards.run<LifecycleStage::kConfigure>(recorder.body(LifecycleStage::kConfigure, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(recorder.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kArm>(recorder.body(LifecycleStage::kArm, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kStart>(recorder.body(LifecycleStage::kStart, kOk)), kOk);

  BodyRecorder restart;
  EXPECT_EQ(guards.run<LifecycleStage::kStop>(restart.body(LifecycleStage::kStop, kOk)), kOk);
  EXPECT_EQ(guards.run<LifecycleStage::kRelease>(restart.body(LifecycleStage::kRelease, kOk)), kOk);
  EXPECT_FALSE(guards.armed());
  EXPECT_FALSE(guards.allocated());
  EXPECT_FALSE(guards.started());

  EXPECT_EQ(guards.run<LifecycleStage::kAllocate>(restart.body(LifecycleStage::kAllocate, kOk)),
            kOk);
  EXPECT_TRUE(restart.ran(LifecycleStage::kAllocate));
  EXPECT_EQ(guards.run<LifecycleStage::kWarmUp>(restart.body(LifecycleStage::kWarmUp, kOk)), kOk);
  EXPECT_TRUE(restart.ran(LifecycleStage::kWarmUp));
  EXPECT_EQ(guards.run<LifecycleStage::kArm>(restart.body(LifecycleStage::kArm, kOk)), kOk);
  EXPECT_TRUE(restart.ran(LifecycleStage::kArm));
  EXPECT_EQ(guards.run<LifecycleStage::kStart>(restart.body(LifecycleStage::kStart, kOk)), kOk);
  EXPECT_TRUE(restart.ran(LifecycleStage::kStart));
  EXPECT_TRUE(guards.started());
}

}  // namespace
}  // namespace holoscan::sensor_io
