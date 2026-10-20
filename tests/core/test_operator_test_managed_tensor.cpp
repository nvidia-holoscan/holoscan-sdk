/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "../testing/managed_tensor_source.hpp"

namespace {
using namespace std::chrono_literals;
using namespace holoscan;
using namespace holoscan::testing;
using namespace holoscan::testing::test;

constexpr std::size_t kTightRetainRecords{6U};

struct TightRetainProbe : ManagedTensorProbe {
  std::array<void*, kTightRetainRecords> emitted{};
  std::array<void*, kTightRetainRecords - 1U> unpublished{};
  std::size_t held_loans{};
  std::size_t rejected_writers{};

  bool wait_for_held_loan(std::size_t count) {
    std::unique_lock lock(mutex);
    return changed.wait_for(lock, 5s, [&] { return held_loans >= count; });
  }
};

// Keep the qualified zero-input finite source profile. Each restart carries a real
// unpublished writer; the next compute retires it before requesting replacement capacity.
class TightRetainSource final : public Operator<> {
 public:
  explicit TightRetainSource(std::shared_ptr<TightRetainProbe> probe) : probe_(std::move(probe)) {}

  void setup(OperatorSpec& spec) override {
    spec.output(out_, "out")
        .max_emits_per_compute(1U)
        .produces_tensor({.representation = {.memory_kind = MemoryKind::kHost,
                                             .dtype = DLDataType{kDLUInt, 8, 1},
                                             .rank = 1U},
                          .bounds = tensor_bounds(4U),
                          .storage = TensorOutputStorage::kRuntimePool});
    spec.notification_source(event_, "step").capacity(1U).sender_reference_capacity(2U);
    spec.lifecycle()
        .stage(LifecycleStage::kAllocate, &TightRetainSource::allocate)
        .stage(LifecycleStage::kArm, &TightRetainSource::arm);
  }

  Contract contract() const override {
    Contract c;
    c.trigger(OnNotified{.event = event_})
        .complete_after(ComputeCalls{.limit = kTightRetainRecords});
    return c;
  }

  LifecycleStatus allocate(LifecycleContext& context) noexcept {
    if (context.transition_phase() == LifecycleTransitionPhase::kRestartForward) {
      const std::lock_guard lock(probe_->mutex);
      ++probe_->restart_allocations;
    }
    return LifecycleStatus::kOk;
  }

  LifecycleStatus arm(LifecycleContext& context) noexcept {
    auto sender = context.notification_sender(event_);
    if (!sender) {
      context.report_error(sender.error());
      return LifecycleStatus::kFatalFailure;
    }
    const std::lock_guard lock(probe_->mutex);
    probe_->sender = std::move(*sender);
    ++probe_->arms;
    probe_->changed.notify_all();
    return LifecycleStatus::kOk;
  }

  expected<void, Error> compute(ExecutionContext&) override {
    if (stale_guard_) {
      auto committed = std::move(*stale_guard_).commit();
      const bool rejected = !committed && (committed.error().code == ErrorCode::kClosed ||
                                           committed.error().code == ErrorCode::kInvalidArgument);
      // The guard borrows the loan. Retire both before another allocation can reuse this slot.
      stale_guard_.reset();
      stale_loan_.reset();
      if (!rejected)
        return make_unexpected(Error{ErrorCode::kFailure, "stale Tensor writer was not revoked"});
      const std::lock_guard lock(probe_->mutex);
      ++probe_->rejected_writers;
    }

    const std::array<std::int64_t, 1> shape{4};
    auto loan = out_.allocate_tensor(shape);
    if (!loan)
      return make_unexpected(loan.error());
    void* emitted_data{};
    auto filled = loan->write_host_and_commit([&](const auto& write) {
      emitted_data = write.data();
      std::memset(write.data(), static_cast<int>(11U * (next_ + 1U)), write.byte_size());
    });
    if (!filled)
      return filled;
    auto emitted = out_.emit(std::move(*loan));
    if (!emitted)
      return emitted;
    {
      const std::lock_guard lock(probe_->mutex);
      probe_->emitted[next_] = emitted_data;
    }

    if (next_ + 1U < kTightRetainRecords) {
      auto extra = out_.allocate_tensor(shape);
      if (!extra)
        return make_unexpected(extra.error());
      stale_loan_.emplace(std::move(*extra));
      auto writer = stale_loan_->write_host();
      if (!writer)
        return make_unexpected(writer.error());
      stale_guard_.emplace(std::move(*writer));
      std::memset(stale_guard_->data(), 0xee, stale_guard_->byte_size());
      const std::lock_guard lock(probe_->mutex);
      probe_->unpublished[next_] = stale_guard_->data();
      ++probe_->held_loans;
      probe_->changed.notify_all();
    }
    ++next_;
    return {};
  }

 private:
  std::shared_ptr<TightRetainProbe> probe_;
  Output<Tensor> out_;
  NotificationSource event_;
  std::size_t next_{};
  std::optional<TensorOutputLoan> stale_loan_;
  std::optional<TensorOutputWriteGuard> stale_guard_;
};
}  // namespace

// REQ: OPERATOR-073
// REQ: OPERATOR-070
TEST(OperatorTestingManagedTensor, RetainPreservesShallowAndDeepBackingAndRejectsOldWriters) {
  for (const auto stage : {LifecycleStage::kArm, LifecycleStage::kAllocate}) {
    SCOPED_TRACE(static_cast<int>(stage));
    auto probe = std::make_shared<ManagedTensorProbe>();
    auto h = create_operator_test<ManagedTensorSource<Tensor>>(probe, Tensor{}, true);
    h.capture<Tensor>("out", Retain{.max_records = 8U});
    h.configure_graph([](auto& view) {
      view.graph().lifecycle().allow_external_restart(view.tested_operator());
      view.compile_options().lifecycle_transactions = tensor_restart_capacity();
    });
    auto run = h.run_async();
    auto original_sender = probe->wait_sender(1U);
    ASSERT_TRUE(post_tensor(original_sender));
    ASSERT_TRUE(run.wait_for_output("out", 1U, 5s));
    const auto target = run.operator_ref();
    ASSERT_TRUE(target);
    const std::array targets{*target};
    const std::array stages{stage};
    auto control =
        run.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
    ASSERT_TRUE(control);
    const auto request = control->request_restart(*target, {.from_stage = stage});
    ASSERT_TRUE(request);
    const auto terminal = control->await(*request, 5s);
    ASSERT_TRUE(terminal);
    ASSERT_EQ(terminal->state, LifecycleTransactionState::kPublished);
    EXPECT_EQ(terminal->effective_stage, stage);
    EXPECT_EQ(original_sender.post().error().code, ErrorCode::kStaleEndpoint);
    ASSERT_TRUE(post_tensor(probe->wait_sender(2U)));
    ASSERT_TRUE(run.wait_for_output("out", 2U, 5s));
    const auto result = run.finish();
    ASSERT_TRUE(result.ok()) << result.describe_failure();
    EXPECT_EQ(result.termination(), RunTermination::kStopped);
    const auto values = result.values<Tensor>("out");
    ASSERT_EQ(values.size(), 2U);
    EXPECT_EQ(values[0].data_as<std::uint8_t>()[0], 11U);
    EXPECT_EQ(values[1].data_as<std::uint8_t>()[0], 22U);
    EXPECT_NE(values[0].data(), values[1].data());
    const std::lock_guard lock(probe->mutex);
    EXPECT_EQ(probe->restart_allocations, stage == LifecycleStage::kAllocate ? 1U : 0U);
    EXPECT_TRUE(probe->stale_guard_rejected);
  }
}

// REQ: OPERATOR-073
// REQ: OPERATOR-070
TEST(OperatorTestingManagedTensor, ShallowRestartReclaimsUnpublishedLoansAtExactRetainCapacity) {
  auto probe = std::make_shared<TightRetainProbe>();
  auto h = create_operator_test<TightRetainSource>(probe);
  h.capture<Tensor>("out", Retain{.max_records = kTightRetainRecords});
  h.output_flow("out", ConnectionOptions{.queue_depth = kTightRetainRecords});
  h.configure_graph([](auto& view) {
    view.graph().lifecycle().allow_external_restart(view.tested_operator());
    view.compile_options().lifecycle_transactions = tensor_restart_capacity();
  });
  auto run = h.run_async({.timeout = 60s});
  for (std::size_t index = 0; index < kTightRetainRecords; ++index) {
    SCOPED_TRACE(index);
    auto sender = probe->wait_sender(index + 1U);
    ASSERT_TRUE(post_tensor(sender));
    ASSERT_TRUE(run.wait_for_output("out", index + 1U, 5s));
    EXPECT_EQ(run.output_count("out"), index + 1U);
    if (index + 1U == kTightRetainRecords)
      break;
    // Output observation alone is not a loan or quiescence witness. The operator
    // separately confirms that its unpublished guard is still live across this request.
    ASSERT_TRUE(probe->wait_for_held_loan(index + 1U));
    const auto target = run.operator_ref();
    ASSERT_TRUE(target);
    const std::array targets{*target};
    constexpr std::array stages{LifecycleStage::kArm};
    auto control =
        run.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
    ASSERT_TRUE(control);
    const auto request = control->request_restart(*target, {.from_stage = LifecycleStage::kArm});
    ASSERT_TRUE(request);
    const auto terminal = control->await(*request, 5s);
    ASSERT_TRUE(terminal);
    ASSERT_EQ(terminal->state, LifecycleTransactionState::kPublished);
    ASSERT_EQ(terminal->effective_stage, LifecycleStage::kArm);
    const auto stale_post = sender.post();
    ASSERT_FALSE(stale_post);
    EXPECT_EQ(stale_post.error().code, ErrorCode::kStaleEndpoint);
  }
  const auto result = run.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  EXPECT_EQ(result.termination(), RunTermination::kStopped);

  // Inspect the exact accepted plan, rather than assuming a physical slot count
  // from max_records. Only producer and retirement contributors form the reserve.
  const auto& plan = result.plan();
  const auto allocations = plan.tensor_allocations();
  ASSERT_EQ(allocations.size(), 1U);
  ASSERT_TRUE(plan.tensor_field_allocations().empty());
  const auto& allocation = allocations.front();
  const auto ports = plan.ports();
  const auto output = std::find_if(
      ports.begin(), ports.end(), [&](const auto& row) { return row.id == allocation.output; });
  ASSERT_NE(output, ports.end());
  ASSERT_EQ(output->direction, PortDirection::kOutput);
  ASSERT_EQ(output->local_name, "out");
  ASSERT_EQ(output->max_emits_per_compute, 1U);
  const auto rules = plan.operator_completion_rules();
  ASSERT_EQ(rules.size(), 1U);
  ASSERT_EQ(rules.front().kind, OperatorCompletionRuleKind::kComputeCalls);
  ASSERT_EQ(rules.front().limit, kTightRetainRecords);
  const auto records = rules.front().limit * output->max_emits_per_compute;

  const auto providers = plan.provider_allocation_plans();
  const auto provider = std::find_if(providers.begin(), providers.end(), [&](const auto& row) {
    return row.pool_id == allocation.pool_id;
  });
  ASSERT_NE(provider, providers.end());
  ASSERT_EQ(provider->geometry_mode, AllocationGeometryMode::kCompileConcrete);
  const auto first = provider->first_sizing_dependency.value;
  const auto end = std::uint64_t{first} + provider->sizing_dependency_count;
  const auto contributors = plan.pool_capacity_contributors();
  std::uint64_t reserve{};
  bool producer_seen{}, retirement_seen{};
  for (const auto& dependency : plan.pool_capacity_dependencies()) {
    if (dependency.id.value < first || dependency.id.value >= end)
      continue;
    ASSERT_EQ(dependency.pool_id, allocation.pool_id);
    if (dependency.origin != PoolClaimOrigin::kProducerLoan &&
        dependency.origin != PoolClaimOrigin::kRetirement)
      continue;
    ASSERT_EQ(dependency.source, PoolCapacityDependencySource::kPredecessorContributor);
    ASSERT_EQ(dependency.mapping_multiplicity, 1U);
    const auto contributor =
        std::find_if(contributors.begin(), contributors.end(), [&](const auto& row) {
          return row.pool_id == allocation.pool_id &&
                 row.contributor_id == dependency.predecessor_contributor_id;
        });
    ASSERT_NE(contributor, contributors.end());
    reserve += contributor->maximum_distinct_slots;
    producer_seen |= dependency.origin == PoolClaimOrigin::kProducerLoan;
    retirement_seen |= dependency.origin == PoolClaimOrigin::kRetirement;
  }
  ASSERT_TRUE(producer_seen);
  ASSERT_TRUE(retirement_seen);
  ASSERT_EQ(allocation.slot_count, records + reserve);
  // More cancelled loans than spare slots prevents a leak from hiding in the reserve.
  ASSERT_GT(kTightRetainRecords - 1U, reserve);

  const auto values = result.values<Tensor>("out");
  ASSERT_EQ(values.size(), records);
  const std::lock_guard lock(probe->mutex);
  EXPECT_EQ(probe->restart_allocations, 0U);
  EXPECT_EQ(probe->held_loans, kTightRetainRecords - 1U);
  EXPECT_EQ(probe->rejected_writers, kTightRetainRecords - 1U);
  bool reclaimed_unpublished_slot{};
  for (std::size_t index = 0; index < values.size(); ++index) {
    ASSERT_EQ(values[index].nbytes(), 4);
    EXPECT_EQ(values[index].data(), probe->emitted[index]);
    const auto* bytes = values[index].data_as<std::uint8_t>();
    ASSERT_NE(bytes, nullptr);
    EXPECT_TRUE(
        std::all_of(bytes, bytes + 4, [index](auto byte) { return byte == 11U * (index + 1U); }));
    for (std::size_t earlier = 0; earlier < index; ++earlier) {
      EXPECT_NE(probe->emitted[index], probe->emitted[earlier]);
      reclaimed_unpublished_slot |= probe->emitted[index] == probe->unpublished[earlier];
      if (index < probe->unpublished.size())
        reclaimed_unpublished_slot |= probe->unpublished[index] == probe->unpublished[earlier];
    }
  }
  EXPECT_TRUE(reclaimed_unpublished_slot);
}

// REQ: OPERATOR-070
TEST(OperatorTestingManagedTensor, LifecycleBoundActiveCapacityRejectsBeforeStartup) {
  auto probe = std::make_shared<ManagedTensorProbe>();
  auto h = create_operator_test<ManagedTensorSource<Tensor>>(probe, Tensor{}, false, true);
  h.capture<Tensor>("out", Retain{.max_records = 8U});
  h.output_flow("out", ConnectionOptions{.queue_depth = 8U});
  h.configure_graph([](auto& view) {
    view.graph().lifecycle().allow_external_restart(view.tested_operator());
    view.compile_options().lifecycle_transactions = tensor_restart_capacity();
  });
  try {
    auto run = h.run_async();
    run.request_stop();
    const auto result = run.finish();
    FAIL() << "expected cold lifecycle-bound-capacity rejection: " << result.describe_failure();
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string{error.what()}.find("lifecycle-bound pool capacity"), std::string::npos);
  }
  EXPECT_EQ(probe->arms, 0U);
}

// REQ: OPERATOR-073
// REQ: OPERATOR-067
// REQ: OPERATOR-070
TEST(OperatorTestingManagedTensor, DelayedCUDAReadinessAndSnapshotsSurviveDeepRestart) {
  int devices{};
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    GTEST_SKIP() << "CUDA device unavailable";
  int previous_device{};
  ASSERT_EQ(cudaGetDevice(&previous_device), cudaSuccess);
  ASSERT_EQ(cudaSetDevice(devices > 1 ? 1 : 0), cudaSuccess);
  struct CopyStream {
    cudaStream_t stream{};
    int previous_device{};
    ~CopyStream() {
      if (stream != nullptr)
        cudaStreamDestroy(stream);
      cudaSetDevice(previous_device);
    }
  } copy{.previous_device = previous_device};
  ASSERT_EQ(cudaStreamCreateWithFlags(&copy.stream, cudaStreamNonBlocking), cudaSuccess);
  for (const bool retain : {false, true}) {
    SCOPED_TRACE(retain);
    auto probe = std::make_shared<ManagedTensorProbe>();
    probe->delay = true;
    const auto result = [&] {
      auto h = create_operator_test<ManagedTensorSource<Tensor>>(
          probe, Tensor{}, false, false, 2U, true);
      if (retain)
        h.capture<Tensor>("out", Retain{.max_records = 8U});
      else
        h.capture_snapshot<Tensor>("out");
      h.output_device("out", DeviceId{devices > 1 ? 1 : 0});
      h.configure_graph([](auto& view) {
        view.graph().lifecycle().allow_external_restart(view.tested_operator());
        view.compile_options().lifecycle_transactions = tensor_restart_capacity();
      });
      auto run = h.run_async();
      struct ReleaseGate {
        std::shared_ptr<ManagedTensorProbe> probe;
        ~ReleaseGate() { probe->release_delay(); }
      } const release{probe};
      const auto fail = [&](const char* message) {
        ADD_FAILURE() << message;
        probe->release_delay();
        run.request_stop();
        return run.finish();
      };
      if (!post_tensor(probe->wait_sender(1U)) || !probe->wait_delay())
        return fail("CUDA publication did not reach the real readiness gate");
      EXPECT_FALSE(run.wait_for_output("out", 1U, 0ms));
      EXPECT_EQ(run.output_count("out"), 0U);
      probe->release_delay();
      if (!run.wait_for_output("out", 1U, 5s))
        return fail("CUDA capture did not complete");
      const auto target = run.operator_ref();
      if (!target)
        return fail("no current run target");
      const std::array targets{*target};
      constexpr std::array stages{LifecycleStage::kAllocate};
      auto control =
          run.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
      if (!control)
        return fail("no lifecycle handle");
      const auto request =
          control->request_restart(*target, {.from_stage = LifecycleStage::kAllocate});
      if (!request)
        return fail("restart was not accepted");
      const auto terminal = control->await(*request, 5s);
      if (!terminal || terminal->state != LifecycleTransactionState::kPublished)
        return fail("restart did not publish");
      if (!post_tensor(probe->wait_sender(2U)) || !run.wait_for_output("out", 2U, 5s))
        return fail("successor output did not arrive");
      return run.finish();
    }();
    ASSERT_TRUE(result.ok()) << result.describe_failure();
    EXPECT_FALSE(probe->delay_expired);
    if (retain) {
      const auto values = result.values<Tensor>("out");
      ASSERT_EQ(values.size(), 2U);
      for (std::size_t i = 0; i < values.size(); ++i) {
        std::array<std::byte, 4> bytes{};
        const auto copied = values[i].copy_to_host(bytes.data(), bytes.size(), copy.stream);
        ASSERT_TRUE(copied) << copied.error().message;
        EXPECT_TRUE(std::all_of(bytes.begin(), bytes.end(), [i](auto b) {
          return b == static_cast<std::byte>(11U * (i + 1U));
        }));
      }
    } else {
      const auto values = result.snapshots<Tensor>("out");
      ASSERT_EQ(values.size(), 2U);
      EXPECT_EQ(values[0].bytes(), std::vector<std::byte>(4U, std::byte{11}));
      EXPECT_EQ(values[1].bytes(), std::vector<std::byte>(4U, std::byte{22}));
    }
  }
}

// REQ: OPERATOR-070
TEST(OperatorTestingManagedTensor, RestartDoesNotMakeAnIncompleteRetainedPrefixSuccessful) {
  auto probe = std::make_shared<ManagedTensorProbe>();
  auto h = create_operator_test<ManagedTensorSource<Tensor>>(probe, Tensor{});
  h.capture<Tensor>("out", Retain{.max_records = 8U});
  h.configure_graph([](auto& view) {
    view.graph().lifecycle().allow_external_restart(view.tested_operator());
    view.compile_options().lifecycle_transactions = tensor_restart_capacity();
  });
  auto run = h.run_async({.expect_stop = true});
  ASSERT_TRUE(post_tensor(probe->wait_sender(1U)));
  ASSERT_TRUE(run.wait_for_output("out", 1U, 5s));
  const auto target = run.operator_ref();
  ASSERT_TRUE(target);
  const std::array targets{*target};
  constexpr std::array stages{LifecycleStage::kAllocate};
  auto control = run.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
  ASSERT_TRUE(control);
  const auto request = control->request_restart(*target, {.from_stage = LifecycleStage::kAllocate});
  ASSERT_TRUE(request);
  const auto terminal = control->await(*request, 5s);
  ASSERT_TRUE(terminal);
  ASSERT_EQ(terminal->state, LifecycleTransactionState::kPublished);
  run.request_stop();
  const auto result = run.finish();
  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.termination(), RunTermination::kStopped);
  EXPECT_TRUE(result.diagnostics().empty()) << result.describe_failure();
  EXPECT_NE(result.describe_failure().find("Retain completion boundary was not reached"),
            std::string::npos);
  EXPECT_EQ(result.describe_failure().find("watchdog expired"), std::string::npos);
  const auto values = result.values<Tensor>("out");
  ASSERT_EQ(values.size(), 1U);
  EXPECT_EQ(values[0].data_as<std::uint8_t>()[0], 11U);
}

// REQ: OPERATOR-073
// REQ: OPERATOR-068
TEST(OperatorTestingManagedTensor, RestartWithOutstandingCUDAReadinessResumesSnapshots) {
  int devices{};
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    GTEST_SKIP() << "CUDA device unavailable";
  auto probe = std::make_shared<ManagedTensorProbe>();
  probe->delay = true;
  auto h =
      create_operator_test<ManagedTensorSource<Tensor>>(probe, Tensor{}, false, false, 2U, true);
  h.capture_snapshot<Tensor>("out");
  h.output_device("out", DeviceId{devices > 1 ? 1 : 0});
  h.configure_graph([](auto& view) {
    view.graph().lifecycle().allow_external_restart(view.tested_operator());
    view.compile_options().lifecycle_transactions = tensor_restart_capacity();
  });
  auto run = h.run_async();
  struct ReleaseReadiness {
    std::shared_ptr<ManagedTensorProbe> probe;
    ~ReleaseReadiness() { probe->release_delay(); }
  } const release{probe};
  ASSERT_TRUE(post_tensor(probe->wait_sender(1U)));
  ASSERT_TRUE(probe->wait_delay());
  EXPECT_EQ(run.output_count("out"), 0U);
  const auto target = run.operator_ref();
  ASSERT_TRUE(target);
  const std::array targets{*target};
  constexpr std::array stages{LifecycleStage::kAllocate};
  auto control = run.lifecycle_control({.allowed_targets = targets, .allowed_from_stages = stages});
  ASSERT_TRUE(control);
  const auto request = control->request_restart(*target, {.from_stage = LifecycleStage::kAllocate});
  ASSERT_TRUE(request);
  probe->release_delay();
  const auto terminal = control->await(*request, 5s);
  ASSERT_TRUE(terminal);
  ASSERT_EQ(terminal->state, LifecycleTransactionState::kPublished);
  ASSERT_TRUE(post_tensor(probe->wait_sender(2U)));
  const auto result = run.finish();
  ASSERT_TRUE(result.ok()) << result.describe_failure();
  const auto values = result.snapshots<Tensor>("out");
  ASSERT_FALSE(values.empty());
  EXPECT_EQ(values.back().bytes(), std::vector<std::byte>(4U, std::byte{22}));
  // Native restart may retire or discard the old queued publication; either way, any
  // captured predecessor bytes must be complete, and resumed data must be the successor.
  ASSERT_LE(values.size(), 2U);
  if (values.size() == 2U)
    EXPECT_EQ(values.front().bytes(), std::vector<std::byte>(4U, std::byte{11}));
  EXPECT_FALSE(probe->delay_expired);
}

namespace {
struct HeldInputCudaStopProbe {
  std::mutex mutex;
  std::condition_variable changed;
  bool gate_entered{};
  bool gate_released{};
  bool gate_expired{};
  bool tail_completed{};
  bool stop_entered{};
  bool held_at_stop{};
  bool released_in_stop{};
  std::int32_t value_at_stop{};
  unsigned stops{};

  void release() {
    const std::lock_guard lock(mutex);
    gate_released = true;
    changed.notify_all();
  }
  bool wait_for(bool HeldInputCudaStopProbe::* flag) {
    std::unique_lock lock(mutex);
    return changed.wait_for(lock, 5s, [&] { return this->*flag; });
  }
  static void block_stream(void* opaque) {
    auto& probe = *static_cast<HeldInputCudaStopProbe*>(opaque);
    std::unique_lock lock(probe.mutex);
    probe.gate_entered = true;
    probe.changed.notify_all();
    probe.gate_expired = !probe.changed.wait_for(lock, 10s, [&] { return probe.gate_released; });
  }
  static void complete_stream(void* opaque) {
    auto& probe = *static_cast<HeldInputCudaStopProbe*>(opaque);
    const std::lock_guard lock(probe.mutex);
    probe.tail_completed = true;
    probe.changed.notify_all();
  }
};

struct ReleaseHeldInputCudaGate {
  std::shared_ptr<HeldInputCudaStopProbe> probe;
  ~ReleaseHeldInputCudaGate() { probe->release(); }
};

class HeldInputCudaStop final : public Operator<> {
 public:
  HeldInputCudaStop(std::shared_ptr<HeldInputCudaStopProbe> probe, bool fail_cleanup)
      : probe_(std::move(probe)), fail_cleanup_(fail_cleanup) {}
  ~HeldInputCudaStop() override {
    // Fallback also makes an early failed assertion safe. The test requires stop() itself
    // to release the claim and stream, so fallback cleanup cannot make it pass.
    if (stream_ != nullptr) {
      probe_->release();
      static_cast<void>(cudaStreamSynchronize(stream_));
      static_cast<void>(cudaStreamDestroy(stream_));
    }
  }
  void setup(OperatorSpec& spec) override {
    spec.input(in, "in");
    spec.output(out, "out");
  }
  void start() override {
    const auto status = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    if (status != cudaSuccess)
      throw std::runtime_error{cudaGetErrorString(status)};
  }
  expected<void, Error> compute(ExecutionContext&) override {
    auto received = in.receive();
    if (!received)
      return make_unexpected(std::move(received).error());
    held_.emplace(std::move(*received));
    auto status = cudaLaunchHostFunc(stream_, &HeldInputCudaStopProbe::block_stream, probe_.get());
    if (status == cudaSuccess)
      status = cudaLaunchHostFunc(stream_, &HeldInputCudaStopProbe::complete_stream, probe_.get());
    if (status != cudaSuccess)
      return make_unexpected(Error{ErrorCode::kFailure, cudaGetErrorString(status)});
    // The copied result can be observed while the original input claim and CUDA stream
    // work remain live. Capturing it does not prove drain or trigger an implicit stop.
    return out.emit(held_->data);
  }
  void stop() override {
    {
      const std::lock_guard lock(probe_->mutex);
      ++probe_->stops;
      probe_->stop_entered = true;
      probe_->held_at_stop = held_.has_value();
      if (held_)
        probe_->value_at_stop = held_->data;
      probe_->changed.notify_all();
    }
    const auto synchronized = cudaStreamSynchronize(stream_);
    const auto destroyed = cudaStreamDestroy(stream_);
    stream_ = nullptr;
    held_.reset();
    {
      const std::lock_guard lock(probe_->mutex);
      probe_->released_in_stop = true;
    }
    if (synchronized != cudaSuccess || destroyed != cudaSuccess)
      throw std::runtime_error{"held-input CUDA cleanup failed"};
    if (fail_cleanup_)
      throw std::runtime_error{"injected managed cleanup failure after CUDA join"};
  }
  Input<std::int32_t> in;
  Output<std::int32_t> out;

 private:
  std::shared_ptr<HeldInputCudaStopProbe> probe_;
  bool fail_cleanup_;
  cudaStream_t stream_{};
  std::optional<Sample<std::int32_t>> held_;
};
}  // namespace

// REQ: OPERATOR-066
// REQ: OPERATOR-067
TEST(OperatorTestingManagedTensor, ExpectedStopAfterOutputJoinsCudaAndPreservesCleanupFailure) {
  int devices{};
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
    GTEST_SKIP() << "CUDA device unavailable";
  for (const bool fail_cleanup : {false, true}) {
    SCOPED_TRACE(fail_cleanup);
    auto probe = std::make_shared<HeldInputCudaStopProbe>();
    auto h = create_operator_test<HeldInputCudaStop>(probe, fail_cleanup);
    h.push("in", std::int32_t{83});
    h.capture<std::int32_t>("out");
    auto running = h.run_async({.timeout = 20s, .expect_stop = true});
    const ReleaseHeldInputCudaGate release_on_early_exit{probe};
    ASSERT_TRUE(probe->wait_for(&HeldInputCudaStopProbe::gate_entered));
    ASSERT_TRUE(running.wait_for_output("out", 1U, 5s));
    EXPECT_EQ(running.output_count("out"), 1U);
    running.request_stop();
    auto joined = std::async(std::launch::async, [&] { return running.finish(); });
    // Declared after the future: release the CUDA callback before its destructor can wait
    // during exceptional test exit. The outer guard handles all earlier assertions.
    const ReleaseHeldInputCudaGate release_before_future{probe};
    const bool entered_stop = probe->wait_for(&HeldInputCudaStopProbe::stop_entered);
    EXPECT_TRUE(entered_stop);
    if (entered_stop) {
      EXPECT_EQ(joined.wait_for(0ms), std::future_status::timeout);
      const std::lock_guard lock(probe->mutex);
      EXPECT_TRUE(probe->held_at_stop);
      EXPECT_FALSE(probe->tail_completed);
      EXPECT_FALSE(probe->released_in_stop);
    }
    probe->release();
    const auto result = joined.get();
    EXPECT_FALSE(running.is_running());
    EXPECT_EQ(result.values<std::int32_t>("out"), (std::vector<std::int32_t>{83}));
    EXPECT_EQ(result.ok(), !fail_cleanup) << result.describe_failure();
    EXPECT_EQ(result.termination(),
              fail_cleanup ? RunTermination::kFailed : RunTermination::kStopped);
    if (fail_cleanup) {
      EXPECT_NE(result.describe_failure().find("injected managed cleanup failure after CUDA join"),
                std::string::npos);
      EXPECT_TRUE(std::any_of(
          result.diagnostics().begin(), result.diagnostics().end(), [](const auto& diagnostic) {
            return diagnostic.severity == DiagnosticSeverity::kError;
          }));
    } else {
      EXPECT_TRUE(result.diagnostics().empty());
      EXPECT_TRUE(result.describe_failure().empty());
    }
    EXPECT_EQ(result.describe_failure().find("watchdog expired"), std::string::npos);
    {
      const std::lock_guard lock(probe->mutex);
      EXPECT_EQ(probe->stops, 1U);
      EXPECT_TRUE(probe->held_at_stop);
      EXPECT_EQ(probe->value_at_stop, 83);
      EXPECT_TRUE(probe->tail_completed);
      EXPECT_TRUE(probe->released_in_stop);
      EXPECT_FALSE(probe->gate_expired);
    }
    EXPECT_EQ(running.finish().ok(), !fail_cleanup);
    EXPECT_NO_THROW(h.set_compile_options({}));
  }
}
