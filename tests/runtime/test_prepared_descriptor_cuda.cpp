/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "common/cuda_test_utils.hpp"
#include "compiler/compile_internal.hpp"
#include "runtime/runtime_test_access.hpp"
#include "support/prepared_descriptor_fixture.hpp"

namespace holoscan::test {
cudaError_t launch_borrowed_image_kernel(const std::uint64_t* input, std::uint64_t* output,
                                         std::uint64_t* observed, std::uint64_t increment,
                                         cudaStream_t stream);

namespace prepared_descriptor {
namespace {
using namespace std::chrono_literals;
constexpr std::array<std::uint64_t, 2> kValues{0x2a2a2a2a2a2a2a2aULL, 0x3b3b3b3b3b3b3b3bULL};
enum class Exit { kReuse, kFillFailure, kPublicationFailure, kCancel, kStop, kRestart, kComplete };
using LifecyclePoint = detail::RuntimeTensorOutputTestControl::LifecyclePoint;

template <typename Predicate>
bool until(Predicate&& predicate, std::chrono::steady_clock::duration timeout = 30s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate())
      return true;
    std::this_thread::yield();
  }
  return false;
}

struct CUDAProbe {
  ~CUDAProbe() {
    release();
    for (auto event : done)
      if (event)
        static_cast<void>(cudaEventDestroy(event));
    if (observed)
      static_cast<void>(cudaFreeHost(observed));
  }
  void release() {
    const std::lock_guard lock(mutex);
    released = true;
    cv.notify_all();
  }
  static void CUDART_CB block(void* opaque) {
    auto& self = *static_cast<CUDAProbe*>(opaque);
    std::unique_lock lock(self.mutex);
    self.entered.store(true, std::memory_order_release);
    // Outlast the test's 30 s polls so a stalled observation is reported with the gate held.
    if (!self.cv.wait_for(lock, 60s, [&] { return self.released; }))
      self.timed_out.store(true);
  }
  static void observe_lifecycle(void* opaque, LifecyclePoint point) noexcept {
    auto& self = *static_cast<CUDAProbe*>(opaque);
    if (point != self.lifecycle_point)
      return;
    std::unique_lock lock(self.mutex);
    self.lifecycle_entered.store(true, std::memory_order_release);
    if (!self.cv.wait_for(lock, 30s, [&] { return self.released; }))
      self.timed_out.store(true);
  }

  std::mutex mutex;
  std::condition_variable cv;
  bool released{};
  bool fanout{};
  std::atomic<bool> entered{};
  std::atomic<bool> timed_out{};
  std::atomic<bool> sibling_order_timed_out{};
  LifecyclePoint lifecycle_point{LifecyclePoint::kCudaQuiesce};
  std::atomic<bool> lifecycle_entered{};
  std::atomic<unsigned> callbacks{};
  std::atomic<unsigned> starts{};
  std::atomic<unsigned> probes{};
  std::atomic<unsigned> successful_probe_bundles{};
  std::atomic<bool> premature_reuse{};
  std::atomic<unsigned> recycled_fields{};
  std::array<std::atomic<const void*>, 2> source_pointers{};
  std::array<cudaStream_t, 2> streams{};
  std::array<cudaEvent_t, 2> done{};
  std::array<std::atomic<bool>, 2> recorded{};
  std::uint64_t* observed{};
};

class CUDASource final : public Operator<> {
 public:
  explicit CUDASource(std::shared_ptr<CUDAProbe> probe)
      : probe_(std::move(probe)), metadata_(make_metadata(kPatterns - 1U)) {}
  void setup(OperatorSpec& spec) override {
    declare_output(spec, output, MemoryKind::kCudaManaged);
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 100us});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    auto bundle = output.allocate_tensors(shape_requests());
    if (!bundle && bundle.error().code == ErrorCode::kResourceExhausted) {
      probe_->probes.fetch_add(1U);
      return {};
    }
    if (!bundle)
      return make_unexpected(std::move(bundle).error());
    const auto write = [&](TensorFieldSelector<Packet> field, std::size_t index) {
      return bundle->write_and_commit(
          field,
          context.cuda_stream(),
          [&](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
            if (!emitted_) {
              probe_->source_pointers[index].store(writer.data(), std::memory_order_release);
            } else if (writer.data() ==
                       probe_->source_pointers[index].load(std::memory_order_acquire)) {
              // A never-recorded event queries as complete, so an unsubmitted read is pending.
              const auto pending = [&](std::size_t reader) {
                return !probe_->recorded[reader].load(std::memory_order_acquire) ||
                       cudaEventQuery(probe_->done[reader]) != cudaSuccess;
              };
              if (pending(0U) || (probe_->fanout && pending(1U)))
                probe_->premature_reuse.store(true, std::memory_order_release);
              probe_->recycled_fields.fetch_or(1U << index, std::memory_order_release);
            }
            const int byte = emitted_ ? 0x5a : index == 0U ? 0x2a : 0x3b;
            return cudaMemsetAsync(
                       writer.data(), byte, sizeof(std::uint64_t), context.cuda_stream().get()) ==
                           cudaSuccess
                       ? expected<void, Error>{}
                       : expected<void, Error>{make_unexpected(Error{ErrorCode::kFailure})};
          });
    };
    HOLOSCAN_TRY(write(Fields::first, 0U));
    HOLOSCAN_TRY(write(Fields::second, 1U));
    if (emitted_) {
      probe_->successful_probe_bundles.fetch_add(1U);
      probe_->probes.fetch_add(1U);
      return {};
    }
    metadata_.sequence = 1U;
    auto emitted = output.emit_tensors(std::move(*bundle), metadata_);
    if (!emitted && emitted.error().code == ErrorCode::kBackpressured)
      return {};
    if (emitted)
      emitted_ = true;
    return emitted;
  }
  Output<Packet> output{"output"};

 private:
  std::shared_ptr<CUDAProbe> probe_;
  Packet metadata_;
  bool emitted_{};
};

class CUDATransform final : public Operator<> {
 public:
  CUDATransform(std::shared_ptr<CUDAProbe> probe, std::size_t index, Exit exit)
      : probe_(std::move(probe)), index_(index), exit_(exit), oversized_(make_metadata(0U)) {
    oversized_.label.assign(129U, 'x');
  }
  void setup(OperatorSpec& spec) override {
    declare_input(spec, input, MemoryKind::kCudaManaged);
    declare_output(spec, output, MemoryKind::kCudaManaged);
  }
  void start() override {
    if (!scratch_) {
      auto scratch = Scratch::create();
      if (!scratch)
        throw RuntimeError(scratch.error());
      scratch_.emplace(std::move(*scratch));
    }
    probe_->starts.fetch_add(1U, std::memory_order_release);
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    HOLOSCAN_TRY(auto view, input.selected_sample());
    const auto& value = view.data();
    if (!value.first || !value.second ||
        value.first->data() != probe_->source_pointers[0].load(std::memory_order_acquire) ||
        value.second->data() != probe_->source_pointers[1].load(std::memory_order_acquire))
      return make_unexpected(Error{ErrorCode::kFailure});
    HOLOSCAN_TRY(scratch_->copy_metadata_from(value));
    edit_metadata(scratch_->descriptor());
    HOLOSCAN_TRY(auto bundle, output.allocate_tensors(shape_requests()));
    const auto stream = context.cuda_stream().get();
    probe_->streams[index_] = stream;
    if (stream == nullptr)
      return make_unexpected(Error{ErrorCode::kFailure});
    // Distinct streams may share a hardware work queue (CUDA_DEVICE_MAX_CONNECTIONS), and work
    // queued behind the blocked callback waits for the gate. Gate only after the sibling submits.
    if (index_ == 0U && probe_->fanout &&
        !until([&] { return probe_->recorded[1].load(std::memory_order_acquire); }))
      probe_->sibling_order_timed_out.store(true);
    const auto write =
        [&](TensorFieldSelector<Packet> field, std::size_t field_index, const Tensor& source) {
          return bundle.write_and_commit(
              field,
              context.cuda_stream(),
              [&](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
                if (writer.data() == value.first->data() || writer.data() == value.second->data())
                  return make_unexpected(Error{ErrorCode::kFailure});
                if (index_ == 0U && field_index == 0U &&
                    cudaLaunchHostFunc(stream, CUDAProbe::block, probe_.get()) != cudaSuccess)
                  return make_unexpected(Error{ErrorCode::kFailure});
                if (launch_borrowed_image_kernel(source.data_as<std::uint64_t>(),
                                                 writer.data_as<std::uint64_t>(),
                                                 probe_->observed + index_ * 2U + field_index,
                                                 index_ * 10U + field_index + 1U,
                                                 stream) != cudaSuccess)
                  return make_unexpected(Error{ErrorCode::kFailure});
                if (field_index == 1U) {
                  if (cudaEventRecord(probe_->done[index_], stream) != cudaSuccess)
                    return make_unexpected(Error{ErrorCode::kFailure});
                  probe_->recorded[index_].store(true, std::memory_order_release);
                }
                return exit_ == Exit::kFillFailure && field_index == 1U
                           ? expected<void, Error>{make_unexpected(Error{ErrorCode::kFailure})}
                           : expected<void, Error>{};
              });
        };
    auto result = write(Fields::first, 0U, *value.first);
    if (result)
      result = write(Fields::second, 1U, *value.second);
    if (result && exit_ != Exit::kCancel) {
      const Packet& metadata =
          exit_ == Exit::kPublicationFailure ? oversized_ : scratch_->descriptor();
      result =
          output.emit_tensors(std::move(bundle), metadata, sensor_io::derive_from(view.metadata()));
    }
    if (exit_ == Exit::kRestart) {
      HOLOSCAN_TRY(auto receipt, context.request_restart({.from_stage = LifecycleStage::kStart}));
      static_cast<void>(receipt);
    }
    if (exit_ == Exit::kComplete)
      context.request_completion();
    probe_->callbacks.fetch_add(1U, std::memory_order_release);
    return result;
  }
  Input<Packet> input{"input"};
  Output<Packet> output{"output"};

 private:
  std::shared_ptr<CUDAProbe> probe_;
  std::size_t index_;
  Exit exit_;
  Packet oversized_;
  std::optional<Scratch> scratch_;
};

class CUDASink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { declare_input(spec, input, MemoryKind::kCudaManaged); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto view, input.selected_sample());
    const auto& value = view.data();
    return value.first && value.second && !value.label.empty() && value.label.front() == 'e' &&
                   !value.readings.empty() && value.readings.front() == 14.F
               ? expected<void, Error>{}
               : expected<void, Error>{make_unexpected(Error{ErrorCode::kFailure})};
  }
  Input<Packet> input{"input"};
};

ExecutionPlan cuda_plan(const std::shared_ptr<CUDAProbe>& probe, Exit exit, bool fanout,
                        const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control) {
  Graph graph{"prepared-descriptor-cuda-lifetime"};
  auto source = graph.op<CUDASource>("source", probe);
  auto transform = graph.op<CUDATransform>("transform", probe, std::size_t{0}, exit);
  auto sink = graph.op<CUDASink>("sink");
  graph.add_flow(source->output, transform->input, ConnectionOptions{.queue_depth = 1U});
  graph.add_flow(transform->output, sink->input, ConnectionOptions{.queue_depth = 1U});
  graph.partition("source").add(source);
  graph.partition("transform").add(transform);
  graph.partition("sink").add(sink);
  CompileOptions options;
  options.deployment.bind_tensor_output_device({"source", "output", DeviceId{0}});
  options.deployment.bind_tensor_output_device({"transform", "output", DeviceId{0}});
  if (fanout) {
    auto sibling = graph.op<CUDATransform>("sibling", probe, std::size_t{1}, Exit::kReuse);
    auto sibling_sink = graph.op<CUDASink>("sibling_sink");
    graph.add_flow(source->output, sibling->input, ConnectionOptions{.queue_depth = 1U});
    graph.add_flow(sibling->output, sibling_sink->input, ConnectionOptions{.queue_depth = 1U});
    graph.partition("sibling").add(sibling);
    graph.partition("sibling_sink").add(sibling_sink);
    options.deployment.bind_tensor_output_device({"sibling", "output", DeviceId{0}});
  }
  if (exit == Exit::kRestart) {
    graph.lifecycle().allow_self_restart(transform);
    options.lifecycle_transactions = LifecycleTransactionOptions{.max_pending_requests = 4,
                                                                 .max_pending_transactions = 4,
                                                                 .max_receipts = 8,
                                                                 .max_fenced_operators = 8,
                                                                 .max_candidate_generations = 8,
                                                                 .max_events = 64,
                                                                 .max_audit_records = 16,
                                                                 .max_quiesce_records = 16,
                                                                 .max_rollback_records = 32};
  }
  graph.set_default_clock(graph.add_clock<RealtimeClock>("clock"));
  const auto plan = detail::compile_with_process_local_holomq(graph, options);
  return plan.ok() ? detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control)
                   : plan;
}

std::vector<std::size_t> source_pool_indices(const ExecutionPlan& plan) {
  std::vector<std::size_t> result;
  for (const auto& vertex : plan.graph().vertices()) {
    if (vertex.name != "source")
      continue;
    for (const auto& port : plan.ports()) {
      if (port.owner != vertex.id || port.direction != PortDirection::kOutput)
        continue;
      for (std::size_t index = 0U; index < plan.tensor_field_allocations().size(); ++index)
        if (plan.tensor_field_allocations()[index].output == port.id)
          result.push_back(index);
    }
  }
  return result;
}

// Failure-only context for a stalled CUDA observation: the device and driver settings that
// influence stream-to-hardware-queue assignment, alongside stream, event, and source-pool state.
std::string describe_cuda_state(const CUDAProbe& probe,
                                const detail::RuntimeTensorOutputTestControl& control,
                                const std::vector<std::size_t>& source_pools) {
  std::ostringstream out;
  out << describe_cuda_device();
  for (std::size_t i = 0U; i < probe.streams.size(); ++i) {
    // The worker publishes its stream handle before release-storing the recorded-event flag.
    const bool recorded = probe.recorded[i].load(std::memory_order_acquire);
    if (!recorded) {
      out << "\nstream[" << i << "]=<not yet published> done recorded=0";
      continue;
    }
    const auto stream = probe.streams[i];
    int priority = 0;
    static_cast<void>(cudaStreamGetPriority(stream, &priority));
    out << "\nstream[" << i << "]=" << stream << " priority=" << priority
        << " query=" << cudaGetErrorName(cudaStreamQuery(stream)) << " done recorded=" << recorded
        << " query=" << cudaGetErrorName(cudaEventQuery(probe.done[i]));
  }
  out << "\ngate entered=" << probe.entered.load() << " timed_out=" << probe.timed_out.load()
      << " sibling order timed_out=" << probe.sibling_order_timed_out.load();
  for (const auto index : source_pools) {
    const auto pool = control.snapshot(index);
    out << "\nsource pool " << index << " readiness=" << pool.cuda_readiness_records
        << " readiness_elisions=" << pool.cuda_readiness_record_elisions
        << " stream_waits=" << pool.cuda_stream_waits
        << " stream_wait_elisions=" << pool.cuda_stream_wait_elisions
        << " host_waits=" << pool.cuda_host_waits;
  }
  return out.str();
}

void exercise_cuda(Exit exit, bool fanout) {
  SCOPED_TRACE(fanout);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  auto probe = std::make_shared<CUDAProbe>();
  probe->fanout = fanout;
  // The host reads kernel results while other partitions keep the GPU busy. Devices without
  // concurrent managed access (for example Jetson Orin) forbid host access to managed memory
  // while GPU work is pending, so the results live in pinned host memory instead.
  ASSERT_EQ(cudaMallocHost(reinterpret_cast<void**>(&probe->observed), 4U * sizeof(std::uint64_t)),
            cudaSuccess);
  std::fill_n(probe->observed, 4U, std::uint64_t{0});
  ASSERT_EQ(launch_borrowed_image_kernel(
                probe->observed, probe->observed + 1U, probe->observed + 1U, 1U, nullptr),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  for (auto& event : probe->done)
    ASSERT_EQ(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), cudaSuccess);

  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  const bool observe_lifecycle = exit != Exit::kReuse && exit != Exit::kCancel;
  if (observe_lifecycle) {
    probe->lifecycle_point =
        exit == Exit::kRestart ? LifecyclePoint::kRestartFenced : LifecyclePoint::kCudaQuiesce;
    control->set_lifecycle_observer(probe.get(), &CUDAProbe::observe_lifecycle);
  }
  const auto plan = cuda_plan(probe, exit, fanout, control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  ASSERT_EQ(plan.tensor_allocations().size(), 0U);
  ASSERT_EQ(plan.tensor_field_allocations().size(), fanout ? 6U : 4U);
  const auto source_pools = source_pool_indices(plan);
  ASSERT_EQ(source_pools.size(), 2U);
  for (std::size_t index = 0U; index < plan.tensor_field_allocations().size(); ++index) {
    const auto& allocation = plan.tensor_field_allocations()[index];
    const bool source = index == source_pools[0] || index == source_pools[1];
    // Each transported subscriber retains an endpoint head beside its queue and typed holder.
    EXPECT_EQ(allocation.slot_count, source ? (fanout ? 8U : 5U) : 5U);
  }

  const auto consumers = fanout ? 2U : 1U;
  auto session = run_async(plan);
  struct Release {
    CUDAProbe& probe;
    ~Release() { probe.release(); }
  } release_on_exit{*probe};
  ASSERT_TRUE(until([&] {
    return probe->callbacks.load(std::memory_order_acquire) == consumers &&
           probe->entered.load(std::memory_order_acquire);
  })) << "callbacks="
      << probe->callbacks.load() << '\n'
      << describe_cuda_state(*probe, *control, source_pools);
  EXPECT_EQ(cudaEventQuery(probe->done[0]), cudaErrorNotReady);
  if (fanout) {
    EXPECT_NE(probe->streams[0], probe->streams[1]);
    EXPECT_FALSE(probe->sibling_order_timed_out.load());
    if (!until([&] { return cudaEventQuery(probe->done[1]) == cudaSuccess; })) {
      // Releasing the gate separates a dependency on the gated stream from any other stall.
      const auto stalled = describe_cuda_state(*probe, *control, source_pools);
      probe->release();
      const bool after_release =
          until([&] { return cudaEventQuery(probe->done[1]) == cudaSuccess; }, 5s);
      FAIL() << "sibling event incomplete while the delayed reader was gated; after gate release "
             << (after_release ? "it completed (it depended on the gated stream)"
                               : "it stayed incomplete")
             << '\n'
             << stalled;
    }
    EXPECT_EQ(probe->observed[2], kValues[0] + 11U);
    EXPECT_EQ(probe->observed[3], kValues[1] + 12U);
  }
  if (exit == Exit::kReuse || exit == Exit::kCancel) {
    const auto initial = probe->probes.load();
    ASSERT_TRUE(until([&] { return probe->probes.load() >= initial + 16U; }));
    for (const auto index : source_pools) {
      EXPECT_EQ(control->snapshot(index).physical_slot_count,
                plan.tensor_field_allocations()[index].slot_count);
      EXPECT_NE(control->snapshot(index).minimum_retired_slot, 0U);
    }
    EXPECT_FALSE(probe->premature_reuse.load());
  }
  if (exit == Exit::kStop)
    session.request_stop();
  if (observe_lifecycle) {
    // A callback counter alone precedes selection retirement. Hold the lifecycle owner at the
    // selected cold boundary as well as holding the GPU, so neither can race these observations.
    ASSERT_TRUE(until([&] { return probe->lifecycle_entered.load(std::memory_order_acquire); }))
        << "lifecycle boundary=" << static_cast<int>(probe->lifecycle_point);
    EXPECT_EQ(cudaEventQuery(probe->done[0]), cudaErrorNotReady);
    for (const auto index : source_pools) {
      const auto retained = control->snapshot(index);
      EXPECT_NE(retained.minimum_retired_slot, 0U);
      EXPECT_EQ(retained.metrics.oldest_retained_slot, 0U);
      EXPECT_NE(retained.metrics.oldest_retained_claim_mask, 0U);
      EXPECT_GT(retained.live_materialization_count, 0U);
      EXPECT_GT(retained.live_requested_payload_backing_bytes, 0U);
      EXPECT_FALSE(retained.terminal);
    }
    if (exit == Exit::kRestart)
      EXPECT_EQ(probe->starts.load(std::memory_order_acquire), 1U);
  }
  probe->release();
  ASSERT_TRUE(until([&] { return cudaEventQuery(probe->done[0]) == cudaSuccess; }))
      << describe_cuda_state(*probe, *control, source_pools);
  EXPECT_EQ(probe->observed[0], kValues[0] + 1U);
  EXPECT_EQ(probe->observed[1], kValues[1] + 2U);
  if (exit == Exit::kRestart)
    ASSERT_TRUE(until([&] { return probe->starts.load() == 2U; }));
  if (exit == Exit::kReuse || exit == Exit::kCancel || exit == Exit::kRestart) {
    for (const auto index : source_pools)
      ASSERT_TRUE(until([&] { return control->snapshot(index).minimum_retired_slot == 0U; }))
          << "source pool=" << index << " probes=" << probe->probes.load()
          << " session termination=" << static_cast<int>(session.termination());
    // Both original slots must return to their pools, but repeated cancellation can keep a
    // different slot on top of each LIFO free list forever. Require progress after retirement
    // without assuming which addresses a subsequent bundle receives.
    const auto bundles_after_retirement = probe->successful_probe_bundles.load();
    ASSERT_TRUE(until([&] {
      return probe->successful_probe_bundles.load() > bundles_after_retirement;
    })) << "source bundles="
        << probe->successful_probe_bundles.load() << " probes=" << probe->probes.load()
        << " recycled fields=" << probe->recycled_fields.load()
        << " session termination=" << static_cast<int>(session.termination());
  }
  const bool fails = exit == Exit::kFillFailure || exit == Exit::kPublicationFailure;
  if (!fails && exit != Exit::kComplete)
    session.request_stop();
  session.wait();
  EXPECT_EQ(session.termination(),
            fails                     ? RunTermination::kFailed
            : exit == Exit::kComplete ? RunTermination::kRequestedCompleted
                                      : RunTermination::kStopped);
  EXPECT_FALSE(probe->timed_out.load());
  EXPECT_FALSE(probe->premature_reuse.load());
  for (std::size_t index = 0U; index < plan.tensor_field_allocations().size(); ++index) {
    const auto retired = control->snapshot(index);
    EXPECT_EQ(retired.cuda_events_created, retired.cuda_events_destroyed);
    EXPECT_EQ(retired.cuda_streams_created, retired.cuda_streams_destroyed);
    EXPECT_EQ(retired.cuda_host_waits, 0U);
  }
}

// REQ: SCHEMA-009
// REQ: TENSOR-STORAGE-012
TEST(PreparedDescriptorCuda, DelayedTwoTensorReadsAndFanoutPreventPrematureReuse) {
  if (!has_cuda_device())
    GTEST_SKIP() << "CUDA device required";
  exercise_cuda(Exit::kReuse, false);
  exercise_cuda(Exit::kReuse, true);
}

// REQ: SCHEMA-009
// REQ: TENSOR-STORAGE-012
TEST(PreparedDescriptorCuda, SubmittedReadsSurviveFailureCancellationStopRestartAndCompletion) {
  if (!has_cuda_device())
    GTEST_SKIP() << "CUDA device required";
  for (const auto exit : {Exit::kFillFailure,
                          Exit::kPublicationFailure,
                          Exit::kCancel,
                          Exit::kStop,
                          Exit::kRestart,
                          Exit::kComplete}) {
    SCOPED_TRACE(static_cast<int>(exit));
    exercise_cuda(exit, false);
  }
}

}  // namespace
}  // namespace prepared_descriptor
}  // namespace holoscan::test
