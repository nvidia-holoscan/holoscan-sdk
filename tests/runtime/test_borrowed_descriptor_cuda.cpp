/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected_macros.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/sensor_io/provenance.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "common/cuda_test_utils.hpp"
#include "runtime/runtime_test_access.hpp"

namespace holoscan::test {
cudaError_t launch_borrowed_image_kernel(const std::uint64_t* input, std::uint64_t* output,
                                         std::uint64_t* observed, std::uint64_t increment,
                                         cudaStream_t stream);
namespace {
using namespace std::chrono_literals;
constexpr std::array<std::int64_t, 1> kShape{1};
constexpr std::uint64_t kPixels = 0x2a2a2a2a2a2a2a2aULL;
enum class Exit { kReuse, kFillFailure, kPublicationFailure, kCancel, kStop, kRestart, kComplete };

template <typename Predicate>
bool until(Predicate&& predicate) {
  const auto deadline = std::chrono::steady_clock::now() + 30s;
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
      static_cast<void>(cudaFree(observed));
  }
  std::mutex mutex;
  std::condition_variable cv;
  bool released{};
  std::atomic<bool> entered{};
  std::atomic<bool> timed_out{};
  std::atomic<unsigned> callbacks{};
  std::atomic<unsigned> starts{};
  std::atomic<unsigned> probes{};
  std::atomic<unsigned> successful_probe_loans{};
  std::atomic<bool> premature_reuse{};
  std::atomic<bool> recycled{};
  std::atomic<const void*> first_pointer{};
  std::array<cudaStream_t, 2> streams{};
  std::array<cudaEvent_t, 2> done{};
  std::uint64_t* observed{};
  void release() {
    const std::lock_guard lock(mutex);
    released = true;
    cv.notify_all();
  }
  static void CUDART_CB block(void* opaque) {
    auto& self = *static_cast<CUDAProbe*>(opaque);
    std::unique_lock lock(self.mutex);
    self.entered.store(true, std::memory_order_release);
    if (!self.cv.wait_for(lock, 30s, [&] { return self.released; })) {
      self.timed_out.store(true);
    }
  }
};

void input_spec(OperatorSpec& spec, Input<schema::ImageT>& input) {
  spec.input(input).queue_depth(1).expects_tensor(
      TensorInputSpec{.representation = {.memory_kind = MemoryKind::kCudaManaged,
                                         .dtype = DLDataType{kDLUInt, 64, 1},
                                         .rank = 1}});
}
void output_spec(OperatorSpec& spec, Output<schema::ImageT>& output) {
  spec.output(output).max_emits_per_compute(1).produces_tensor(
      TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kCudaManaged,
                                          .dtype = DLDataType{kDLUInt, 64, 1},
                                          .rank = 1},
                       .bounds = tensor_bounds(8),
                       .storage = TensorOutputStorage::kRuntimePool});
}

class CUDASource final : public Operator<> {
 public:
  explicit CUDASource(std::shared_ptr<CUDAProbe> probe) : probe_(std::move(probe)) {}
  void setup(OperatorSpec& spec) override { output_spec(spec, output); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = 100us});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    auto loan = output.allocate_tensor({.shape = kShape, .dtype = {kDLUInt, 64, 1}});
    if (!loan && loan.error().code == ErrorCode::kResourceExhausted) {
      probe_->probes.fetch_add(1);
      return {};
    }
    if (!loan)
      return make_unexpected(std::move(loan).error());
    HOLOSCAN_TRY(loan->write_and_commit(
        context.cuda_stream(), [&](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
          if (!emitted_) {
            probe_->first_pointer.store(writer.data());
          } else if (writer.data() == probe_->first_pointer.load()) {
            // Event completion, not a host release flag, is the reuse boundary under test.
            if (cudaEventQuery(probe_->done[0]) != cudaSuccess) {
              probe_->premature_reuse.store(true);
            }
            probe_->recycled.store(true);
          }
          if (cudaMemsetAsync(
                  writer.data(), emitted_ ? 0x5a : 0x2a, 8, context.cuda_stream().get()) !=
              cudaSuccess) {
            return make_unexpected(Error{ErrorCode::kFailure});
          }
          return {};
        }));
    if (emitted_) {
      probe_->successful_probe_loans.fetch_add(1);
      probe_->probes.fetch_add(1);
      return std::move(*loan).cancel();
    }
    schema::ImageT image;
    image.width = 1;
    image.height = 1;
    HOLOSCAN_TRY(output.emit_tensor(std::move(*loan), image));
    emitted_ = true;
    return {};
  }
  Output<schema::ImageT> output{"output"};

 private:
  std::shared_ptr<CUDAProbe> probe_;
  bool emitted_{};
};

class CUDATransform final : public Operator<> {
 public:
  CUDATransform(std::shared_ptr<CUDAProbe> probe, std::size_t index, Exit exit)
      : probe_(std::move(probe)), index_(index), exit_(exit) {}
  void setup(OperatorSpec& spec) override {
    input_spec(spec, input);
    output_spec(spec, output);
  }
  void start() override { probe_->starts.fetch_add(1); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    HOLOSCAN_TRY(auto view, input.selected_sample());
    HOLOSCAN_TRY(auto loan, output.allocate_tensor({.shape = kShape, .dtype = {kDLUInt, 64, 1}}));
    const auto stream = context.cuda_stream().get();
    probe_->streams[index_] = stream;
    if (!stream || !view.data().data)
      return make_unexpected(Error{ErrorCode::kFailure});
    if (view.data().data->data() != probe_->first_pointer.load()) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
    const auto filled = loan.write_and_commit(
        context.cuda_stream(), [&](const TensorOutputWriteGuard& writer) -> expected<void, Error> {
          // Only one host callback is blocked: CUDA may serialize host callbacks across streams.
          // The sibling still runs its kernel on a distinct admitted stream.
          if (index_ == 0 &&
              cudaLaunchHostFunc(stream, CUDAProbe::block, probe_.get()) != cudaSuccess) {
            return make_unexpected(Error{ErrorCode::kFailure});
          }
          if (launch_borrowed_image_kernel(view.data().data->data_as<std::uint64_t>(),
                                           writer.data_as<std::uint64_t>(),
                                           probe_->observed + index_,
                                           index_ + 1,
                                           stream) != cudaSuccess ||
              cudaEventRecord(probe_->done[index_], stream) != cudaSuccess) {
            return make_unexpected(Error{ErrorCode::kFailure});
          }
          return exit_ == Exit::kFillFailure ? make_unexpected(Error{ErrorCode::kFailure})
                                             : expected<void, Error>{};
        });
    expected<void, Error> result;
    if (!filled) {
      result = make_unexpected(filled.error());
    } else if (exit_ == Exit::kCancel) {
      result = std::move(loan).cancel();
    } else if (exit_ == Exit::kPublicationFailure) {
      schema::ImageT invalid;
      invalid.width = -1;
      result = output.emit_tensor(std::move(loan), invalid);
    } else {
      result =
          output.emit_tensor(std::move(loan), view.data(), sensor_io::derive_from(view.metadata()));
    }
    if (exit_ == Exit::kRestart) {
      HOLOSCAN_TRY(auto receipt, context.request_restart({.from_stage = LifecycleStage::kStart}));
      static_cast<void>(receipt);
    }
    if (exit_ == Exit::kComplete)
      context.request_completion();
    probe_->callbacks.fetch_add(1, std::memory_order_release);
    return result;
  }
  Input<schema::ImageT> input{"input"};
  Output<schema::ImageT> output{"output"};

 private:
  std::shared_ptr<CUDAProbe> probe_;
  std::size_t index_;
  Exit exit_;
};

class CUDASink final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { input_spec(spec, input); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnEach{input});
    return result;
  }
  expected<void, Error> compute(ExecutionContext&) override {
    HOLOSCAN_TRY(auto view, input.selected_sample());
    return view.data().width == 1 ? expected<void, Error>{}
                                  : make_unexpected(Error{ErrorCode::kFailure});
  }
  Input<schema::ImageT> input{"input"};
};

ExecutionPlan cuda_plan(const std::shared_ptr<CUDAProbe>& probe, Exit exit, bool fanout,
                        const std::shared_ptr<detail::RuntimeTensorOutputTestControl>& control) {
  Graph graph{"borrowed-image-cuda-lifetime"};
  auto source = graph.op<CUDASource>("source", probe);
  auto transform = graph.op<CUDATransform>("transform", probe, std::size_t{0}, exit);
  auto sink = graph.op<CUDASink>("sink");
  graph.add_flow(source->output, transform->input, ConnectionOptions{.queue_depth = 1});
  graph.add_flow(transform->output, sink->input, ConnectionOptions{.queue_depth = 1});
  graph.partition("source").add(source);
  graph.partition("transform").add(transform);
  graph.partition("sink").add(sink);
  CompileOptions options;
  options.deployment.bind_tensor_output_device({"source", "output", DeviceId{0}});
  options.deployment.bind_tensor_output_device({"transform", "output", DeviceId{0}});
  if (fanout) {
    auto sibling = graph.op<CUDATransform>("sibling", probe, std::size_t{1}, Exit::kReuse);
    auto sibling_sink = graph.op<CUDASink>("sibling_sink");
    graph.add_flow(source->output, sibling->input, ConnectionOptions{.queue_depth = 1});
    graph.add_flow(sibling->output, sibling_sink->input, ConnectionOptions{.queue_depth = 1});
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
  const auto plan = compile(graph, options);
  return plan.ok() ? detail::RuntimeTestPlanAccess::with_tensor_output_test_control(plan, control)
                   : plan;
}

void exercise_cuda(Exit exit, bool fanout) {
  SCOPED_TRACE(fanout);
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  auto probe = std::make_shared<CUDAProbe>();
  ASSERT_EQ(cudaMallocManaged(&probe->observed, 2 * sizeof(std::uint64_t)), cudaSuccess);
  // Load/JIT the kernel before installing a blocking callback. Lazy CUDA module loading can
  // synchronize the context on the first launch; that cold operation must not wait on our gate.
  ASSERT_EQ(cudaMemset(probe->observed, 0, 2 * sizeof(std::uint64_t)), cudaSuccess);
  ASSERT_EQ(launch_borrowed_image_kernel(
                probe->observed, probe->observed + 1, probe->observed + 1, 1, nullptr),
            cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);  // Cold test setup only.
  for (auto& event : probe->done) {
    ASSERT_EQ(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), cudaSuccess);
  }
  auto control = std::make_shared<detail::RuntimeTensorOutputTestControl>();
  const auto plan = cuda_plan(probe, exit, fanout, control);
  ASSERT_TRUE(plan.ok()) << plan.json();
  std::size_t source_pool = plan.tensor_allocations().size();
  for (const auto& vertex : plan.graph().vertices()) {
    if (vertex.name != "source")
      continue;
    for (const auto& port : plan.ports()) {
      if (port.owner != vertex.id || port.direction != PortDirection::kOutput)
        continue;
      for (std::size_t i = 0; i < plan.tensor_allocations().size(); ++i) {
        if (plan.tensor_allocations()[i].output == port.id)
          source_pool = i;
      }
    }
  }
  ASSERT_LT(source_pool, plan.tensor_allocations().size());
  const auto consumers = fanout ? 2U : 1U;
  auto session = run_async(plan);
  struct Release {
    CUDAProbe& probe;
    ~Release() { probe.release(); }
  } release_on_exit{*probe};  // Destroy before session, including fatal-assertion unwinding.
  ASSERT_TRUE(until([&] {
    return probe->callbacks.load(std::memory_order_acquire) == consumers &&
           probe->entered.load(std::memory_order_acquire);
  })) << "callbacks="
      << probe->callbacks.load() << " gate=" << probe->entered.load();
  EXPECT_EQ(cudaEventQuery(probe->done[0]), cudaErrorNotReady);
  if (fanout) {
    EXPECT_NE(probe->streams[0], probe->streams[1]);
    ASSERT_TRUE(until([&] { return cudaEventQuery(probe->done[1]) == cudaSuccess; }));
    EXPECT_EQ(probe->observed[1], kPixels + 2);
  }
  if (exit == Exit::kReuse || exit == Exit::kCancel) {
    const auto initial = probe->probes.load();
    ASSERT_TRUE(until([&] { return probe->probes.load() >= initial + 16; }));
    // Compiler-admitted capacities are used unchanged; slot zero is the first published slot.
    EXPECT_EQ(control->snapshot(source_pool).physical_slot_count,
              plan.tensor_allocations()[source_pool].slot_count);
    EXPECT_NE(control->snapshot(source_pool).minimum_retired_slot, 0U);
    EXPECT_FALSE(probe->premature_reuse.load());
  }
  if (exit == Exit::kStop)
    session.request_stop();
  probe->release();
  ASSERT_TRUE(until([&] { return cudaEventQuery(probe->done[0]) == cudaSuccess; }));
  EXPECT_EQ(probe->observed[0], kPixels + 1);
  if (exit == Exit::kReuse || exit == Exit::kCancel) {
    ASSERT_TRUE(until([&] { return control->snapshot(source_pool).minimum_retired_slot == 0; }));
    // A retired slot is available again, but the pool does not promise which free slot the next
    // loan selects. Prove that source loans continue after retirement without requiring slot zero
    // to be selected from a LIFO free list under repeated cancellation.
    const auto loans_after_retirement = probe->successful_probe_loans.load();
    ASSERT_TRUE(until([&] {
      return probe->successful_probe_loans.load() > loans_after_retirement;
    })) << "source loans="
        << probe->successful_probe_loans.load() << " probes=" << probe->probes.load()
        << " retired slot=" << control->snapshot(source_pool).minimum_retired_slot
        << " session termination=" << static_cast<int>(session.termination());
  }
  if (exit == Exit::kRestart) {
    ASSERT_TRUE(until([&] { return probe->starts.load() == 2; }));
    ASSERT_TRUE(until([&] { return probe->recycled.load(); }));
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
  for (std::size_t i = 0; i < plan.tensor_allocations().size(); ++i) {
    const auto retired = control->snapshot(i);
    EXPECT_EQ(retired.cuda_events_created, retired.cuda_events_destroyed);
    EXPECT_EQ(retired.cuda_streams_created, retired.cuda_streams_destroyed);
    EXPECT_EQ(retired.cuda_host_waits, 0U);
  }
}

// REQ: OPERATOR-056
// REQ: TENSOR-STORAGE-012
TEST(BorrowedDescriptorCuda, DelayedKernelAndFanoutPreventPrematureReuse) {
  if (!has_cuda_device())
    GTEST_SKIP() << "CUDA device required";
  exercise_cuda(Exit::kReuse, false);
  exercise_cuda(Exit::kReuse, true);
}

// REQ: OPERATOR-056
// REQ: TENSOR-STORAGE-012
TEST(BorrowedDescriptorCuda, SubmittedReadsSurviveFailureCancellationStopRestartAndCompletion) {
  if (!has_cuda_device())
    GTEST_SKIP() << "CUDA device required";
  for (auto exit : {Exit::kFillFailure,
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
}  // namespace holoscan::test
