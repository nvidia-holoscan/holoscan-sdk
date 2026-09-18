/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/rate.hpp>
#include <holoscan/core/rate_generation.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

namespace {

constexpr DLDataType kByteDtype{kDLUInt, 8U, 1U};
constexpr std::uint64_t kCameraByteSpan = 3840U * 2160U * 4U;

enum class OutputStorage : std::uint8_t {
  kHost,
  kPinned,
};

[[nodiscard]] constexpr holoscan::MemoryKind memory_kind(OutputStorage storage) noexcept {
  return storage == OutputStorage::kHost ? holoscan::MemoryKind::kHost
                                         : holoscan::MemoryKind::kPinnedHost;
}

struct Arguments {
  std::vector<std::uint32_t> output_shape{1080U, 1920U, 4U};
  OutputStorage output_storage{OutputStorage::kHost};
};

[[nodiscard]] bool parse_arguments(int argc, char** argv, Arguments& result) {
  if (argc == 1) {
    return true;
  }
  if (argc != 4 && argc != 5) {
    std::cerr << "usage: " << argv[0] << " [HEIGHT WIDTH CHANNELS [host|pinned]]\n";
    return false;
  }
  for (std::size_t index = 0U; index < result.output_shape.size(); ++index) {
    const std::string_view text{argv[index + 1U]};
    std::uint32_t value = 0U;
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc() || end != text.data() + text.size() || value == 0U) {
      std::cerr << "shape extents must be positive uint32 values\n";
      return false;
    }
    result.output_shape[index] = value;
  }
  if (result.output_shape[2] != 4U) {
    std::cerr << "CHANNELS must be 4 for the packed RGBA example\n";
    return false;
  }
  if (argc == 5) {
    const std::string_view storage{argv[4]};
    if (storage == "host") {
      result.output_storage = OutputStorage::kHost;
    } else if (storage == "pinned") {
      result.output_storage = OutputStorage::kPinned;
    } else {
      std::cerr << "memory kind must be host or pinned\n";
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool checked_multiply(std::uint64_t left, std::uint64_t right,
                                    std::uint64_t& result) noexcept {
  if (right != 0U && left > std::numeric_limits<std::uint64_t>::max() / right) {
    return false;
  }
  result = left * right;
  return true;
}

struct ResizeParams {
  // A deployment controls the exact HWC output shape without changing setup() or rebuilding the
  // Operator. Because this value determines pool capacity, changing it requires a new plan.
  std::vector<std::uint32_t> output_shape{720U, 1280U, 4U};
  OutputStorage output_storage{OutputStorage::kHost};
};

HOLOSCAN_PARAM_SCHEMA(
    ResizeParams, holoscan::param_schema("com.nvidia.holoscan.examples.DerivedResizeParams", 1),
    holoscan::field<&ResizeParams::output_shape>("output_shape")
        .max_elements(3U)
        .recompile_required()
        .description("Exact HWC output shape used to size the compiled Tensor pool."),
    holoscan::field<&ResizeParams::output_storage>("output_storage")
        .enum_values({holoscan::param_enum_value("host", OutputStorage::kHost),
                      holoscan::param_enum_value("pinned", OutputStorage::kPinned)})
        .recompile_required()
        .description("Physical memory kind selected when the Tensor pool is compiled."));

struct BatchProjectionParams {
  std::uint32_t width{1280U};
  std::uint32_t height{720U};
  std::uint32_t max_batch_size{2U};
};

HOLOSCAN_PARAM_SCHEMA(
    BatchProjectionParams,
    holoscan::param_schema("com.nvidia.holoscan.examples.BatchProjectionParams", 1),
    holoscan::field<&BatchProjectionParams::width>("width").range(1U, 16384U).recompile_required(),
    holoscan::field<&BatchProjectionParams::height>("height")
        .range(1U, 16384U)
        .recompile_required(),
    holoscan::field<&BatchProjectionParams::max_batch_size>("max_batch_size")
        .range(1U, 64U)
        .recompile_required());

class CameraSource final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(output, "output")
        .nominal_rate(holoscan::Rate::hz(60U))
        .produces_tensor(holoscan::TensorOutputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = kByteDtype,
                               .rank = 3U},
            .bounds = holoscan::tensor_bounds(kCameraByteSpan),
            .storage = holoscan::TensorOutputStorage::kRuntimePool,
        });
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Output<holoscan::Tensor> output;
};

class ConfiguredResize final : public holoscan::Operator<ResizeParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .expected_rate(holoscan::Rate::hz(60U))
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = kByteDtype,
                               .rank = 3U},
        });
    spec.output(output, "output")
        .follows(input, holoscan::RateRatio{1U, 4U})
        .produces_tensor(
            holoscan::TensorOutputSpec{
                .representation = {.dtype = kByteDtype, .rank = 3U},
                .storage = holoscan::TensorOutputStorage::kRuntimePool,
            },
            holoscan::TensorOutputDerivations{
                // Reduce the configured HWC shape and the output's exact dtype to one immutable
                // byte capacity during compilation.
                .max_byte_span = holoscan::parameter_contiguous_byte_span(
                    holoscan::param_ref<&ResizeParams::output_shape>()),
                // Resolve the module-specific enum through a closed mapping. Runtime receives
                // only the resulting concrete capacity and memory kind in its pool plan.
                .memory_kind = holoscan::parameter_memory_kind(
                    holoscan::param_ref<&ResizeParams::output_storage>(),
                    {holoscan::memory_kind_case(OutputStorage::kHost, holoscan::MemoryKind::kHost),
                     holoscan::memory_kind_case(OutputStorage::kPinned,
                                                holoscan::MemoryKind::kPinnedHost)}),
            });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<holoscan::Tensor> input;
  holoscan::Output<holoscan::Tensor> output;
};

class ProjectedBatchOutput final : public holoscan::Operator<BatchProjectionParams> {
 public:
  [[nodiscard]] static holoscan::expected<std::uint64_t, holoscan::Error> output_max_byte_span(
      const BatchProjectionParams& params) noexcept {
    constexpr std::uint64_t kRgbaChannels = 4U;
    constexpr std::uint64_t kFloatBytes = sizeof(float);
    std::uint64_t result = params.width;
    for (const std::uint64_t factor : {static_cast<std::uint64_t>(params.height),
                                       static_cast<std::uint64_t>(params.max_batch_size),
                                       kRgbaChannels,
                                       kFloatBytes}) {
      std::uint64_t next = 0U;
      if (!checked_multiply(result, factor, next)) {
        return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kArithmeticOverflow});
      }
      result = next;
    }
    return result;
  }

  void setup(holoscan::OperatorSpec& spec) override {
    // This dependency list is normative compiler input, not documentation: it must name every
    // Params field that output_max_byte_span reads, because Core cannot inspect the function body.
    const auto capacity =
        holoscan::parameter_projected_byte_span<&ProjectedBatchOutput::output_max_byte_span>(
            "com.nvidia.holoscan.examples.ProjectedBatchOutput.output-max-byte-span.v1",
            holoscan::param_ref<&BatchProjectionParams::width>(),
            holoscan::param_ref<&BatchProjectionParams::height>(),
            holoscan::param_ref<&BatchProjectionParams::max_batch_size>());
    spec.output(output, "output")
        .produces_tensor(
            holoscan::TensorOutputSpec{
                .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                                   .dtype = DLDataType{kDLFloat, 32U, 1U},
                                   .rank = 4U},
                .storage = holoscan::TensorOutputStorage::kRuntimePool,
            },
            capacity);
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Output<holoscan::Tensor> output;
};

class ProjectedBatchSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = DLDataType{kDLFloat, 32U, 1U},
                               .rank = 4U},
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<holoscan::Tensor> input;
};

class PackedGrayscale final : public holoscan::Operator<> {
 public:
  explicit PackedGrayscale(holoscan::MemoryKind input_memory_kind)
      : input_memory_kind_(input_memory_kind) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = {.memory_kind = input_memory_kind_, .dtype = kByteDtype, .rank = 3U},
        });
    spec.output(output, "output")
        .produces_tensor(
            holoscan::TensorOutputSpec{
                .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                                   .dtype = kByteDtype,
                                   .rank = 2U},
                .storage = holoscan::TensorOutputStorage::kRuntimePool,
            },
            // Packed RGBA uses four input bytes for every one-byte grayscale output pixel, so this
            // checked scalar formula gives an exact cap. A real Operator must not assume this ratio
            // when the input bound includes padded rows or other non-packed layouts. Expressions
            // can also use multiplied_by(), maximum_with(), and aligned_to() as needed.
            holoscan::input_max_byte_span(input).ceiling_divided_by(
                holoscan::tensor_byte_span(4U)));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::seconds{1}})
        .input(input, holoscan::Exact{})
        .on_missing(input, holoscan::Skip{});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<holoscan::Tensor> input;
  holoscan::Output<holoscan::Tensor> output;

 private:
  holoscan::MemoryKind input_memory_kind_;
};

class ConservativeGrayscale final : public holoscan::Operator<> {
 public:
  explicit ConservativeGrayscale(holoscan::MemoryKind input_memory_kind)
      : input_memory_kind_(input_memory_kind) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = {.memory_kind = input_memory_kind_, .dtype = kByteDtype, .rank = 3U},
        });
    spec.output(output, "output")
        .produces_tensor(
            holoscan::TensorOutputSpec{
                .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                                   .dtype = kByteDtype,
                                   .rank = 2U},
                .storage = holoscan::TensorOutputStorage::kRuntimePool,
            },
            // When no tighter scalar proof is available, preserve the input's entire finite byte
            // cap. This may over-reserve each grayscale slot, but never assumes packed rows.
            holoscan::input_max_byte_span(input));
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnEach{input});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<holoscan::Tensor> input;
  holoscan::Output<holoscan::Tensor> output;

 private:
  holoscan::MemoryKind input_memory_kind_;
};

// Example-test helper for inspecting compiler evidence. Ordinary application operators allocate
// through Output<Tensor> and do not search ExecutionPlan allocation records themselves.
[[nodiscard]] const holoscan::TensorAllocationPlanRecord* allocation_for(
    const holoscan::ExecutionPlan& plan, holoscan::PortId output) {
  const auto found = std::ranges::find_if(plan.tensor_allocations(), [&](const auto& allocation) {
    return allocation.output == output;
  });
  return found == plan.tensor_allocations().end() ? nullptr : &*found;
}

struct LifecycleCameraParams {
  std::uint64_t maximum_frame_bytes{6U * 1024U * 1024U};
};

HOLOSCAN_PARAM_SCHEMA(
    LifecycleCameraParams,
    holoscan::param_schema("com.nvidia.holoscan.examples.LifecycleCameraParams", 1),
    holoscan::field<&LifecycleCameraParams::maximum_frame_bytes>("maximum_frame_bytes")
        .recompile_required()
        .description("Maximum logical payload bytes in one camera frame."));

struct LifecycleRateExampleState {
  static void reset() noexcept {
    discover_calls.store(0U, std::memory_order_release);
    failure.store(0U, std::memory_order_release);
    allow_restart.store(false, std::memory_order_release);
    first_observed.store(false, std::memory_order_release);
    second_observed.store(false, std::memory_order_release);
    first_run_generation.store(0U, std::memory_order_release);
    first_rate_generation.store(0U, std::memory_order_release);
    second_run_generation.store(0U, std::memory_order_release);
    second_rate_generation.store(0U, std::memory_order_release);
  }

  static void fail(std::uint32_t code) noexcept {
    std::uint32_t expected{};
    static_cast<void>(failure.compare_exchange_strong(expected, code, std::memory_order_acq_rel));
  }

  [[nodiscard]] static holoscan::Rate next_rate() noexcept {
    return discover_calls.fetch_add(1U, std::memory_order_acq_rel) == 0U ? holoscan::Rate::hz(60U)
                                                                         : holoscan::Rate::hz(30U);
  }

  static inline std::atomic<std::uint32_t> discover_calls{};
  static inline std::atomic<std::uint32_t> failure{};
  static inline std::atomic<bool> allow_restart{};
  static inline std::atomic<bool> first_observed{};
  static inline std::atomic<bool> second_observed{};
  static inline std::atomic<std::uint64_t> first_run_generation{};
  static inline std::atomic<std::uint64_t> first_rate_generation{};
  static inline std::atomic<std::uint64_t> second_run_generation{};
  static inline std::atomic<std::uint64_t> second_rate_generation{};
};

class LifecycleCameraSource final : public holoscan::Operator<LifecycleCameraParams> {
 public:
  void setup(holoscan::OperatorSpec& spec) override {
    spec.output(frame, "frame")
        .lifecycle_nominal_rate(holoscan::LifecycleStage::kDiscover)
        .max_nominal_rate(holoscan::Rate::hz(120U))
        .late_joiner_history_depth(4U)
        .produces_tensor(
            holoscan::TensorOutputSpec{
                .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                                   .dtype = kByteDtype,
                                   .rank = 2U},
                .storage = holoscan::TensorOutputStorage::kRuntimePool,
            },
            holoscan::parameter_byte_span(
                holoscan::param_ref<&LifecycleCameraParams::maximum_frame_bytes>()));
    spec.lifecycle()
        .stage(holoscan::LifecycleStage::kDiscover, &LifecycleCameraSource::discover)
        .stage(holoscan::LifecycleStage::kAllocate, &LifecycleCameraSource::allocate);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override {
    const auto view = context.rate(frame);
    if (!view || view->state != holoscan::PortRateViewState::kResolved ||
        view->origin != holoscan::ResolvedRateOrigin::kLifecycleBinding ||
        view->max_nominal_rate != holoscan::Rate::hz(120U) || view->run_generation == 0U ||
        view->rate_generation == 0U) {
      LifecycleRateExampleState::fail(1U);
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kFailure, "rate view unavailable"});
    }
    if (phase_ == 0U) {
      if (view->resolved_nominal_rate != holoscan::Rate::hz(60U)) {
        LifecycleRateExampleState::fail(2U);
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "initial rate mismatch"});
      }
      LifecycleRateExampleState::first_run_generation.store(view->run_generation,
                                                            std::memory_order_release);
      LifecycleRateExampleState::first_rate_generation.store(view->rate_generation,
                                                             std::memory_order_release);
      LifecycleRateExampleState::first_observed.store(true, std::memory_order_release);
      if (!LifecycleRateExampleState::allow_restart.load(std::memory_order_acquire)) {
        return {};
      }
      auto receipt = context.request_restart(
          holoscan::LifecycleRestartOptions{.from_stage = holoscan::LifecycleStage::kDiscover});
      if (!receipt) {
        LifecycleRateExampleState::fail(3U);
        return holoscan::make_unexpected(std::move(receipt.error()));
      }
      phase_ = 1U;
      return {};
    }
    if (phase_ == 1U) {
      if (view->resolved_nominal_rate != holoscan::Rate::hz(30U)) {
        LifecycleRateExampleState::fail(4U);
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "restart rate mismatch"});
      }
      LifecycleRateExampleState::second_run_generation.store(view->run_generation,
                                                             std::memory_order_release);
      LifecycleRateExampleState::second_rate_generation.store(view->rate_generation,
                                                              std::memory_order_release);
      LifecycleRateExampleState::second_observed.store(true, std::memory_order_release);
      phase_ = 2U;
    }
    return {};
  }

  holoscan::LifecycleStatus discover(holoscan::LifecycleContext& context) noexcept {
    const holoscan::Rate selected = LifecycleRateExampleState::next_rate();
    const auto bound = context.bind_nominal_rate(frame, selected);
    const auto repeated = context.bind_nominal_rate(
        frame, holoscan::Rate::hz(selected.numerator_hz * 2U, selected.denominator_hz * 2U));
    const auto pending = context.rate(frame);
    if (!bound || !repeated || !pending ||
        pending->state != holoscan::PortRateViewState::kPendingBarrier ||
        pending->resolved_nominal_rate.has_value()) {
      LifecycleRateExampleState::fail(5U);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::LifecycleStatus allocate(holoscan::LifecycleContext& context) noexcept {
    const auto view = context.rate(frame);
    const holoscan::Rate expected =
        LifecycleRateExampleState::discover_calls.load(std::memory_order_acquire) == 1U
            ? holoscan::Rate::hz(60U)
            : holoscan::Rate::hz(30U);
    if (!view || view->state != holoscan::PortRateViewState::kResolved ||
        view->resolved_nominal_rate != expected ||
        view->max_nominal_rate != holoscan::Rate::hz(120U)) {
      LifecycleRateExampleState::fail(6U);
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    return holoscan::LifecycleStatus::kOk;
  }

  holoscan::Output<holoscan::Tensor> frame;

 private:
  std::uint32_t phase_{};
};

class LifecycleCameraSink final : public holoscan::Operator<> {
 public:
  explicit LifecycleCameraSink(bool clocked) noexcept : clocked_(clocked) {}

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(frame, "frame")
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = {.memory_kind = holoscan::MemoryKind::kHost,
                               .dtype = kByteDtype,
                               .rank = 2U},
        });
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    if (clocked_) {
      result.trigger(holoscan::OnClock{.period = std::chrono::microseconds{62500}});
    } else {
      result.trigger(holoscan::OnEach{frame});
    }
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<holoscan::Tensor> frame;

 private:
  bool clocked_{};
};

[[nodiscard]] bool wait_for(const std::atomic<bool>& value) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (!value.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return value.load(std::memory_order_acquire);
}

[[nodiscard]] const holoscan::ProviderAllocationGenerationRecord* provider_for(
    const holoscan::RateGenerationSnapshot& snapshot,
    const holoscan::RateGenerationEvidenceRecord& generation,
    holoscan::ProviderAllocationPlanRecordId provider) {
  const auto records = snapshot.provider_allocations(generation);
  const auto found =
      std::ranges::find(records, provider, &holoscan::ProviderAllocationGenerationRecord::recipe);
  return found == records.end() ? nullptr : &*found;
}

[[nodiscard]] const holoscan::PoolCapacityGenerationRecord* pool_for(
    const holoscan::RateGenerationSnapshot& snapshot,
    const holoscan::RateGenerationEvidenceRecord& generation,
    holoscan::ProviderAllocationPlanRecordId provider) {
  const auto records = snapshot.pool_capacities(generation);
  const auto found =
      std::ranges::find(records, provider, &holoscan::PoolCapacityGenerationRecord::recipe);
  return found == records.end() ? nullptr : &*found;
}

[[nodiscard]] bool run_lifecycle_rate_example() {
  constexpr std::uint64_t kFrameBytes = 6U * 1024U * 1024U;
  LifecycleRateExampleState::reset();

  holoscan::Graph graph{"lifecycle-rate-and-parameter-capacity"};
  auto source = graph.op<LifecycleCameraSource>("camera");
  const auto recorder = graph.op<LifecycleCameraSink>("recorder", true);
  source.set_params(LifecycleCameraParams{.maximum_frame_bytes = kFrameBytes});
  graph.add_flow(source->frame, recorder->frame);
  graph.lifecycle().allow_self_restart(source);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("camera-clock"));

  holoscan::CompileOptions options;
  options.lifecycle_transactions = holoscan::LifecycleTransactionOptions{
      .max_pending_requests = 2U,
      .max_pending_transactions = 2U,
      .max_receipts = 4U,
      .max_fenced_operators = 4U,
      .max_candidate_generations = 4U,
      .max_events = 32U,
      .max_audit_records = 8U,
      .max_quiesce_records = 8U,
      .max_rollback_records = 16U,
  };
  const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
  if (!plan.ok() || plan.tensor_allocations().size() != 1U ||
      plan.provider_allocation_plans().size() != 1U ||
      plan.tensor_allocations().front().bounds.max_byte_span != kFrameBytes ||
      plan.tensor_allocations().front().slot_count != 22U ||
      plan.tensor_allocations().front().aggregate_byte_capacity != 132U * 1024U * 1024U) {
    const auto actual_slots =
        plan.tensor_allocations().empty() ? 0U : plan.tensor_allocations().front().slot_count;
    const auto actual_bytes = plan.tensor_allocations().empty()
                                  ? 0U
                                  : plan.tensor_allocations().front().aggregate_byte_capacity;
    std::cerr << "finite-maximum lifecycle pool mismatch: expected_slots=22 actual_slots="
              << actual_slots << " expected_bytes=" << 132U * 1024U * 1024U
              << " actual_bytes=" << actual_bytes
              << " allocations=" << plan.tensor_allocations().size()
              << " providers=" << plan.provider_allocation_plans().size() << '\n';
    return false;
  }
  const auto provider = plan.provider_allocation_plans().front().id;

  holoscan::RunSession session = holoscan::run_async(plan);
  if (!wait_for(LifecycleRateExampleState::first_observed)) {
    session.request_stop();
    session.wait();
    std::cerr << "initial lifecycle rate was not published\n";
    return false;
  }
  const auto initial = session.capture_rate_generations();
  if (!initial || initial->published() == nullptr) {
    session.request_stop();
    session.wait();
    std::cerr << "initial rate-generation evidence was unavailable\n";
    return false;
  }
  LifecycleRateExampleState::allow_restart.store(true, std::memory_order_release);
  if (!wait_for(LifecycleRateExampleState::second_observed)) {
    session.request_stop();
    session.wait();
    std::cerr << "restarted lifecycle rate was not published\n";
    return false;
  }
  const auto restarted = session.capture_rate_generations();
  session.request_stop();
  session.wait();
  if (!restarted || restarted->published() == nullptr ||
      session.termination() != holoscan::RunTermination::kStopped ||
      !session.diagnostics().empty() ||
      LifecycleRateExampleState::failure.load(std::memory_order_acquire) != 0U) {
    std::cerr << "lifecycle rate run did not stop cleanly\n";
    return false;
  }

  const auto* initial_provider = provider_for(*initial, *initial->published(), provider);
  const auto* restarted_provider = provider_for(*restarted, *restarted->published(), provider);
  const auto* initial_pool = pool_for(*initial, *initial->published(), provider);
  const auto* restarted_pool = pool_for(*restarted, *restarted->published(), provider);
  if (initial_provider == nullptr || restarted_provider == nullptr || initial_pool == nullptr ||
      restarted_pool == nullptr || initial_provider->action != holoscan::AllocationAction::kFresh ||
      restarted_provider->action != holoscan::AllocationAction::kReusedLarger ||
      !initial_provider->allocation_generation || !restarted_provider->allocation_generation ||
      initial_provider->allocation_generation != restarted_provider->allocation_generation ||
      !initial_provider->storage_binding_generation ||
      !restarted_provider->storage_binding_generation ||
      initial_provider->storage_binding_generation ==
          restarted_provider->storage_binding_generation ||
      initial_pool->required_pool_entries != 14U ||
      initial_pool->admitted_pool_entry_ceiling != 22U ||
      initial_pool->max_payload_byte_span_per_slot != kFrameBytes ||
      initial_pool->logical_payload_capacity_bytes != 84U * 1024U * 1024U ||
      initial_pool->admitted_logical_payload_capacity_ceiling_bytes != 132U * 1024U * 1024U ||
      restarted_pool->required_pool_entries != 10U ||
      LifecycleRateExampleState::first_run_generation.load(std::memory_order_acquire) !=
          LifecycleRateExampleState::second_run_generation.load(std::memory_order_acquire) ||
      LifecycleRateExampleState::first_rate_generation.load(std::memory_order_acquire) ==
          LifecycleRateExampleState::second_rate_generation.load(std::memory_order_acquire)) {
    std::cerr << "rate, pool, or allocation-generation evidence did not match the example\n";
    return false;
  }

  const auto lifecycle_edge = std::ranges::find_if(plan.payload_edges(), [&](const auto& edge) {
    const auto source = std::ranges::find(plan.ports(), edge.source, &holoscan::PortPlanRecord::id);
    const auto target = std::ranges::find(plan.ports(), edge.target, &holoscan::PortPlanRecord::id);
    return source != plan.ports().end() && target != plan.ports().end() &&
           source->local_name == "frame" && target->local_name == "frame";
  });
  if (lifecycle_edge == plan.payload_edges().end()) {
    std::cerr << "lifecycle rate inspection edge was not found\n";
    return false;
  }
  const auto inspection =
      render_rate_queue_inspection(plan,
                                   *restarted,
                                   restarted->published()->rate_generation,
                                   lifecycle_edge->edge,
                                   holoscan::RateQueueInspectionFormat::kHumanText);
  const auto canonical_inspection =
      render_rate_queue_inspection(plan,
                                   *restarted,
                                   restarted->published()->rate_generation,
                                   lifecycle_edge->edge,
                                   holoscan::RateQueueInspectionFormat::kCanonicalJson);
  if (!inspection || !canonical_inspection ||
      inspection->find("edge: camera.frame -> recorder.frame") == std::string::npos ||
      inspection->find("provider allocations:") == std::string::npos ||
      canonical_inspection->find(
          R"("schema":"holoscan.rate_queue_inspection.canonical_json.v1")") == std::string::npos ||
      canonical_inspection->find(R"("logical_payload_capacity_bytes":)") == std::string::npos ||
      canonical_inspection->find(R"("reserved_bytes_total":)") == std::string::npos) {
    std::cerr << "rate queue inspection evidence was incomplete\n";
    return false;
  }

  std::cout << "lifecycle_rate=60/1 max_rate=120/1 active_entries=14 active_bytes="
            << *initial_pool->logical_payload_capacity_bytes
            << " admitted_entries=22 admitted_bytes="
            << *initial_pool->admitted_logical_payload_capacity_ceiling_bytes
            << " restart_rate=30/1 restart_entries=" << restarted_pool->required_pool_entries
            << " allocation=reused-larger binding=fresh\n";
  std::cout << "rate_inspection_human=" << *inspection << "\n"
            << "rate_inspection_canonical_json=" << *canonical_inspection << '\n';
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Arguments arguments;
  if (!parse_arguments(argc, argv, arguments)) {
    return 1;
  }

  holoscan::Graph graph{"derived-tensor-pool-capacity"};
  const auto selected_memory_kind = memory_kind(arguments.output_storage);
  const auto camera = graph.op<CameraSource>("camera");
  auto resize = graph.op<ConfiguredResize>("resize");
  auto projected_batch = graph.op<ProjectedBatchOutput>("projected_batch");
  const auto projected_batch_sink = graph.op<ProjectedBatchSink>("projected_batch_sink");
  const auto packed_grayscale = graph.op<PackedGrayscale>("packed_grayscale", selected_memory_kind);
  const auto conservative_grayscale =
      graph.op<ConservativeGrayscale>("conservative_grayscale", selected_memory_kind);
  graph.add_flow(camera->output, resize->input);
  graph.add_flow(projected_batch->output, projected_batch_sink->input);
  graph.add_flow(resize->output, packed_grayscale->input);
  graph.add_flow(resize->output,
                 conservative_grayscale->input,
                 holoscan::ConnectionOptions{.queue_depth = 4U});
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("app-clock"));

  // A deployment supplies this aggregate and recompiles a plan; setup() and the binary stay fixed.
  resize.set_params(ResizeParams{.output_shape = arguments.output_shape,
                                 .output_storage = arguments.output_storage});
  constexpr std::uint32_t kExampleMaxBatchSize = 2U;
  projected_batch.set_params(BatchProjectionParams{.width = arguments.output_shape[1],
                                                   .height = arguments.output_shape[0],
                                                   .max_batch_size = kExampleMaxBatchSize});

  // Timing compilation is example instrumentation used to illustrate the cold-path cost; normal
  // applications only need to compile the graph and retain the accepted plan.
  const auto compile_started = std::chrono::steady_clock::now();
  const holoscan::ExecutionPlan plan = holoscan::compile(graph);
  const auto compile_elapsed = std::chrono::steady_clock::now() - compile_started;
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }

  // Everything below is example self-validation and explanatory reporting. It demonstrates where
  // the resolved memory facts appear, but typical applications pass the plan to run()/run_async()
  // without walking these compiler records or printing artifact statistics.
  if (plan.tensor_byte_span_derivations().size() != 4U) {
    std::cerr << "expected four derived output capacities\n";
    return 2;
  }
  if (plan.tensor_memory_kind_derivations().size() != 1U) {
    std::cerr << "expected one derived output memory kind\n";
    return 3;
  }
  const auto* resize_derivation = plan.find_tensor_byte_span_derivation("resize", "output");
  const auto* packed_derivation =
      plan.find_tensor_byte_span_derivation("packed_grayscale", "output");
  const auto* conservative_derivation =
      plan.find_tensor_byte_span_derivation("conservative_grayscale", "output");
  const auto* projected_batch_derivation =
      plan.find_tensor_byte_span_derivation("projected_batch", "output");
  if (resize_derivation == nullptr || packed_derivation == nullptr ||
      conservative_derivation == nullptr || projected_batch_derivation == nullptr) {
    std::cerr << "expected named resize, grayscale, and projected-batch derivations\n";
    return 3;
  }
  const auto* resize_memory_derivation =
      plan.find_tensor_memory_kind_derivation("resize", "output");
  if (resize_memory_derivation == nullptr ||
      resize_memory_derivation->resolved_memory_kind != selected_memory_kind) {
    std::cerr << "configured output memory kind was not resolved\n";
    return 3;
  }

  const auto resize_edges = std::ranges::count_if(plan.payload_edges(), [&](const auto& edge) {
    return edge.source == resize_derivation->output;
  });
  const auto fixed_rate = std::ranges::find_if(plan.port_rates(), [](const auto& record) {
    return record.kind == holoscan::PortRateKind::kFixedNominal &&
           record.declared_nominal_rate == holoscan::Rate::hz(60U);
  });
  const auto followed_rate = std::ranges::find_if(plan.port_rates(), [](const auto& record) {
    return record.kind == holoscan::PortRateKind::kFollowsInput &&
           record.follows_ratio == holoscan::RateRatio{1U, 4U};
  });
  bool saw_automatic_depth = false;
  bool saw_explicit_depth = false;
  for (const auto& edge : plan.payload_edges()) {
    if (edge.source != resize_derivation->output) {
      continue;
    }
    const auto recipe = std::ranges::find(
        plan.queue_sizing_recipes(), edge.edge, &holoscan::QueueSizingRecipePlanRecord::edge);
    if (recipe == plan.queue_sizing_recipes().end()) {
      continue;
    }
    saw_automatic_depth = saw_automatic_depth ||
                          (edge.queue_depth == 30U && !recipe->explicit_depth.has_value() &&
                           recipe->origin == holoscan::QueueDepthOrigin::kRateAndContractDerived);
    saw_explicit_depth =
        saw_explicit_depth || (edge.queue_depth == 4U && recipe->explicit_depth == 4U &&
                               recipe->origin == holoscan::QueueDepthOrigin::kAddFlowOption);
  }
  if (resize_edges != 2U || fixed_rate == plan.port_rates().end() ||
      followed_rate == plan.port_rates().end() || !saw_automatic_depth || !saw_explicit_depth) {
    std::cerr << "fixed, expected, follows, automatic, and explicit rate evidence is incomplete\n";
    return 4;
  }
  const auto report_capacity = [&](std::string_view label, const auto& derivation) {
    const auto* allocation = allocation_for(plan, derivation.output);
    if (allocation == nullptr) {
      return false;
    }
    std::cout << "output=" << label << " per_slot_bytes=" << allocation->bounds.max_byte_span
              << " slots=" << allocation->slot_count
              << " aggregate_bytes=" << allocation->aggregate_byte_capacity << '\n';
    return true;
  };
  if (!report_capacity("resize.output", *resize_derivation) ||
      !report_capacity("packed_grayscale.output", *packed_derivation) ||
      !report_capacity("conservative_grayscale.output", *conservative_derivation) ||
      !report_capacity("projected_batch.output", *projected_batch_derivation)) {
    std::cerr << "derived output has no authoritative allocation\n";
    return 5;
  }
  const auto* resize_allocation = allocation_for(plan, resize_derivation->output);
  const auto* packed_allocation = allocation_for(plan, packed_derivation->output);
  const auto* conservative_allocation = allocation_for(plan, conservative_derivation->output);
  const auto* projected_batch_allocation = allocation_for(plan, projected_batch_derivation->output);
  std::uint64_t configured_gray_byte_span = 0U;
  std::uint64_t configured_rgba_byte_span = 0U;
  std::uint64_t projected_batch_byte_span = 0U;
  if (!checked_multiply(
          arguments.output_shape[0], arguments.output_shape[1], configured_gray_byte_span) ||
      !checked_multiply(configured_gray_byte_span, 4U, configured_rgba_byte_span) ||
      !checked_multiply(configured_rgba_byte_span,
                        kExampleMaxBatchSize * sizeof(float),
                        projected_batch_byte_span) ||
      resize_allocation == nullptr || packed_allocation == nullptr ||
      conservative_allocation == nullptr || projected_batch_allocation == nullptr ||
      resize_allocation->memory_kind != selected_memory_kind ||
      resize_allocation->bounds.max_byte_span != configured_rgba_byte_span ||
      packed_allocation->bounds.max_byte_span != configured_gray_byte_span ||
      conservative_allocation->bounds.max_byte_span != configured_rgba_byte_span ||
      projected_batch_allocation->bounds.max_byte_span != projected_batch_byte_span) {
    std::cerr << "resolved capacities do not match configured and checked transform bounds\n";
    return 6;
  }

  if (!run_lifecycle_rate_example()) {
    return 7;
  }

  const auto compile_microseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(compile_elapsed).count();
  std::cout << "compile_us=" << compile_microseconds << " artifact_bytes=" << plan.json().size()
            << " resize_memory="
            << (selected_memory_kind == holoscan::MemoryKind::kHost ? "host" : "pinned")
            << " packed_grayscale_reserved_per_slot=" << packed_allocation->bounds.max_byte_span
            << " conservative_grayscale_reserved_per_slot="
            << conservative_allocation->bounds.max_byte_span << '\n';
  std::cout << "projector=ProjectedBatchOutput.output-max-byte-span.v1 max_batch_size="
            << kExampleMaxBatchSize
            << " projected_per_slot_bytes=" << projected_batch_allocation->bounds.max_byte_span
            << '\n';
  std::cout << "fixed_rate=60/1 follows_ratio=1/4 resolved_rate=15/1 automatic_depth=30"
               " explicit_depth=4\n";
  std::cout << "derived tensor pool capacity: configured and checked transform bounds verified\n";
  return 0;
}
