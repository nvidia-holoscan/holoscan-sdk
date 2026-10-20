/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>

#include "execution_context_internal.hpp"

namespace holoscan::detail {

/// @brief Identify the accessor cardinality frozen by a private synthetic selector.
enum class SyntheticSelectorShape : std::uint8_t {
  kScalar,  ///< Scalar `receive()` and `receive_data()` accessors are admitted.
  kBatch,   ///< `selected_batch()` and `for_each_selected()` are admitted.
};

/// @brief Describe one test-only input binding without introducing a public testing API.
struct SyntheticInputState {
  SyntheticSelectorShape selector_shape{SyntheticSelectorShape::kScalar};
  InputSelection selection{InputState::kAbsent, SelectionReason::kNeverSeen};
  bool stale_generation{};
  bool closed{};
  bool advertises_value{};
  bool omit_batch_active_generation{};
  bool mismatch_batch_generation{};
  bool omit_scalar_read_claim{};
  bool return_partial_scalar_read_claim{};
  bool return_unexpected_scalar_read_claim{};
  bool omit_scalar_sample{};
  bool omit_scalar_view_generation{};
  bool zero_scalar_view_generation{};
  bool mismatch_scalar_view_generation{};
  ErrorCode backend_error{ErrorCode::kSuccess};
};

/// @brief Supply deterministic callback evidence to the private one-shot fixture.
struct SyntheticActivationOptions {
  Tag tag{.sequence = 1, .generation = 1};
  TagTime tag_time{};
  TimePoint activation_time{};
  ClockRef clock{};
  cudaStream_t cuda_stream{};
  TraceContext trace{};
  ActivationStatus activation{ActivationReason::kNominal, OnViolation::kEmitWitness};
  bool cancellation_requested{};
  std::optional<ParamGeneration> params_generation{};
  void* restart_state{};
  RestartRequestFunction request_restart{};
  void* termination_request_state{};
  TerminationRequestFunction request_termination{};
  OperatorRef self{};
};

/// @brief Own samples explicitly removed from a private synthetic output binding.
template <PublishedPayload T>
class SyntheticCapturedSamples {
 public:
  [[nodiscard]] std::size_t size() const noexcept { return samples_.size(); }
  [[nodiscard]] bool empty() const noexcept { return samples_.empty(); }
  [[nodiscard]] const Sample<T>& operator[](std::size_t index) const noexcept {
    return samples_[index];
  }
  [[nodiscard]] auto begin() const noexcept { return samples_.begin(); }
  [[nodiscard]] auto end() const noexcept { return samples_.end(); }

 private:
  explicit SyntheticCapturedSamples(std::vector<Sample<T>> samples)
      : samples_(std::move(samples)) {}

  std::vector<Sample<T>> samples_;

  friend class SyntheticOperatorBinding;
};

/// @brief Bind one real Operator callback to deterministic finite test state.
///
/// This non-installed implementation seam deliberately omits scheduler, graph, lifecycle,
/// Channel, waiting, and public `load`/`retrieve`/`run` behavior. All storage is allocated and
/// reserved before `compute()` begins. The callback executes exactly once per invocation and uses
/// the same late-bound port operations intended for the production runtime.
class SyntheticOperatorBinding {
 public:
  SyntheticOperatorBinding() = default;
  SyntheticOperatorBinding(const SyntheticOperatorBinding&) = delete;
  SyntheticOperatorBinding& operator=(const SyntheticOperatorBinding&) = delete;
  SyntheticOperatorBinding(SyntheticOperatorBinding&&) = delete;
  SyntheticOperatorBinding& operator=(SyntheticOperatorBinding&&) = delete;
  ~SyntheticOperatorBinding() {
    // Port declarations outlive this fixture. Retire only bindings that still name this fixture's
    // exact slot so destroying an older fixture cannot erase a newer runtime/test rebind.
    for (const std::unique_ptr<InputSlotBase>& input : inputs_) {
      input->unbind_if_owned();
    }
    for (const std::unique_ptr<OutputSlotBase>& output : outputs_) {
      output->unbind_if_owned();
    }
  }

  template <typename Op>
  [[nodiscard]] expected<void, Error> declare_ports(Op& op) {
    if (declared_) {
      return make_unexpected(Error{ErrorCode::kDuplicateName});
    }
    OperatorSpec spec;
    op.setup(spec);
    if (spec.declaration_error_.code != ErrorCode::kSuccess) {
      return make_unexpected(spec.declaration_error_);
    }
    if (!valid_builder_records(spec)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    declared_inputs_.reserve(spec.ports_.size());
    declared_outputs_.reserve(spec.ports_.size());
    inputs_.reserve(spec.ports_.size());
    outputs_.reserve(spec.ports_.size());
    for (const OperatorSpec::PortRecord& record : spec.ports_) {
      if (record.direction == PortDirection::kInput) {
        declared_inputs_.push_back(DeclaredInput{
            .declaration = record.declaration,
            .queue_depth = spec.input_builders_[record.builder_index].queue_depth_.value_or(1U),
        });
      } else {
        const std::uint32_t max_emits =
            record.tensor
                ? spec.tensor_output_builders_[record.builder_index].max_emits_per_compute_
                : spec.output_builders_[record.builder_index].max_emits_per_compute_;
        declared_outputs_.push_back(DeclaredOutput{
            .declaration = record.declaration,
            .max_emits_per_compute = max_emits,
        });
      }
    }
    declared_ = true;
    return {};
  }

  template <PublishedPayload T>
  [[nodiscard]] expected<void, Error> bind_scalar(Input<T>& port, Sample<T> sample,
                                                  InputSelection selection) {
    const SyntheticInputState state{
        .selector_shape = SyntheticSelectorShape::kScalar,
        .selection = selection,
        .advertises_value = true,
    };
    return bind_scalar_state(port, std::move(sample), state);
  }

  template <PublishedPayload T>
  [[nodiscard]] expected<void, Error> bind_scalar_state(Input<T>& port, Sample<T> sample,
                                                        SyntheticInputState state) {
    std::unique_ptr<InputSlot<T>> slot;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      slot = std::make_unique<InputSlot<T>>(state);
      slot->scalar.emplace(std::move(sample));
      return install_input(port, std::move(slot));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  template <PublishedPayload T>
  [[nodiscard]] expected<void, Error> bind_batch(Input<T>& port, std::span<const Sample<T>> samples,
                                                 InputSelection selection) {
    const DeclaredInput* const declaration = find_declared_input(&port);
    if (declaration == nullptr || samples.size() > declaration->queue_depth) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    SyntheticInputState state{
        .selector_shape = SyntheticSelectorShape::kBatch,
        .selection = selection,
        .advertises_value = true,
    };
    std::unique_ptr<InputSlot<T>> slot;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      slot = std::make_unique<InputSlot<T>>(state);
      slot->owned_batch.assign(samples.begin(), samples.end());
      slot->batch.reserve(slot->owned_batch.size());
      for (const Sample<T>& sample : slot->owned_batch) {
        slot->batch.push_back(&sample);
      }
      return install_input(port, std::move(slot));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  template <PublishedPayload T>
  [[nodiscard]] expected<void, Error> bind_batch_refs(Input<T>& port,
                                                      std::span<const Sample<T>* const> samples,
                                                      InputSelection selection) {
    const DeclaredInput* const declaration = find_declared_input(&port);
    if (declaration == nullptr || samples.size() > declaration->queue_depth) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    SyntheticInputState state{
        .selector_shape = SyntheticSelectorShape::kBatch,
        .selection = selection,
        .advertises_value = true,
    };
    std::unique_ptr<InputSlot<T>> slot;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      slot = std::make_unique<InputSlot<T>>(state);
      slot->batch.assign(samples.begin(), samples.end());
      if (std::any_of(slot->batch.begin(), slot->batch.end(), [](const Sample<T>* sample) {
            return sample == nullptr;
          })) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      return install_input(port, std::move(slot));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  template <PublishedPayload T>
  [[nodiscard]] expected<void, Error> bind_state(Input<T>& port, SyntheticInputState state) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      return install_input(port, std::make_unique<InputSlot<T>>(state));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  template <PublishedPayload T>
    requires(!std::same_as<T, Tensor>)
  [[nodiscard]] expected<void, Error> bind_output(
      Output<T>& port, std::optional<std::uint32_t> max_emits = std::nullopt) {
    const DeclaredOutput* const declaration = find_declared_output(&port);
    const std::uint32_t effective_max_emits =
        max_emits.value_or(declaration == nullptr ? 0 : declaration->max_emits_per_compute);
    if (declaration == nullptr || effective_max_emits == 0 ||
        effective_max_emits > declaration->max_emits_per_compute) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    if (has_output(&port)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    std::unique_ptr<OutputSlot<T>> slot;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      // Loan storage is constructed on the cold bind path. Allocation during compute only lends
      // already-constructed bounded storage and never grows this fixture.
      slot = std::make_unique<OutputSlot<T>>(&port, effective_max_emits);
      slot->captured.reserve(effective_max_emits);
      OutputSlot<T>* const slot_address = slot.get();
      replace_output(&port, std::move(slot));
      port.binding_ = OutputBinding<T>{.state = slot_address, .ops = &OutputSlot<T>::kOps};
      return {};
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  template <PublishedPayload T>
  [[nodiscard]] expected<void, Error> fail_next_loan_publication_indeterminate(
      Output<T>& port) noexcept {
    OutputSlot<T>* const slot = find_output<T>(&port);
    if (slot == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    slot->fail_next_loan_publication_indeterminate = true;
    return {};
  }

  template <typename Op>
  [[nodiscard]] expected<void, Error> invoke_once(Op& op, SyntheticActivationOptions options = {}) {
    if (!declared_) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    if (std::any_of(
            outputs_.begin(), outputs_.end(), [](const std::unique_ptr<OutputSlotBase>& output) {
              return output->has_captured();
            })) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    }

    constexpr bool parameterless = std::same_as<typename Op::Params, NoParams>;
    if constexpr (parameterless) {
      if (options.params_generation.has_value() && options.params_generation.value() != 0) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
    }
    // Parameterless operators always run at generation 0; others default to the first generation.
    const ParamGeneration params_generation =
        parameterless ? ParamGeneration{0} : options.params_generation.value_or(1);

    for (const std::unique_ptr<OutputSlotBase>& output : outputs_) {
      output->begin_activation();
    }
    ExecutionContextState state;
    state.tag = options.tag;
    state.tag_time = options.tag_time;
    state.activation_time = options.activation_time;
    state.clock = &options.clock;
    state.cuda_stream = options.cuda_stream;
    state.trace = options.trace;
    state.activation = options.activation;
    state.cancellation_requested.store(options.cancellation_requested, std::memory_order_relaxed);
    state.params = &op.params();
    state.params_type = op.defining_param_type().local_type_token;
    state.params_generation = params_generation;
    state.restart_state = options.restart_state;
    state.request_restart = options.request_restart;
    state.termination_request_state = options.termination_request_state;
    state.request_termination = options.request_termination;
    state.self = options.self;
    ExecutionContext context{&state};

    // A public semantic value copy may throw through compute() in admitted exception-enabled
    // profiles. Retire every activation-scoped binding on both the result and exception paths.
    const ActivationRetirement retirement{*this};
    return op.invoke_compute(context);
  }

  void retire_activation() noexcept {
    for (const std::unique_ptr<OutputSlotBase>& output : outputs_) {
      output->end_activation();
    }
    for (const std::unique_ptr<InputSlotBase>& input : inputs_) {
      input->invalidate();
    }
  }

  template <PublishedPayload T>
  [[nodiscard]] std::span<const Sample<T>> captured(const Output<T>& port) const noexcept {
    const OutputSlot<T>* const slot = find_output<T>(&port);
    return slot == nullptr ? std::span<const Sample<T>>{}
                           : std::span<const Sample<T>>{slot->captured};
  }

  template <PublishedPayload T>
  [[nodiscard]] std::span<const Sample<T>> captured_as(const OutputBase& port) const noexcept {
    const OutputSlot<T>* const slot = find_output<T>(&port);
    return slot == nullptr ? std::span<const Sample<T>>{}
                           : std::span<const Sample<T>>{slot->captured};
  }

  template <PublishedPayload T>
  [[nodiscard]] expected<SyntheticCapturedSamples<T>, Error> take_captured(const Output<T>& port) {
    OutputSlot<T>* const slot = find_output<T>(&port);
    if (slot == nullptr) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    std::vector<Sample<T>> samples = std::move(slot->captured);
    slot->captured.clear();
    slot->captured.reserve(slot->max_emits);
    return SyntheticCapturedSamples<T>{std::move(samples)};
  }

 private:
  struct ActivationRetirement {
    SyntheticOperatorBinding& owner;

    ~ActivationRetirement() noexcept { owner.retire_activation(); }
  };

  struct InputSlotBase {
    virtual ~InputSlotBase() = default;
    virtual void invalidate() noexcept = 0;
    virtual void unbind_if_owned() noexcept = 0;
    const void* declaration{};
  };

  template <PublishedPayload T>
  struct InputSlot final : InputSlotBase {
    struct ReadClaimPool;

    struct ReadClaimSlot {
      ReadClaimPool* pool{};
      std::atomic<std::uint32_t> references{};
      std::atomic<bool> in_use{};
    };

    struct ReadClaimPool {
      ReadClaimPool() noexcept {
        for (ReadClaimSlot& slot : slots) {
          slot.pool = this;
        }
      }

      [[nodiscard]] expected<PublishedReadClaimBinding, Error> acquire() noexcept {
        for (ReadClaimSlot& slot : slots) {
          bool available = false;
          if (slot.in_use.compare_exchange_strong(
                  available, true, std::memory_order_acquire, std::memory_order_relaxed)) {
            slot.references.store(1, std::memory_order_relaxed);
            references.fetch_add(1, std::memory_order_relaxed);
            return PublishedReadClaimBinding{
                .state = &slot,
                .retain = &retain,
                .release = &release,
            };
          }
        }
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }

      void release_owner() noexcept {
        if (references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
          delete this;
        }
      }

      static void retain(void* opaque) noexcept {
        auto& slot = *static_cast<ReadClaimSlot*>(opaque);
        slot.references.fetch_add(1, std::memory_order_relaxed);
      }

      static void release(void* opaque) noexcept {
        auto& slot = *static_cast<ReadClaimSlot*>(opaque);
        if (slot.references.fetch_sub(1, std::memory_order_acq_rel) != 1) {
          return;
        }
        slot.in_use.store(false, std::memory_order_release);
        ReadClaimPool* const pool = slot.pool;
        if (pool->references.fetch_sub(1, std::memory_order_acq_rel) == 1) {
          delete pool;
        }
      }

      static constexpr std::size_t kDefaultCapacity = 2;
      std::array<ReadClaimSlot, kDefaultCapacity> slots{};
      std::atomic<std::uint32_t> references{1};
    };

    explicit InputSlot(SyntheticInputState state) : state(state), claim_pool(new ReadClaimPool) {}

    ~InputSlot() override { claim_pool->release_owner(); }

    static InputSelection selection(const void* opaque, std::uint64_t generation) noexcept {
      const auto& slot = *static_cast<const InputSlot*>(opaque);
      if (slot.state.stale_generation || generation != slot.active_generation) {
        return {InputState::kAbsent, SelectionReason::kNeverSeen};
      }
      return slot.state.selection;
    }

    [[nodiscard]] static expected<void, Error> classify(const InputSlot& slot,
                                                        std::uint64_t generation,
                                                        SyntheticSelectorShape accessor) noexcept {
      if (slot.state.stale_generation || generation != slot.active_generation) {
        return make_unexpected(Error{ErrorCode::kStaleEndpoint});
      }
      if (slot.state.closed) {
        return make_unexpected(Error{ErrorCode::kClosed});
      }
      if (slot.state.selection.state == InputState::kNotSelected) {
        return make_unexpected(Error{ErrorCode::kInputNotSelected});
      }
      if (slot.state.selector_shape != accessor) {
        return make_unexpected(Error{ErrorCode::kInputSelectionCardinalityMismatch});
      }
      if (!slot.state.advertises_value) {
        return make_unexpected(Error{ErrorCode::kInputEmpty});
      }
      if (slot.state.backend_error != ErrorCode::kSuccess) {
        return make_unexpected(Error{slot.state.backend_error});
      }
      return {};
    }

    static expected<ScalarReadBinding<T>, Error> scalar_value(const void* opaque,
                                                              std::uint64_t generation,
                                                              bool acquire_claim) noexcept {
      auto& slot = *static_cast<InputSlot*>(const_cast<void*>(opaque));
      auto classified = classify(slot, generation, SyntheticSelectorShape::kScalar);
      if (!classified) {
        return make_unexpected(std::move(classified).error());
      }
      if (!slot.scalar.has_value()) {
        return make_unexpected(Error{ErrorCode::kReceiveError});
      }
      PublishedReadClaimBinding claim{};
      if (acquire_claim) {
        if (!slot.state.omit_scalar_read_claim) {
          auto acquired = slot.claim_pool->acquire();
          if (!acquired) {
            return make_unexpected(std::move(acquired).error());
          }
          claim = *acquired;
          if (slot.state.return_partial_scalar_read_claim) {
            claim.retain = nullptr;
          }
        }
      } else if (slot.state.return_unexpected_scalar_read_claim) {
        auto acquired = slot.claim_pool->acquire();
        if (!acquired) {
          return make_unexpected(std::move(acquired).error());
        }
        claim = *acquired;
        if (slot.state.return_partial_scalar_read_claim) {
          claim.retain = nullptr;
        }
      }
      return ScalarReadBinding<T>{
          .sample = slot.state.omit_scalar_sample ? nullptr : &*slot.scalar,
          .claim = claim,
          .view_generation =
              slot.state.omit_scalar_view_generation ? nullptr : &slot.view_generation,
          .expected_view_generation =
              slot.state.zero_scalar_view_generation
                  ? 0
                  : slot.view_generation + (slot.state.mismatch_scalar_view_generation ? 1 : 0),
      };
    }

    static expected<SelectedBatchBinding<T>, Error> batch_value(const void* opaque,
                                                                std::uint64_t generation) noexcept {
      const auto& slot = *static_cast<const InputSlot*>(opaque);
      auto classified = classify(slot, generation, SyntheticSelectorShape::kBatch);
      if (!classified) {
        return make_unexpected(std::move(classified).error());
      }
      return SelectedBatchBinding<T>{
          .samples = slot.batch.empty() ? nullptr : slot.batch.data(),
          .size = slot.batch.size(),
          .active_generation =
              slot.state.omit_batch_active_generation ? nullptr : &slot.active_generation,
          .expected_generation =
              slot.state.omit_batch_active_generation
                  ? 0
                  : (slot.state.mismatch_batch_generation ? generation + 1U : generation),
      };
    }

    void invalidate() noexcept override {
      ++view_generation;
      ++active_generation;
    }

    void unbind_if_owned() noexcept override {
      auto& port = *static_cast<Input<T>*>(const_cast<void*>(this->declaration));
      if (port.binding_.state == this && port.binding_.ops == &kOps) {
        port.binding_ = {};
      }
    }

    SyntheticInputState state;
    std::optional<Sample<T>> scalar;
    std::vector<Sample<T>> owned_batch;
    std::vector<const Sample<T>*> batch;
    std::uint64_t active_generation{1};
    std::uint64_t view_generation{1};
    ReadClaimPool* claim_pool{};

    static inline const InputBindingOps<T> kOps{
        .selection = &selection,
        .scalar = &scalar_value,
        .batch = &batch_value,
    };
  };

  struct OutputSlotBase {
    virtual ~OutputSlotBase() = default;
    [[nodiscard]] virtual bool has_captured() const noexcept = 0;
    virtual void unbind_if_owned() noexcept = 0;

    void begin_activation() noexcept {
      ++active_generation;
      active = true;
    }

    void end_activation() noexcept { active = false; }

    const void* declaration{};
    const void* payload_type{};
    bool active{};
    std::uint64_t active_generation{};
  };

  template <PublishedPayload T>
  struct OutputSlot final : OutputSlotBase {
    struct LoanControl {
      explicit LoanControl(OutputSlot* owner) : owner(owner) {
        if constexpr (std::default_initializable<T> && std::assignable_from<T&, T>) {
          value.emplace();
        }
      }

      OutputSlot* owner{};
      std::optional<T> value;
      std::uint32_t references{1};
      std::uint64_t activation_generation{};
      bool loan_live{};
    };

    OutputSlot(const Output<T>* declaration, std::uint32_t max_emits)
        : max_emits(max_emits), loan_control(new LoanControl(this)) {
      this->declaration = declaration;
      this->payload_type = &kSyntheticPayloadTypeToken<T>;
    }

    ~OutputSlot() override {
      loan_control->owner = nullptr;
      release_control(loan_control);
    }

    [[nodiscard]] bool has_captured() const noexcept override { return !captured.empty(); }

    void unbind_if_owned() noexcept override {
      auto& port = *static_cast<Output<T>*>(const_cast<void*>(this->declaration));
      if (port.binding_.state == this && port.binding_.ops == &kOps) {
        port.binding_ = {};
      }
    }

    static void cancel_loan(void* opaque) noexcept {
      auto& control = *static_cast<LoanControl*>(opaque);
      if (!control.loan_live) {
        return;
      }
      control.loan_live = false;
      release_control(&control);
    }

    static expected<LoanAllocation<T>, Error> allocate_loan(void* opaque) noexcept
      requires std::default_initializable<T> && std::assignable_from<T&, T>
    {  // NOLINT(whitespace/braces)
      auto& slot = *static_cast<OutputSlot*>(opaque);
      if (!slot.active) {
        return make_unexpected(Error{ErrorCode::kStaleEndpoint});
      }
      LoanControl& control = *slot.loan_control;
      if (control.loan_live) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      // The owner and at most one live loan retain this cold-allocated control block. The loan's
      // reference keeps both its value and tombstone alive across activation retirement, rebind,
      // and fixture destruction.
      ++control.references;
      control.loan_live = true;
      control.activation_generation = slot.active_generation;
      return LoanAllocation<T>{
          .value = &*control.value,
          .state = &control,
          .ops = &kLoanOps,
      };
    }

    static expected<void, Error> publish_loan(void* output_opaque, void* loan_opaque,
                                              EmitOptions options) noexcept
      requires std::default_initializable<T> && std::assignable_from<T&, T>
    {  // NOLINT(whitespace/braces)
      auto& slot = *static_cast<OutputSlot*>(output_opaque);
      if (slot.loan_control != loan_opaque) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      auto& control = *static_cast<LoanControl*>(loan_opaque);
      if (!control.loan_live) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument});
      }
      if (!slot.active || control.owner != &slot ||
          control.activation_generation != slot.active_generation) {
        return make_unexpected(Error{ErrorCode::kStaleEndpoint});
      }
      if (slot.captured.size() >= slot.max_emits) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      if (slot.fail_next_loan_publication_indeterminate) {
        // This bounded fixture injection models crossing the submit boundary without claiming a
        // transport. Sequence and sole publication authority are consumed exactly as they are for
        // a real indeterminate submission.
        slot.fail_next_loan_publication_indeterminate = false;
        ++slot.next_sequence;
        control.loan_live = false;
        release_control(&control);
        return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
      }

      SampleMetadata metadata{};
      if (options.capture_time.has_value()) {
        metadata.capture_timestamp_ns = options.capture_time->timestamp_ns;
        metadata.source_clock_id = options.capture_time->clock_id;
      }
      if (options.frame_id.has_value()) {
        metadata.frame_id = *options.frame_id;
      }
      if (options.trace.has_value()) {
        metadata.trace_id = options.trace->trace_id;
      }
      metadata.flags = options.flags;
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        Sample<T> sample{std::move(*control.value), metadata};
        sample.metadata.version = SampleMetadataVersion{};
        sample.metadata.sequence = slot.next_sequence++;
        if constexpr (std::same_as<T, Tensor>) {
          sample.metadata.payload_kind = PayloadKind::kDescriptorBacked;
          sample.metadata.byte_size = 0;
        } else if constexpr (kPublishedScalar<T>) {
          sample.metadata.payload_kind = PayloadKind::kInlineTyped;
          sample.metadata.byte_size = sizeof(T);
        } else {
          sample.metadata.payload_kind = PayloadKind::kSerializedBytes;
          sample.metadata.byte_size = 0;
        }
        slot.captured.push_back(std::move(sample));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (...) {
        // Construction or insertion may have transferred payload state after the submit boundary.
        // Consume authority and forbid retry of the same logical sample.
        control.loan_live = false;
        release_control(&control);
        return make_unexpected(Error{ErrorCode::kPublicationIndeterminate});
      }
#endif
      control.loan_live = false;
      release_control(&control);
      return {};
    }

    static expected<void, Error> publish(void* opaque, Sample<T> sample) noexcept {
      auto& slot = *static_cast<OutputSlot*>(opaque);
      if (!slot.active) {
        return make_unexpected(Error{ErrorCode::kStaleEndpoint});
      }
      if (slot.captured.size() >= slot.max_emits) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      sample.metadata.version = SampleMetadataVersion{};
      sample.metadata.sequence = slot.next_sequence;
      sample.metadata.publish_timestamp_ns = 0;
      sample.metadata.publish_clock_id = ClockId{};
      if constexpr (std::same_as<T, Tensor>) {
        sample.metadata.payload_kind = PayloadKind::kDescriptorBacked;
        sample.metadata.byte_size = 0;
      } else if constexpr (kPublishedScalar<T>) {
        sample.metadata.payload_kind = PayloadKind::kInlineTyped;
        sample.metadata.byte_size = sizeof(T);
      } else {
        sample.metadata.payload_kind = PayloadKind::kSerializedBytes;
        sample.metadata.byte_size = 0;
      }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      try {
#endif
        slot.captured.push_back(std::move(sample));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
      } catch (...) {
        return make_unexpected(Error{ErrorCode::kFailure});
      }
#endif
      ++slot.next_sequence;
      return {};
    }

    std::uint32_t max_emits{1};
    std::uint64_t next_sequence{1};
    std::vector<Sample<T>> captured;
    LoanControl* loan_control{};
    bool fail_next_loan_publication_indeterminate{};

    static void release_control(LoanControl* control) noexcept {
      if (--control->references == 0) {
        delete control;
      }
    }

    static inline const LoanBindingOps<T> kLoanOps{
        .cancel = &cancel_loan,
    };

    static inline const OutputBindingOps<T> kOps{
        .emit_sample = &publish,
        .allocate = [](void* opaque) noexcept -> expected<LoanAllocation<T>, Error> {
          if constexpr (std::default_initializable<T> && std::assignable_from<T&, T>) {
            return allocate_loan(opaque);
          } else {
            return make_unexpected(Error{ErrorCode::kNotSupported});
          }
        },
        .emit_loan = [](void* output_opaque, void* loan_opaque,
                        EmitOptions options) noexcept -> expected<void, Error> {
          if constexpr (std::default_initializable<T> && std::assignable_from<T&, T>) {
            return publish_loan(output_opaque, loan_opaque, options);
          } else {
            return make_unexpected(Error{ErrorCode::kNotSupported});
          }
        },
    };
  };

  template <PublishedPayload T>
  static inline const std::byte kSyntheticPayloadTypeToken{};

  template <PublishedPayload T>
  [[nodiscard]] expected<void, Error> install_input(Input<T>& port,
                                                    std::unique_ptr<InputSlot<T>> slot) {
    if (find_declared_input(&port) == nullptr || !valid_input_state(slot->state)) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    slot->declaration = &port;
    InputSlot<T>* const slot_address = slot.get();
    replace_input(&port, std::move(slot));
    port.binding_ = InputBinding<T>{
        .state = slot_address,
        .ops = &InputSlot<T>::kOps,
        .generation = slot_address->active_generation,
    };
    return {};
  }

  struct DeclaredInput {
    void* declaration{};
    std::uint32_t queue_depth{};
  };

  [[nodiscard]] const DeclaredInput* find_declared_input(const void* declaration) const noexcept {
    const auto found = std::find_if(declared_inputs_.begin(),
                                    declared_inputs_.end(),
                                    [declaration](const DeclaredInput& candidate) {
                                      return candidate.declaration == declaration;
                                    });
    return found == declared_inputs_.end() ? nullptr : &*found;
  }

  [[nodiscard]] static bool valid_input_state(const SyntheticInputState& state) noexcept {
    const bool selected = state.selection.state == InputState::kSelected;
    const bool reused = state.selection.state == InputState::kReused;
    const bool has_value = selected || reused;
    if (state.advertises_value != has_value) {
      return false;
    }
    if (selected) {
      return state.selection.reason == SelectionReason::kNone;
    }
    if (reused) {
      return state.selection.reason != SelectionReason::kNone &&
             state.selection.reason != SelectionReason::kNotSelectedForActivation;
    }
    if (state.selection.state == InputState::kNotSelected) {
      return state.selection.reason == SelectionReason::kNotSelectedForActivation;
    }
    return state.selection.reason != SelectionReason::kNone &&
           state.selection.reason != SelectionReason::kNotSelectedForActivation;
  }

  [[nodiscard]] static bool valid_builder_records(const OperatorSpec& spec) noexcept {
    for (const OperatorSpec::PortRecord& record : spec.ports_) {
      if (record.direction == PortDirection::kInput) {
        if (record.builder_index >= spec.input_builders_.size() ||
            spec.input_builders_[record.builder_index].queue_depth_.value_or(1U) == 0U ||
            spec.input_builders_[record.builder_index].rate_.error.code != ErrorCode::kSuccess ||
            spec.input_builders_[record.builder_index].tensor_contract_error_.code !=
                ErrorCode::kSuccess) {
          return false;
        }
        continue;
      }
      if (!record.tensor) {
        if (record.builder_index >= spec.output_builders_.size() ||
            spec.output_builders_[record.builder_index].max_emits_per_compute_ == 0 ||
            spec.output_builders_[record.builder_index].rate_.error.code != ErrorCode::kSuccess ||
            spec.output_builders_[record.builder_index].tensor_contract_error_.code !=
                ErrorCode::kSuccess) {
          return false;
        }
        continue;
      }
      if (record.builder_index >= spec.tensor_output_builders_.size()) {
        return false;
      }
      const TensorOutputSpecBuilder& builder = spec.tensor_output_builders_[record.builder_index];
      if (builder.max_emits_per_compute_ == 0 || builder.rate_.error.code != ErrorCode::kSuccess ||
          builder.tensor_contract_error_.code != ErrorCode::kSuccess ||
          (builder.has_tensor_allocation_ && (builder.tensor_allocation_.max_rank == 0 ||
                                              builder.tensor_allocation_.max_byte_span == 0))) {
        return false;
      }
    }

    return valid_source_builder_records(spec);
  }

  [[nodiscard]] static bool valid_source_builder_records(const OperatorSpec& spec) noexcept {
    for (const OperatorSpec::SourceRecord& record : spec.sources_) {
      switch (record.builder_family) {
        case OperatorSpec::SourceBuilderFamily::kSingleWitness:
          if (record.builder_index >= spec.single_source_builders_.size() ||
              spec.single_source_builders_[record.builder_index].capacity_ != 1) {
            return false;
          }
          break;
        case OperatorSpec::SourceBuilderFamily::kNotification:
          if (record.builder_index >= spec.notification_source_builders_.size()) {
            return false;
          }
          if (const NotificationSourceSpecBuilder& builder =
                  spec.notification_source_builders_[record.builder_index];
              builder.capacity_ == 0 || builder.sender_reference_capacity_ == 0) {
            return false;
          }
          break;
        case OperatorSpec::SourceBuilderFamily::kDoorbell:
          if (record.builder_index >= spec.doorbell_source_builders_.size() ||
              spec.doorbell_source_builders_[record.builder_index].capacity_ == 0) {
            return false;
          }
          break;
      }
    }
    return true;
  }

  struct DeclaredOutput {
    void* declaration{};
    std::uint32_t max_emits_per_compute{};
  };

  [[nodiscard]] const DeclaredOutput* find_declared_output(const void* declaration) const noexcept {
    const auto found = std::find_if(declared_outputs_.begin(),
                                    declared_outputs_.end(),
                                    [declaration](const DeclaredOutput& candidate) {
                                      return candidate.declaration == declaration;
                                    });
    return found == declared_outputs_.end() ? nullptr : &*found;
  }

  void replace_input(const void* declaration, std::unique_ptr<InputSlotBase> slot) {
    const auto found = std::find_if(inputs_.begin(),
                                    inputs_.end(),
                                    [declaration](const std::unique_ptr<InputSlotBase>& candidate) {
                                      return candidate->declaration == declaration;
                                    });
    if (found == inputs_.end()) {
      inputs_.push_back(std::move(slot));
    } else {
      *found = std::move(slot);
    }
  }

  void replace_output(const void* declaration, std::unique_ptr<OutputSlotBase> slot) {
    const auto found =
        std::find_if(outputs_.begin(),
                     outputs_.end(),
                     [declaration](const std::unique_ptr<OutputSlotBase>& candidate) {
                       return candidate->declaration == declaration;
                     });
    if (found == outputs_.end()) {
      outputs_.push_back(std::move(slot));
    } else {
      *found = std::move(slot);
    }
  }

  [[nodiscard]] bool has_output(const void* declaration) const noexcept {
    return std::any_of(outputs_.begin(),
                       outputs_.end(),
                       [declaration](const std::unique_ptr<OutputSlotBase>& candidate) {
                         return candidate->declaration == declaration;
                       });
  }

  template <PublishedPayload T>
  [[nodiscard]] OutputSlot<T>* find_output(const void* declaration) noexcept {
    for (const std::unique_ptr<OutputSlotBase>& candidate : outputs_) {
      if (candidate->declaration == declaration &&
          candidate->payload_type == &kSyntheticPayloadTypeToken<T>) {
        return static_cast<OutputSlot<T>*>(candidate.get());
      }
    }
    return nullptr;
  }

  template <PublishedPayload T>
  [[nodiscard]] const OutputSlot<T>* find_output(const void* declaration) const noexcept {
    for (const std::unique_ptr<OutputSlotBase>& candidate : outputs_) {
      if (candidate->declaration == declaration &&
          candidate->payload_type == &kSyntheticPayloadTypeToken<T>) {
        return static_cast<const OutputSlot<T>*>(candidate.get());
      }
    }
    return nullptr;
  }

  bool declared_{};
  std::vector<DeclaredInput> declared_inputs_;
  std::vector<DeclaredOutput> declared_outputs_;
  std::vector<std::unique_ptr<InputSlotBase>> inputs_;
  std::vector<std::unique_ptr<OutputSlotBase>> outputs_;
};

}  // namespace holoscan::detail
