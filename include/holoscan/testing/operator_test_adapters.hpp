/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <holoscan/holoscan_testing_export.h>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/testing/schema_snapshot.hpp>
#include <holoscan/testing/tensor_snapshot.hpp>

#include <holoscan/testing/operator_test_access.hpp>
#include <holoscan/testing/test_sample.hpp>

namespace holoscan::testing {
namespace detail {

// Closed builtins share one testing-library token across provider modules. Custom native types
// keep their process-local exact identity rather than relying on compatible schema metadata.
HOLOSCAN_TESTING_EXPORT const void* builtin_capture_token(PayloadIdentityKind kind,
                                                          ScalarIdentityKind scalar) noexcept;
template <PublishedPayload T>
const void* capture_type_token() noexcept {
  if constexpr (std::same_as<T, Tensor>) {
    return builtin_capture_token(PayloadIdentityKind::kTensor, ScalarIdentityKind::kBoolean);
  } else if constexpr (std::is_arithmetic_v<T>) {
    return builtin_capture_token(PayloadIdentityKind::kScalar,
                                 holoscan::detail::operator_spec_scalar_identity_kind<T>());
  } else {
    return &holoscan::detail::kOperatorSpecLocalTypeToken<T>;
  }
}

// The cold adapter boundary is shared by typed C++ and registered language bindings.
enum class CaptureKind { kValue, kSnapshot };

struct RetainRequest {
  std::size_t capacity{};
  std::vector<std::uint16_t> field_ids;
};
struct RetainPlan {
  OperatorRef owner;
  PortId output;
  std::uint64_t compute_calls{};
  std::uint64_t records{};
  std::vector<std::string> pool_ids;
};
struct RetainEvidence {
  std::shared_ptr<const RetainPlan> plan;
  RunGenerationId run{};
  std::uint64_t records{};
  bool boundary_observed{};
};
HOLOSCAN_TESTING_EXPORT std::shared_ptr<const RetainPlan> prepare_retention(
    const ExecutionPlan& plan, std::string_view operator_name, std::string_view output,
    const RetainRequest& request);
HOLOSCAN_TESTING_EXPORT expected<void, Error> validate_retained_tensor(const Tensor& tensor,
                                                                       const RetainPlan& plan,
                                                                       std::size_t field,
                                                                       OperatorRef collector);
HOLOSCAN_TESTING_EXPORT bool retained_boundary_completed(const RetainEvidence& evidence,
                                                         const RunSession& session) noexcept;

struct CaptureProgress;
HOLOSCAN_TESTING_EXPORT void capture_recorded(CaptureProgress& progress) noexcept;

struct CaptureStorage {
  std::shared_ptr<CaptureProgress> progress;
  std::optional<CaptureFailure> failure;
  RetainEvidence retained;
  [[nodiscard]] virtual CaptureKind kind() const noexcept = 0;
  virtual ~CaptureStorage() = default;
  [[nodiscard]] virtual const void* type_token() const noexcept = 0;
  // Allocate the destination before execution; transfer only to that exact matching storage.
  [[nodiscard]] virtual std::shared_ptr<CaptureStorage> empty_storage() const = 0;
  virtual void transfer_to(CaptureStorage& destination) noexcept = 0;
};

template <PublishedPayload T>
struct ValueCaptureStorage final : CaptureStorage {
  [[nodiscard]] CaptureKind kind() const noexcept override { return CaptureKind::kValue; }
  [[nodiscard]] const void* type_token() const noexcept override { return capture_type_token<T>(); }
  [[nodiscard]] std::shared_ptr<CaptureStorage> empty_storage() const override {
    return std::make_shared<ValueCaptureStorage>();
  }
  void transfer_to(CaptureStorage& destination) noexcept override {
    auto& target = static_cast<ValueCaptureStorage&>(destination);
    samples.swap(target.samples);
    static_assert(std::is_nothrow_swappable_v<decltype(failure)>);
    failure.swap(target.failure);
    target.retained = retained;
    // Keep the immutable proof unchanged: concurrent replay may read it from the mailbox.
    retained.run = 0U;
    retained.records = 0U;
    retained.boundary_observed = false;
  }
  std::vector<Sample<T>> samples;
};

template <PublishedPayload T>
struct SnapshotType;
template <>
struct SnapshotType<Tensor> {
  using type = TensorSnapshot;
};
template <PublishedPayload T>
  requires holoscan::detail::HasDescriptorTensorSchemaNativeTraits<T>
struct SnapshotType<T> {
  using type = SchemaSnapshot<T>;
};
template <PublishedPayload T>
using Snapshot = typename SnapshotType<T>::type;

template <PublishedPayload T>
struct SnapshotCaptureStorage final : CaptureStorage {
  [[nodiscard]] CaptureKind kind() const noexcept override { return CaptureKind::kSnapshot; }
  [[nodiscard]] const void* type_token() const noexcept override { return capture_type_token<T>(); }
  [[nodiscard]] std::shared_ptr<CaptureStorage> empty_storage() const override {
    return std::make_shared<SnapshotCaptureStorage>();
  }
  void transfer_to(CaptureStorage& destination) noexcept override {
    auto& target = static_cast<SnapshotCaptureStorage&>(destination);
    snapshots.swap(target.snapshots);
    static_assert(std::is_nothrow_swappable_v<decltype(failure)>);
    failure.swap(target.failure);
    target.retained = retained;
    // Keep the immutable proof unchanged: concurrent replay may read it from the mailbox.
    retained.run = 0U;
    retained.records = 0U;
    retained.boundary_observed = false;
  }
  std::vector<Snapshot<T>> snapshots;
};

struct InputPortWriter {
  virtual ~InputPortWriter() = default;
  virtual void setup(OperatorSpec& spec, std::string_view name) = 0;
  [[nodiscard]] virtual const OutputBase& output() const noexcept = 0;
  [[nodiscard]] virtual expected<void, Error> emit(std::size_t index,
                                                   ExecutionContext& context) = 0;
};

struct InputScript {
  virtual ~InputScript() = default;
  [[nodiscard]] virtual std::size_t size() const noexcept = 0;
  [[nodiscard]] virtual PayloadIdentityRecord identity() const = 0;
  [[nodiscard]] virtual std::unique_ptr<InputPortWriter> make_writer() const = 0;
  [[nodiscard]] virtual std::shared_ptr<const InputScript> appended(
      const InputScript& next) const = 0;
  [[nodiscard]] virtual const void* type_token() const noexcept = 0;
  virtual authoring::OpaqueOperatorRef build(Graph& graph,
                                             const authoring::OpaqueOperatorRef& tested,
                                             std::string_view input_name,
                                             std::string_view helper_name,
                                             std::optional<ConnectionOptions> options,
                                             std::optional<TensorOutputSpec> tensor_spec) const = 0;
  // Legacy internal adapters retain token-only behavior until they supply a native witness.
  [[nodiscard]] virtual const std::type_info* payload_type_info() const noexcept { return nullptr; }
};

struct CaptureDeclaration {
  virtual ~CaptureDeclaration() = default;
  [[nodiscard]] virtual std::optional<RetainRequest> retention_request() const { return {}; }
  [[nodiscard]] virtual std::shared_ptr<CaptureStorage> make_storage() const = 0;
  virtual authoring::OpaqueOperatorRef build(Graph& graph,
                                             const authoring::OpaqueOperatorRef& tested,
                                             std::string_view output_name,
                                             std::string_view helper_name,
                                             const std::shared_ptr<CaptureStorage>& storage,
                                             std::optional<ConnectionOptions> options) const = 0;
  [[nodiscard]] virtual PayloadIdentityRecord identity() const = 0;
  [[nodiscard]] virtual const void* type_token() const noexcept = 0;
  [[nodiscard]] virtual const std::type_info* payload_type_info() const noexcept = 0;
};

// These bridges preserve Graph identity and exact native payload type before using base ports.
HOLOSCAN_TESTING_EXPORT void connect_input(Graph& graph, const authoring::OpaqueOperatorRef& tested,
                                           std::string_view name, const OutputBase& source,
                                           const PayloadIdentityRecord& identity,
                                           const void* local_type, ConnectionOptions options);
HOLOSCAN_TESTING_EXPORT void connect_input(Graph& graph, const authoring::OpaqueOperatorRef& tested,
                                           std::string_view name, const OutputBase& source,
                                           const PayloadIdentityRecord& identity,
                                           const void* local_type, ConnectionOptions options,
                                           const std::type_info* local_type_info);
HOLOSCAN_TESTING_EXPORT void connect_output(Graph& graph,
                                            const authoring::OpaqueOperatorRef& tested,
                                            std::string_view name, const InputBase& collector,
                                            const PayloadIdentityRecord& identity,
                                            const void* local_type, ConnectionOptions options);
HOLOSCAN_TESTING_EXPORT void connect_output(Graph& graph,
                                            const authoring::OpaqueOperatorRef& tested,
                                            std::string_view name, const InputBase& collector,
                                            const PayloadIdentityRecord& identity,
                                            const void* local_type, ConnectionOptions options,
                                            const std::type_info* local_type_info);

struct SourceTensorSpec {
  TensorOutputSpec value;
  std::vector<std::int64_t> maximum_shape;
  [[nodiscard]] TensorOutputSpec view() const {
    auto result = value;
    if (!maximum_shape.empty()) {
      if (!result.bounds.has_value()) {
        throw std::logic_error{"A staged maximum shape requires Tensor bounds"};
      }
      result.bounds->max_shape = TensorShapeView{maximum_shape};
    }
    return result;
  }
};
HOLOSCAN_TESTING_EXPORT SourceTensorSpec
derive_tensor_source_spec(std::span<const Tensor* const> values);
HOLOSCAN_TESTING_EXPORT expected<void, Error> fill_staged_tensor_loan(TensorOutputLoan& loan,
                                                                      const Tensor& value,
                                                                      ExecutionContext& context);
HOLOSCAN_TESTING_EXPORT expected<void, Error> copy_staged_tensor_to_host(
    void* destination, std::size_t destination_bytes, const Tensor& value);
HOLOSCAN_TESTING_EXPORT expected<void, Error> copy_staged_tensor_to_device(
    void* destination, std::size_t destination_bytes, DeviceId device, const Tensor& value,
    ::cuda::stream_ref stream);
HOLOSCAN_TESTING_EXPORT EmitOptions staged_emit_options(SampleMetadata metadata);
HOLOSCAN_TESTING_EXPORT expected<void, Error> emit_staged_tensor(Output<Tensor>& output,
                                                                 const Tensor& value,
                                                                 SampleMetadata metadata,
                                                                 ExecutionContext& context);

template <PublishedPayload T>
std::vector<SourceTensorSpec> test_source_specs(const std::vector<TestSample<T>>& values) {
  std::vector<SourceTensorSpec> result;
  if constexpr (std::same_as<T, Tensor>) {
    if (!values.empty()) {
      std::vector<const Tensor*> tensors;
      tensors.reserve(values.size());
      for (const auto& value : values)
        tensors.push_back(&value.data);
      result.push_back(derive_tensor_source_spec(tensors));
    }
  } else if constexpr (holoscan::detail::HasDescriptorTensorSchemaNativeTraits<T>) {
    using Traits = DescriptorTensorSchemaNativeTraits<T>;
    for (std::size_t index = 0; index < Traits::tensor_count; ++index) {
      std::vector<const Tensor*> tensors;
      tensors.reserve(values.size());
      for (const auto& value : values) {
        const auto* tensor = Traits::tensor_at(value.data, index);
        if (!tensor) {
          throw std::invalid_argument{
              "operator test staged schema field missing; tensor_field_id=" +
              std::to_string(Traits::tensor_field_ids[index])};
        }
        tensors.push_back(tensor);
      }
      if (!tensors.empty())
        result.push_back(derive_tensor_source_spec(tensors));
    }
  }
  return result;
}

// The compiled adapter owns Core read leases without installing private Tensor headers.
HOLOSCAN_TESTING_EXPORT std::shared_ptr<const void> acquire_staged_tensor_leases(
    std::span<const Tensor* const> tensors);

template <PublishedPayload T>
std::shared_ptr<const void> staged_tensor_leases(const std::vector<TestSample<T>>& values) {
  std::vector<const Tensor*> tensors;
  if constexpr (std::same_as<T, Tensor>) {
    tensors.reserve(values.size());
    for (const auto& value : values)
      tensors.push_back(&value.data);
  } else if constexpr (holoscan::detail::HasDescriptorTensorSchemaNativeTraits<T>) {
    using Traits = DescriptorTensorSchemaNativeTraits<T>;
    for (const auto& value : values) {
      for (std::size_t index = 0; index < Traits::tensor_count; ++index) {
        if (const auto* tensor = Traits::tensor_at(value.data, index))
          tensors.push_back(tensor);
      }
    }
  }
  return tensors.empty() ? nullptr : acquire_staged_tensor_leases(tensors);
}

template <PublishedPayload T>
void configure_test_source(OperatorSpec& spec, Output<T>& output, std::string_view name,
                           const std::vector<TestSample<T>>& values, std::uint32_t maximum_emits) {
  auto& configured = spec.output(output, name).max_emits_per_compute(maximum_emits);
  const auto specs = test_source_specs(values);
  if constexpr (std::same_as<T, Tensor>) {
    if (!specs.empty())
      configured.produces_tensor(specs.front().view());
  } else if constexpr (holoscan::detail::HasDescriptorTensorSchemaNativeTraits<T>) {
    using Traits = DescriptorTensorSchemaNativeTraits<T>;
    for (std::size_t index = 0; index < specs.size(); ++index) {
      if constexpr (Traits::tensor_count == 1U) {
        configured.produces_tensor(specs[index].view());
      } else {
        configured
            .tensor_field(TensorFieldSelector<T>{Traits::tensor_field_ids[index],
                                                 static_cast<std::uint16_t>(index)})
            .produces_tensor(specs[index].view());
      }
    }
  }
}

template <PublishedPayload T>
  requires holoscan::detail::HasDescriptorTensorSchemaNativeTraits<T>
expected<void, Error> emit_staged_schema_bundle(Output<T>& output, const T& value,
                                                SampleMetadata metadata,
                                                ExecutionContext& context) {
  using Traits = DescriptorTensorSchemaNativeTraits<T>;
  std::array<TensorFieldLoanRequest<T>, Traits::tensor_count> requests;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto* tensor = Traits::tensor_at(value, i);
    if (!tensor)
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "operator test schema Tensor missing"});
    requests[i] = {.field = {Traits::tensor_field_ids[i], static_cast<std::uint16_t>(i)},
                   .tensor = {.shape = tensor->shape_span(), .dtype = tensor->dtype()}};
  }
  auto bundle = output.allocate_tensors(requests);
  if (!bundle)
    return make_unexpected(std::move(bundle).error());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto field = requests[i].field;
    const auto& tensor = *Traits::tensor_at(value, i);
    expected<void, Error> copied;
    const auto kind = bundle->memory_kind(field);
    if (kind == MemoryKind::kHost || kind == MemoryKind::kPinnedHost) {
      copied = bundle->write_host_and_commit(field, [&](const auto& write) {
        return copy_staged_tensor_to_host(write.data(), write.byte_size(), tensor);
      });
    } else {
      const auto stream = context.cuda_stream();
      copied = bundle->write_and_commit(field, stream, [&](const auto& write) {
        return copy_staged_tensor_to_device(
            write.data(), write.byte_size(), bundle->device_id(field), tensor, stream);
      });
    }
    if (!copied) {
      auto error = std::move(copied).error();
      error.message += "; tensor_field_id=" + std::to_string(field.field_id);
      return make_unexpected(std::move(error));
    }
  }
  // No field commit publishes. Submit the complete descriptor exactly once, without retry.
  return output.emit_tensors(std::move(*bundle), value, staged_emit_options(metadata));
}

template <PublishedPayload T>
expected<void, Error> emit_test_value(Output<T>& output, const T& value, SampleMetadata metadata,
                                      ExecutionContext& context) {
  if constexpr (std::same_as<T, Tensor>) {
    return emit_staged_tensor(output, value, metadata, context);
  } else if constexpr (holoscan::detail::HasDescriptorTensorSchemaNativeTraits<T>) {
    using Traits = DescriptorTensorSchemaNativeTraits<T>;
    if constexpr (Traits::tensor_count == 1U) {
      const auto* tensor = Traits::tensor_at(value, 0U);
      if (!tensor)
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "operator test schema Tensor missing"});
      auto loan = output.allocate_tensor(
          TensorLoanRequest{.shape = tensor->shape_span(), .dtype = tensor->dtype()});
      if (!loan)
        return make_unexpected(std::move(loan).error());
      auto copied = fill_staged_tensor_loan(*loan, *tensor, context);
      if (!copied)
        return copied;
      return output.emit_tensor(std::move(*loan), value, staged_emit_options(metadata));
    } else {
      return emit_staged_schema_bundle(output, value, metadata, context);
    }
  } else {
    return output.emit(Sample<T>{value, metadata});
  }
}

// One qualified scalar burst per independent input. Runtime owns admission and EOS.
template <PublishedPayload T>
class ValueSource final : public Operator<> {
 public:
  ValueSource(std::shared_ptr<const std::vector<TestSample<T>>> values,
              std::shared_ptr<const void> read_leases)
      : values_(std::move(values)), read_leases_(std::move(read_leases)) {}
  void setup(OperatorSpec& spec) override {
    configure_test_source(spec,
                          output,
                          "output",
                          *values_,
                          static_cast<std::uint32_t>(std::max<std::size_t>(1U, values_->size())));
  }
  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}})
        .complete_after(ComputeCalls{.limit = values_->empty() ? 0U : 1U});
    return result;
  }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    for (const auto& value : *values_) {
      auto metadata = value.metadata;
      if (value.capture_coordinate) {
        metadata.capture_timestamp_ns = value.capture_coordinate->count();
        metadata.source_clock_id = context.clock().id();
      }
      auto emitted = emit_test_value(output, value.data, metadata, context);
      if (!emitted) {
        return make_unexpected(std::move(emitted).error());
      }
    }
    return {};
  }
  Output<T> output;

 private:
  std::shared_ptr<const std::vector<TestSample<T>>> values_;
  std::shared_ptr<const void> read_leases_;
};

template <PublishedPayload T>
struct ValuePortWriter final : InputPortWriter {
  ValuePortWriter(std::vector<TestSample<T>> values, std::shared_ptr<const void> read_leases)
      : values_(std::move(values)), read_leases_(std::move(read_leases)) {}
  void setup(OperatorSpec& spec, std::string_view name) override {
    configure_test_source(spec, output_, name, values_, 1U);
  }
  [[nodiscard]] const OutputBase& output() const noexcept override { return output_; }
  [[nodiscard]] expected<void, Error> emit(std::size_t index, ExecutionContext& context) override {
    if (index >= values_.size()) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "operator test frame index invalid"});
    }
    const auto& value = values_[index];
    auto metadata = value.metadata;
    if (value.capture_coordinate) {
      metadata.capture_timestamp_ns = value.capture_coordinate->count();
      metadata.source_clock_id = context.clock().id();
    }
    return emit_test_value(output_, value.data, metadata, context);
  }
  Output<T> output_;

 private:
  std::vector<TestSample<T>> values_;
  std::shared_ptr<const void> read_leases_;
};

template <PublishedPayload T>
struct ValueInputScript final : InputScript {
  explicit ValueInputScript(const std::vector<TestSample<T>>& values) : values_(values) {
    // Always cross the owning-copy boundary, even when the caller moved a schema. A caller may
    // still hold aliases to its metadata or Tensor wrappers. Generated schema copies detach
    // those objects before the leases below bind the staged backing.
    if (values_.size() > UINT32_MAX) {
      throw std::invalid_argument{"operator test input batch exceeds uint32 capacity"};
    }
    // Validate the complete candidate before it can replace pending staging.
    static_cast<void>(test_source_specs(values_));
    read_leases_ = staged_tensor_leases(values_);
  }
  [[nodiscard]] std::size_t size() const noexcept override { return values_.size(); }
  [[nodiscard]] PayloadIdentityRecord identity() const override {
    return holoscan::detail::operator_spec_payload_identity<T>();
  }
  [[nodiscard]] std::unique_ptr<InputPortWriter> make_writer() const override {
    return std::make_unique<ValuePortWriter<T>>(values_, read_leases_);
  }
  [[nodiscard]] const void* type_token() const noexcept override { return capture_type_token<T>(); }
  [[nodiscard]] const std::type_info* payload_type_info() const noexcept override {
    return &typeid(T);
  }
  [[nodiscard]] std::shared_ptr<const InputScript> appended(
      const InputScript& next) const override {
    const auto* incoming = dynamic_cast<const ValueInputScript*>(&next);
    if (incoming == nullptr) {
      throw std::invalid_argument{"operator test input adapter or payload type changed"};
    }
    auto values = values_;
    values.insert(values.end(), incoming->values_.begin(), incoming->values_.end());
    return std::make_shared<ValueInputScript>(std::move(values));
  }
  authoring::OpaqueOperatorRef build(Graph& graph, const authoring::OpaqueOperatorRef& tested,
                                     std::string_view input_name, std::string_view helper_name,
                                     std::optional<ConnectionOptions> options,
                                     std::optional<TensorOutputSpec> tensor_spec) const override {
    auto values = std::make_shared<const std::vector<TestSample<T>>>(values_);
    const auto source = graph.op<ValueSource<T>>(helper_name, std::move(values), read_leases_);
    if (tensor_spec)
      graph.configure(source->output).produces_tensor(*tensor_spec);
    connect_input(
        graph,
        tested,
        input_name,
        source->output,
        holoscan::detail::operator_spec_payload_identity<T>(),
        type_token(),
        options.value_or(ConnectionOptions{
            .queue_depth = static_cast<std::uint32_t>(std::max<std::size_t>(1U, values_.size()))}),
        &typeid(T));
    return holoscan::detail::OperatorTestingAccess::erase(source);
  }

 private:
  std::vector<TestSample<T>> values_;
  std::shared_ptr<const void> read_leases_;
};

// Each replayed Runtime instance receives private storage. Only the first start
// claims the handoff destination used by the fixture's initial result.
template <typename Storage>
struct CaptureMailbox {
  explicit CaptureMailbox(std::shared_ptr<Storage> initial) : initial(std::move(initial)) {}
  std::shared_ptr<Storage> initial;
  std::atomic<bool> claimed{};
};

template <PublishedPayload T>
class ValueCollector final : public Operator<> {
 public:
  ValueCollector(std::shared_ptr<CaptureMailbox<ValueCaptureStorage<T>>> mailbox,
                 std::size_t capacity)
      : mailbox_(std::move(mailbox)), capacity_(capacity) {}
  void start() override {
    if (storage_)
      return;
    if (!mailbox_->claimed.exchange(true)) {
      storage_ = mailbox_->initial;
    } else {
      storage_ = std::make_shared<ValueCaptureStorage<T>>();
      storage_->samples.reserve(capacity_);
    }
  }
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    auto received = input.receive();
    if (!received) {
      return make_unexpected(std::move(received).error());
    }
    auto& failure = storage_->failure.emplace(CaptureFailure{.metadata = received->metadata});
    if (storage_->samples.size() == capacity_) {
      failure.error =
          Error{ErrorCode::kResourceExhausted, "operator test capture capacity exhausted"};
      return make_unexpected(failure.error);
    }
    // Do not retain the incoming finite Sample read claim.
    storage_->samples.emplace_back(received->data, received->metadata);
    storage_->failure.reset();
    if (storage_->progress)
      capture_recorded(*storage_->progress);
    return {};
  }
  Input<T> input;

 private:
  std::shared_ptr<CaptureMailbox<ValueCaptureStorage<T>>> mailbox_;
  std::shared_ptr<ValueCaptureStorage<T>> storage_;
  std::size_t capacity_;
};

template <PublishedPayload T>
struct ValueCaptureDeclaration final : CaptureDeclaration {
  explicit ValueCaptureDeclaration(std::size_t capacity) : capacity_(capacity) {
    if constexpr (std::same_as<T, Tensor> ||
                  holoscan::detail::HasDescriptorTensorSchemaNativeTraits<T>) {
      throw std::invalid_argument{
          "Tensor-bearing capture requires snapshot or explicit retain policy"};
    }
    if (capacity == 0U || capacity > UINT32_MAX) {
      throw std::invalid_argument{"operator test capture capacity must fit positive uint32"};
    }
  }
  [[nodiscard]] PayloadIdentityRecord identity() const override {
    return holoscan::detail::operator_spec_payload_identity<T>();
  }
  [[nodiscard]] const void* type_token() const noexcept override { return capture_type_token<T>(); }
  [[nodiscard]] const std::type_info* payload_type_info() const noexcept override {
    return &typeid(T);
  }
  [[nodiscard]] std::shared_ptr<CaptureStorage> make_storage() const override {
    auto result = std::make_shared<ValueCaptureStorage<T>>();
    result->samples.reserve(capacity_);
    return result;
  }
  authoring::OpaqueOperatorRef build(Graph& graph, const authoring::OpaqueOperatorRef& tested,
                                     std::string_view output_name, std::string_view helper_name,
                                     const std::shared_ptr<CaptureStorage>& storage,
                                     std::optional<ConnectionOptions> options) const override {
    const auto collector =
        graph.op<ValueCollector<T>>(helper_name,
                                    std::make_shared<CaptureMailbox<ValueCaptureStorage<T>>>(
                                        std::static_pointer_cast<ValueCaptureStorage<T>>(storage)),
                                    capacity_);
    connect_output(
        graph,
        tested,
        output_name,
        collector->input,
        holoscan::detail::operator_spec_payload_identity<T>(),
        storage->type_token(),
        options.value_or(ConnectionOptions{.queue_depth = static_cast<std::uint32_t>(capacity_)}),
        &typeid(T));
    return holoscan::detail::OperatorTestingAccess::erase(collector);
  }
  std::size_t capacity_;
};

template <PublishedPayload T>
class RetainCollector final : public Operator<> {
 public:
  RetainCollector(std::shared_ptr<CaptureMailbox<ValueCaptureStorage<T>>> mailbox,
                  std::size_t capacity)
      : mailbox_(std::move(mailbox)), capacity_(capacity) {}
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }
  void start() override {
    if (storage_)
      return;
    if (!mailbox_->claimed.exchange(true)) {
      storage_ = mailbox_->initial;
    } else {
      storage_ = std::make_shared<ValueCaptureStorage<T>>();
      storage_->samples.reserve(capacity_);
      storage_->retained.plan = mailbox_->initial->retained.plan;
    }
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    auto received = input.receive();
    if (!received)
      return make_unexpected(std::move(received).error());
    auto& failure = storage_->failure.emplace(CaptureFailure{.metadata = received->metadata});
    auto& evidence = storage_->retained;
    if (!evidence.plan || evidence.records >= evidence.plan->records ||
        storage_->samples.size() >= capacity_) {
      failure.error = Error{ErrorCode::kResourceExhausted, "operator test Retain bound exceeded"};
      return make_unexpected(failure.error);
    }
    auto validate = [&](const Tensor& tensor, std::size_t index) {
      return validate_retained_tensor(tensor, *evidence.plan, index, context.self());
    };
    expected<void, Error> valid;
    if constexpr (std::same_as<T, Tensor>) {
      valid = validate(received->data, 0U);
    } else {
      using Traits = DescriptorTensorSchemaNativeTraits<T>;
      for (std::size_t index = 0U; index < Traits::tensor_count; ++index) {
        const auto* tensor = Traits::tensor_at(received->data, index);
        if (tensor)
          valid = validate(*tensor, index);
        if (!valid) {
          valid.error().message +=
              "; tensor_field_id=" + std::to_string(Traits::tensor_field_ids[index]);
          break;
        }
      }
    }
    if (!valid) {
      failure.error = std::move(valid).error();
      return make_unexpected(failure.error);
    }
    // The owning native copy retains backing, never the incoming finite Sample claim.
    storage_->samples.emplace_back(received->data, received->metadata);
    storage_->failure.reset();
    if (storage_->progress)
      capture_recorded(*storage_->progress);
    evidence.run = context.self().run_generation;
    ++evidence.records;
    if (evidence.records == evidence.plan->records) {
      evidence.boundary_observed = true;
      context.request_stop();
    }
    return {};
  }
  Input<T> input;

 private:
  std::shared_ptr<CaptureMailbox<ValueCaptureStorage<T>>> mailbox_;
  std::shared_ptr<ValueCaptureStorage<T>> storage_;
  std::size_t capacity_;
};

template <PublishedPayload T>
struct RetainCaptureDeclaration final : CaptureDeclaration {
  explicit RetainCaptureDeclaration(std::size_t capacity) : capacity_(capacity) {
    static_assert(std::same_as<T, Tensor> ||
                  holoscan::detail::HasDescriptorTensorSchemaNativeTraits<T>);
    if (capacity == 0U || capacity > UINT32_MAX)
      throw std::invalid_argument{"operator test Retain capacity must fit positive uint32"};
  }
  std::optional<RetainRequest> retention_request() const override {
    RetainRequest request{.capacity = capacity_};
    if constexpr (!std::same_as<T, Tensor>) {
      const auto& ids = DescriptorTensorSchemaNativeTraits<T>::tensor_field_ids;
      request.field_ids.assign(ids.begin(), ids.end());
    }
    return request;
  }
  [[nodiscard]] PayloadIdentityRecord identity() const override {
    return holoscan::detail::operator_spec_payload_identity<T>();
  }
  [[nodiscard]] const void* type_token() const noexcept override { return capture_type_token<T>(); }
  [[nodiscard]] const std::type_info* payload_type_info() const noexcept override {
    return &typeid(T);
  }
  std::shared_ptr<CaptureStorage> make_storage() const override {
    auto result = std::make_shared<ValueCaptureStorage<T>>();
    result->samples.reserve(capacity_);
    return result;
  }
  authoring::OpaqueOperatorRef build(Graph& graph, const authoring::OpaqueOperatorRef& tested,
                                     std::string_view name, std::string_view helper_name,
                                     const std::shared_ptr<CaptureStorage>& storage,
                                     std::optional<ConnectionOptions> options) const override {
    const auto collector =
        graph.op<RetainCollector<T>>(helper_name,
                                     std::make_shared<CaptureMailbox<ValueCaptureStorage<T>>>(
                                         std::static_pointer_cast<ValueCaptureStorage<T>>(storage)),
                                     capacity_);
    connect_output(
        graph,
        tested,
        name,
        collector->input,
        holoscan::detail::operator_spec_payload_identity<T>(),
        storage->type_token(),
        options.value_or(ConnectionOptions{.queue_depth = static_cast<std::uint32_t>(capacity_)}),
        &typeid(T));
    return holoscan::detail::OperatorTestingAccess::erase(collector);
  }
  std::size_t capacity_;
};

template <PublishedPayload T>
class SnapshotCollector final : public Operator<> {
 public:
  SnapshotCollector(std::shared_ptr<CaptureMailbox<SnapshotCaptureStorage<T>>> mailbox,
                    std::size_t capacity, std::size_t maximum_bytes)
      : mailbox_(std::move(mailbox)), capacity_(capacity), maximum_bytes_(maximum_bytes) {}
  void start() override {
    if (storage_)
      return;
    if (!mailbox_->claimed.exchange(true)) {
      storage_ = mailbox_->initial;
    } else {
      storage_ = std::make_shared<SnapshotCaptureStorage<T>>();
      storage_->snapshots.reserve(capacity_);
    }
  }
  void setup(OperatorSpec& spec) override { spec.input(input, "input"); }
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override {
    auto received = input.receive();
    if (!received)
      return make_unexpected(std::move(received).error());
    auto& failure = storage_->failure.emplace(CaptureFailure{.metadata = received->metadata});
    if (storage_->snapshots.size() == capacity_) {
      failure.error =
          Error{ErrorCode::kResourceExhausted, "operator test snapshot capacity exhausted"};
      return make_unexpected(failure.error);
    }
    auto snapshot = [&]() -> expected<Snapshot<T>, Error> {
      if constexpr (std::same_as<T, Tensor>) {
        cudaStream_t stream = nullptr;
        if (received->data.data() && (received->data.device().device_type == kDLCUDA ||
                                      received->data.device().device_type == kDLCUDAManaged)) {
          stream = context.cuda_stream().get();
        }
        return capture_tensor_snapshot(received->data, received->metadata, maximum_bytes_, stream);
      } else {
        return SchemaSnapshotAccess<T>::copy(
            received->data, received->metadata, maximum_bytes_, context);
      }
    }();
    if (!snapshot) {
      failure.error = std::move(snapshot).error();
      return make_unexpected(failure.error);
    }
    storage_->snapshots.push_back(std::move(*snapshot));
    storage_->failure.reset();
    if (storage_->progress)
      capture_recorded(*storage_->progress);
    return {};
  }
  Input<T> input;

 private:
  std::shared_ptr<CaptureMailbox<SnapshotCaptureStorage<T>>> mailbox_;
  std::shared_ptr<SnapshotCaptureStorage<T>> storage_;
  std::size_t capacity_;
  std::size_t maximum_bytes_;
};

template <PublishedPayload T>
struct SnapshotCaptureDeclaration final : CaptureDeclaration {
  SnapshotCaptureDeclaration(std::size_t capacity, std::size_t maximum_bytes)
      : capacity_(capacity), maximum_bytes_(maximum_bytes) {
    if (capacity == 0U || capacity > UINT32_MAX || maximum_bytes == 0U) {
      throw std::invalid_argument{"operator test snapshot bounds must be positive and fit uint32"};
    }
  }
  [[nodiscard]] PayloadIdentityRecord identity() const override {
    return holoscan::detail::operator_spec_payload_identity<T>();
  }
  [[nodiscard]] const void* type_token() const noexcept override { return capture_type_token<T>(); }
  [[nodiscard]] const std::type_info* payload_type_info() const noexcept override {
    return &typeid(T);
  }
  [[nodiscard]] std::shared_ptr<CaptureStorage> make_storage() const override {
    auto result = std::make_shared<SnapshotCaptureStorage<T>>();
    result->snapshots.reserve(capacity_);
    return result;
  }
  authoring::OpaqueOperatorRef build(Graph& graph, const authoring::OpaqueOperatorRef& tested,
                                     std::string_view name, std::string_view helper_name,
                                     const std::shared_ptr<CaptureStorage>& storage,
                                     std::optional<ConnectionOptions> options) const override {
    const auto collector = graph.op<SnapshotCollector<T>>(
        helper_name,
        std::make_shared<CaptureMailbox<SnapshotCaptureStorage<T>>>(
            std::static_pointer_cast<SnapshotCaptureStorage<T>>(storage)),
        capacity_,
        maximum_bytes_);
    connect_output(
        graph,
        tested,
        name,
        collector->input,
        holoscan::detail::operator_spec_payload_identity<T>(),
        storage->type_token(),
        options.value_or(ConnectionOptions{.queue_depth = static_cast<std::uint32_t>(capacity_)}),
        &typeid(T));
    return holoscan::detail::OperatorTestingAccess::erase(collector);
  }
  std::size_t capacity_;
  std::size_t maximum_bytes_;
};

struct HelperPort {
  authoring::OpaqueOperatorRef op;
  std::string port;
};
using HelperPorts = std::map<std::string, HelperPort, std::less<>>;
struct TestResultState;
struct OperatorTestState;
}  // namespace detail
}  // namespace holoscan::testing
