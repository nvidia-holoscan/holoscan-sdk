// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "../core/module_name.hpp"
#include "registered_adapters.hpp"

namespace holoscan::testing::detail {

// Optional testing code alone may create the erased native adapter handles.
struct RegisteredAdapterAccess {
  static StagedInput input(std::shared_ptr<const InputScript> script) {
    return StagedInput{std::move(script)};
  }
  static CaptureRequest capture(std::shared_ptr<const CaptureDeclaration> declaration) {
    return CaptureRequest{std::move(declaration)};
  }
  static const CaptureStorage& storage(const TestResult& result, std::string_view name) {
    return result.capture_storage(name);
  }
};

}  // namespace holoscan::testing::detail

namespace holoscan::testing::python {
namespace {

using authoring::ForeignObjectSample;
using authoring::ForeignSchemaSample;
using authoring::ForeignSchemaValue;
using authoring::OpaqueOperatorRef;
using authoring::OpaquePortRef;
using authoring::RegisteredPayloadType;

template <typename T>
T required(expected<T, Error> value) {
  if (!value)
    throw RuntimeError{std::move(value).error()};
  return std::move(*value);
}

void required(expected<void, Error> value) {
  if (!value)
    throw RuntimeError{std::move(value).error()};
}

struct Registration {
  std::shared_ptr<const BindingOwner> owner;
  const RegisteredPayloadType* row;
  GilSafeObject schema_type;
};

template <typename Sample>
struct Staged {
  Sample sample;
  std::optional<std::int64_t> timestamp;
};

const Tensor* field_tensor(const ForeignSchemaValue& value, std::string_view name) {
  const auto* field = value.field(name);
  return field != nullptr ? field->get<Tensor>() : nullptr;
}

template <typename Sample>
struct InputBatch {
  std::shared_ptr<const Registration> registration;
  std::vector<Staged<Sample>> values;
  std::vector<detail::SourceTensorSpec> specs;
  std::shared_ptr<const void> leases;

  InputBatch(std::shared_ptr<const Registration> row, std::vector<Staged<Sample>> data)
      : registration(std::move(row)), values(std::move(data)) {
    if (values.size() > UINT32_MAX)
      throw std::invalid_argument("input batch exceeds uint32");
    if constexpr (std::same_as<Sample, ForeignSchemaSample>) {
      for (const auto& value : values)
        required(registration->row->validate_schema(value.sample.data));
      std::vector<const Tensor*> all;
      for (const auto& field : registration->row->descriptor_tensor_fields) {
        std::vector<const Tensor*> tensors;
        for (const auto& value : values) {
          const auto* tensor = field_tensor(value.sample.data, field.name);
          if (!tensor)
            throw std::invalid_argument("staged schema Tensor field missing: " +
                                        std::string{field.name});
          tensors.push_back(tensor);
          all.push_back(tensor);
        }
        if (!tensors.empty())
          specs.push_back(detail::derive_tensor_source_spec(tensors));
      }
      if (!all.empty())
        leases = detail::acquire_staged_tensor_leases(all);
    }
  }
};

template <typename Sample>
class PortWriter final : public detail::InputPortWriter {
 public:
  explicit PortWriter(std::shared_ptr<const InputBatch<Sample>> batch, std::uint32_t maximum = 1U)
      : batch_(std::move(batch)), maximum_(maximum) {}
  void setup(OperatorSpec& spec, std::string_view name) override {
    const auto& row = *batch_->registration->row;
    output_ = required(row.bind_spec_output(spec, name, maximum_));
    output_base_ = &row.output_declaration(output_);
    for (std::size_t i = 0; i < batch_->specs.size(); ++i) {
      auto spec = batch_->specs[i].view();
      if (!spec.bounds || !spec.bounds->max_byte_span)
        throw std::logic_error("staged schema Tensor source has no derived byte bound");
      const auto bytes = *spec.bounds->max_byte_span;
      // This registry callback authors an explicit capacity expression. The native
      // builder rejects simultaneous literal bounds, so carry the derived bound
      // through that one canonical source of capacity authority.
      spec.bounds.reset();
      required(row.schema_tensor_output_produces(
          output_, row.descriptor_tensor_fields[i].name, spec, tensor_byte_span(bytes)));
    }
  }
  const OutputBase& output() const noexcept override { return *output_base_; }
  expected<void, Error> emit(std::size_t index, ExecutionContext& context) override {
    if (index >= batch_->values.size())
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    const auto& value = batch_->values[index];
    auto metadata = value.sample.metadata;
    if (value.timestamp) {
      metadata.capture_timestamp_ns = *value.timestamp;
      metadata.source_clock_id = context.clock().id();
    }
    const auto options = detail::staged_emit_options(metadata);
    const auto& row = *batch_->registration->row;
    if constexpr (std::same_as<Sample, ForeignObjectSample>) {
      return row.emit_object(output_, value.sample.data, options);
    } else {
      if (row.descriptor_tensor_fields.empty())
        return row.emit_schema(output_, value.sample.data, options);
      return emit_tensor_schema(row, value.sample.data, options, context);
    }
  }

 private:
  expected<void, Error> emit_tensor_schema(const RegisteredPayloadType& row,
                                           const ForeignSchemaValue& data, EmitOptions options,
                                           ExecutionContext& context) {
    std::vector<authoring::ForeignTensorFieldLoanRequest> requests;
    requests.reserve(row.descriptor_tensor_fields.size());
    for (const auto& field : row.descriptor_tensor_fields) {
      const auto* tensor = field_tensor(data, field.name);
      requests.push_back({field.name, {.shape = tensor->shape_span(), .dtype = tensor->dtype()}});
    }
    auto bundle = row.allocate_schema_tensors(output_, requests);
    if (!bundle)
      return make_unexpected(std::move(bundle).error());
    for (std::size_t i = 0; i < requests.size(); ++i) {
      const auto& tensor = *field_tensor(data, requests[i].field_name);
      const auto kind = bundle->memory_kind(i);
      const bool host = kind == MemoryKind::kHost || kind == MemoryKind::kPinnedHost;
      auto write = host ? bundle->write_host(i) : bundle->write(i, context.cuda_stream());
      if (!write)
        return make_unexpected(std::move(write).error());
      auto copied =
          host ? detail::copy_staged_tensor_to_host(write->data(), write->byte_size(), tensor)
               : detail::copy_staged_tensor_to_device(write->data(),
                                                      write->byte_size(),
                                                      bundle->device_id(i),
                                                      tensor,
                                                      context.cuda_stream());
      if (!copied)
        return copied;
      auto committed = std::move(*write).commit();
      if (!committed)
        return committed;
    }
    return row.emit_schema_tensors(output_, std::move(*bundle), data, options);
  }
  std::shared_ptr<const InputBatch<Sample>> batch_;
  std::uint32_t maximum_;
  OpaquePortRef output_;
  const OutputBase* output_base_{};
};

template <typename Sample>
class Source final : public Operator<> {
 public:
  explicit Source(std::shared_ptr<const InputBatch<Sample>> batch)
      : batch_(std::move(batch)),
        writer_(batch_,
                static_cast<std::uint32_t>(std::max<std::size_t>(1U, batch_->values.size()))) {}
  void setup(OperatorSpec& spec) override { writer_.setup(spec, "output"); }
  Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}})
        .complete_after(ComputeCalls{.limit = batch_->values.empty() ? 0U : 1U});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    for (std::size_t i = 0; i < batch_->values.size(); ++i) {
      auto emitted = writer_.emit(i, context);
      if (!emitted)
        return emitted;
    }
    return {};
  }

 private:
  std::shared_ptr<const InputBatch<Sample>> batch_;
  PortWriter<Sample> writer_;
};

template <typename Sample>
class InputScript final : public detail::InputScript {
 public:
  explicit InputScript(std::shared_ptr<const InputBatch<Sample>> batch)
      : batch_(std::move(batch)) {}
  std::size_t size() const noexcept override { return batch_->values.size(); }
  PayloadIdentityRecord identity() const override { return batch_->registration->row->identity; }
  const void* type_token() const noexcept override {
    return batch_->registration->row->native_type_token;
  }
  const std::type_info* payload_type_info() const noexcept override {
    return batch_->registration->row->native_type_info;
  }
  std::unique_ptr<detail::InputPortWriter> make_writer() const override {
    return std::make_unique<PortWriter<Sample>>(batch_);
  }
  std::shared_ptr<const detail::InputScript> appended(
      const detail::InputScript& next) const override {
    const auto* incoming = dynamic_cast<const InputScript*>(&next);
    if (!incoming || incoming->batch_->registration->row != batch_->registration->row)
      throw std::invalid_argument("registered input adapter changed");
    auto values = batch_->values;
    values.insert(values.end(), incoming->batch_->values.begin(), incoming->batch_->values.end());
    return std::make_shared<InputScript>(
        std::make_shared<InputBatch<Sample>>(batch_->registration, std::move(values)));
  }
  OpaqueOperatorRef build(Graph& graph, const OpaqueOperatorRef& tested, std::string_view name,
                          std::string_view helper, std::optional<ConnectionOptions> options,
                          std::optional<TensorOutputSpec> tensor_spec) const override {
    const auto source =
        holoscan::detail::OperatorTestingAccess::erase(graph.op<Source<Sample>>(helper, batch_));
    const auto& row = *batch_->registration->row;
    if (tensor_spec)
      holoscan::detail::OperatorTestingAccess::output_tensor_spec(source, "output", *tensor_spec);
    row.add_flow(graph,
                 source.port("output"),
                 tested.port(name),
                 options.value_or(ConnectionOptions{.queue_depth = static_cast<std::uint32_t>(
                                                        std::max<std::size_t>(1U, size()))}));
    return source;
  }

 private:
  std::shared_ptr<const InputBatch<Sample>> batch_;
};

template <typename Sample>
struct Storage final : detail::CaptureStorage {
  std::shared_ptr<const Registration> registration;
  CapturePolicy policy;
  std::vector<Sample> samples;
  std::vector<RegisteredSnapshot> snapshots;
  Storage(std::shared_ptr<const Registration> row, CapturePolicy capture)
      : registration(std::move(row)), policy(capture) {}
  detail::CaptureKind kind() const noexcept override {
    return policy == CapturePolicy::kSnapshot ? detail::CaptureKind::kSnapshot
                                              : detail::CaptureKind::kValue;
  }
  const void* type_token() const noexcept override { return registration->row->native_type_token; }
  std::shared_ptr<detail::CaptureStorage> empty_storage() const override {
    auto result = std::make_shared<Storage>(registration, policy);
    result->retained.plan = retained.plan;
    return result;
  }
  void transfer_to(detail::CaptureStorage& destination) noexcept override {
    auto& target = static_cast<Storage&>(destination);
    samples.swap(target.samples);
    snapshots.swap(target.snapshots);
    failure.swap(target.failure);
    target.retained = retained;
    retained.run = 0U;
    retained.records = 0U;
    retained.boundary_observed = false;
  }
};

template <typename Sample>
class Collector final : public Operator<> {
 public:
  Collector(std::shared_ptr<detail::CaptureMailbox<Storage<Sample>>> mailbox, std::size_t capacity,
            std::size_t bytes)
      : mailbox_(std::move(mailbox)), capacity_(capacity), maximum_bytes_(bytes) {}
  void setup(OperatorSpec& spec) override {
    input_ =
        required(mailbox_->initial->registration->row->bind_spec_input(spec, "input", {}, false));
  }
  void start() override {
    if (storage_)
      return;
    storage_ = !mailbox_->claimed.exchange(true)
                   ? mailbox_->initial
                   : std::static_pointer_cast<Storage<Sample>>(mailbox_->initial->empty_storage());
    storage_->samples.reserve(capacity_);
    storage_->snapshots.reserve(capacity_);
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    const auto& row = *storage_->registration->row;
    auto received = [&] {
      if constexpr (std::same_as<Sample, ForeignObjectSample>)
        return row.receive_object(input_);
      else
        return row.receive_schema(input_);
    }();
    if (!received)
      return make_unexpected(std::move(received).error());
    auto& failure = storage_->failure.emplace(CaptureFailure{.metadata = received->metadata});
    if (storage_->samples.size() + storage_->snapshots.size() == capacity_) {
      failure.error =
          Error{ErrorCode::kResourceExhausted, "operator test capture capacity exhausted"};
      return make_unexpected(failure.error);
    }
    auto stored = store_sample(std::move(*received), context);
    if (!stored) {
      failure.error = std::move(stored).error();
      return make_unexpected(failure.error);
    }
    storage_->failure.reset();
    if (storage_->progress)
      detail::capture_recorded(*storage_->progress);
    return {};
  }

 private:
  expected<void, Error> snapshot_schema(const ForeignSchemaSample& sample,
                                        ExecutionContext& context) {
    const auto& row = *storage_->registration->row;
    auto metadata = row.clone_schema_metadata(sample.data);
    if (!metadata)
      return make_unexpected(std::move(metadata).error());
    RegisteredSnapshot snapshot{storage_->registration->owner,
                                storage_->registration->schema_type,
                                std::move(*metadata),
                                sample.metadata,
                                {}};
    snapshot.tensors.reserve(row.descriptor_tensor_fields.size());
    auto remaining = maximum_bytes_;
    for (const auto& field : row.descriptor_tensor_fields) {
      const auto* tensor = field_tensor(sample.data, field.name);
      if (tensor == nullptr) {
        snapshot.tensors.emplace_back(field.name, std::nullopt);
        continue;
      }
      cudaStream_t stream = nullptr;
      if (tensor->data() != nullptr && (tensor->device().device_type == kDLCUDA ||
                                        tensor->device().device_type == kDLCUDAManaged))
        stream = context.cuda_stream().get();
      auto copied = detail::capture_tensor_snapshot(*tensor, sample.metadata, remaining, stream);
      if (!copied) {
        auto error = std::move(copied).error();
        error.message += "; tensor_field_id=" + std::to_string(field.field_id);
        return make_unexpected(std::move(error));
      }
      remaining -= copied->bytes().size();
      snapshot.tensors.emplace_back(field.name, std::move(*copied));
    }
    storage_->snapshots.push_back(std::move(snapshot));
    return {};
  }

  expected<void, Error> retain_schema(const ForeignSchemaSample& sample,
                                      ExecutionContext& context) {
    const auto& plan = storage_->retained.plan;
    if (!plan || storage_->retained.records >= plan->records)
      return make_unexpected(
          Error{ErrorCode::kResourceExhausted, "operator test Retain bound exceeded"});
    const auto& fields = storage_->registration->row->descriptor_tensor_fields;
    for (std::size_t i = 0; i < fields.size(); ++i) {
      if (const auto* tensor = field_tensor(sample.data, fields[i].name)) {
        auto validated = detail::validate_retained_tensor(*tensor, *plan, i, context.self());
        if (!validated)
          return validated;
      }
    }
    storage_->retained.run = context.self().run_generation;
    ++storage_->retained.records;
    if (storage_->retained.records == plan->records) {
      storage_->retained.boundary_observed = true;
      context.request_stop();
    }
    return {};
  }

  expected<void, Error> store_sample(Sample sample, ExecutionContext& context) {
    if constexpr (std::same_as<Sample, ForeignSchemaSample>) {
      if (storage_->policy == CapturePolicy::kSnapshot)
        return snapshot_schema(sample, context);
      if (storage_->policy == CapturePolicy::kNative) {
        auto retained = retain_schema(sample, context);
        if (!retained)
          return retained;
      }
    }
    storage_->samples.push_back(std::move(sample));
    return {};
  }

  std::shared_ptr<detail::CaptureMailbox<Storage<Sample>>> mailbox_;
  std::shared_ptr<Storage<Sample>> storage_;
  std::size_t capacity_;
  std::size_t maximum_bytes_;
  OpaquePortRef input_;
};

template <typename Sample>
struct Declaration final : detail::CaptureDeclaration {
  std::shared_ptr<const Registration> registration;
  CapturePolicy policy;
  std::size_t capacity;
  std::size_t maximum_bytes;
  Declaration(std::shared_ptr<const Registration> row, CapturePolicy capture, std::size_t limit,
              std::size_t bytes)
      : registration(std::move(row)), policy(capture), capacity(limit), maximum_bytes(bytes) {
    if (capacity == 0U || capacity > UINT32_MAX)
      throw std::invalid_argument("capture capacity must fit positive uint32");
    const bool tensors = !registration->row->descriptor_tensor_fields.empty();
    if (tensors == (policy == CapturePolicy::kValue))
      throw std::invalid_argument("capture policy does not match payload");
    if (policy == CapturePolicy::kSnapshot &&
        (maximum_bytes == 0U || registration->row->clone_schema_metadata == nullptr))
      throw std::invalid_argument(
          "snapshot requires a positive byte limit and canonical metadata clone support");
  }
  PayloadIdentityRecord identity() const override { return registration->row->identity; }
  const void* type_token() const noexcept override { return registration->row->native_type_token; }
  const std::type_info* payload_type_info() const noexcept override {
    return registration->row->native_type_info;
  }
  std::optional<detail::RetainRequest> retention_request() const override {
    if (policy != CapturePolicy::kNative)
      return {};
    detail::RetainRequest result{.capacity = capacity};
    for (const auto& field : registration->row->descriptor_tensor_fields)
      result.field_ids.push_back(field.field_id);
    return result;
  }
  std::shared_ptr<detail::CaptureStorage> make_storage() const override {
    auto storage = std::make_shared<Storage<Sample>>(registration, policy);
    storage->samples.reserve(capacity);
    storage->snapshots.reserve(capacity);
    return storage;
  }
  OpaqueOperatorRef build(Graph& graph, const OpaqueOperatorRef& tested, std::string_view name,
                          std::string_view helper,
                          const std::shared_ptr<detail::CaptureStorage>& storage,
                          std::optional<ConnectionOptions> options) const override {
    const auto collector = holoscan::detail::OperatorTestingAccess::erase(
        graph.op<Collector<Sample>>(helper,
                                    std::make_shared<detail::CaptureMailbox<Storage<Sample>>>(
                                        std::static_pointer_cast<Storage<Sample>>(storage)),
                                    capacity,
                                    maximum_bytes));
    registration->row->add_flow(
        graph,
        tested.port(name),
        collector.port("input"),
        options.value_or(ConnectionOptions{.queue_depth = static_cast<std::uint32_t>(capacity)}));
    return collector;
  }
};

template <typename Sample>
class RegisteredAdapter final : public PayloadAdapter {
 public:
  explicit RegisteredAdapter(std::shared_ptr<const Registration> row)
      : registration_(std::move(row)) {}
  StagedInput stage(const nb::sequence& values, const nb::sequence& times) const override {
    if (nb::len(values) != nb::len(times))
      throw nb::value_error("sample values and timestamps differ in length");
    std::vector<Staged<Sample>> samples;
    for (std::size_t i = 0; i < nb::len(values); ++i) {
      Staged<Sample> value;
      value.sample.has_data = true;
      if constexpr (std::same_as<Sample, ForeignObjectSample>)
        value.sample.data = registration_->owner->services->object_from_python(values[i]);
      else
        value.sample.data = registration_->owner->services->schema_from_python(values[i]);
      if (!times[i].is_none())
        value.timestamp = nb::cast<std::int64_t>(times[i]);
      samples.push_back(std::move(value));
    }
    return detail::RegisteredAdapterAccess::input(std::make_shared<InputScript<Sample>>(
        std::make_shared<InputBatch<Sample>>(registration_, std::move(samples))));
  }
  CaptureRequest capture(CapturePolicy policy, std::size_t capacity,
                         std::size_t bytes) const override {
    return detail::RegisteredAdapterAccess::capture(
        std::make_shared<Declaration<Sample>>(registration_, policy, capacity, bytes));
  }
  nb::list samples(const TestResult& result, std::string_view name) const override {
    const auto& storage = checked_storage(result, name);
    nb::list values;
    for (const auto& sample : storage.samples) {
      if constexpr (std::same_as<Sample, ForeignObjectSample>)
        values.append(registration_->owner->services->object_sample(sample));
      else
        values.append(registration_->owner->services->schema_sample(
            sample, registration_->schema_type.borrow()));
    }
    return values;
  }
  nb::list snapshots(const TestResult& result, std::string_view name) const override {
    nb::list values;
    for (const auto& snapshot : checked_storage(result, name).snapshots)
      values.append(nb::cast(snapshot));
    return values;
  }
  CapturePolicy default_policy() const noexcept override {
    return registration_->row->descriptor_tensor_fields.empty() ? CapturePolicy::kValue
                                                                : CapturePolicy::kSnapshot;
  }

 private:
  const Storage<Sample>& checked_storage(const TestResult& result, std::string_view name) const {
    const auto* storage = dynamic_cast<const Storage<Sample>*>(
        &detail::RegisteredAdapterAccess::storage(result, name));
    if (!storage || storage->registration->row != registration_->row)
      throw nb::type_error("registered capture type mismatch");
    return *storage;
  }
  std::shared_ptr<const Registration> registration_;
};

}  // namespace

std::shared_ptr<const PayloadAdapter> registered_payload_adapter(
    const PortRecord& port, std::shared_ptr<const BindingOwner> owner) {
  const auto& row =
      required(authoring::authoring_registry().payload_type(port.payload_identity)).get();
  if (row.native_type_token == nullptr || row.native_type_info == nullptr ||
      row.output_declaration == nullptr)
    throw RuntimeError{
        Error{ErrorCode::kNotSupported, "canonical native payload witness unavailable"}};
  auto registration = std::make_shared<Registration>(Registration{std::move(owner), &row, {}});
  if (row.generated_schema_facade) {
    auto type = nb::module_::import_("holoscan.payload.schema")
                    .attr("_resolve_type")(nb::str(row.name.data(), row.name.size()));
    if (type.is_none())
      throw RuntimeError{
          Error{ErrorCode::kNotSupported, "generated schema Python facade unavailable"}};
    registration->schema_type = GilSafeObject{type};
    return std::make_shared<RegisteredAdapter<ForeignSchemaSample>>(std::move(registration));
  }
  // Resolve the canonical Core row; never instantiate the process-local object type in testing.
  const auto& objects =
      required(authoring::authoring_registry().payload_type("holoscan.PyObjectPayload")).get();
  if (&row == &objects)
    return std::make_shared<RegisteredAdapter<ForeignObjectSample>>(std::move(registration));
  throw RuntimeError{
      Error{ErrorCode::kNotSupported,
            "operator testing payload adapter unavailable; port='" + port.name + "'"}};
}

void init_schema_snapshot(nb::module_& module) {
  holoscan::python::bind_class_in_module<RegisteredSnapshot>(
      module,
      "holoscan.testing",
      "SchemaSnapshot",
      "Own copied schema metadata and independent host snapshots of every present admitted Tensor "
      "field. Absent optional fields retain their absence.")
      .def_prop_ro(
          "metadata",
          [](const RegisteredSnapshot& value) {
            return value.owner->services->schema_to_python(value.metadata,
                                                           value.schema_type.borrow());
          },
          "Return copied schema metadata with Tensor members unset. Use tensor() for field "
          "content; this metadata is not a complete payload to republish.")
      .def_ro("sample_metadata",
              &RegisteredSnapshot::sample_metadata,
              "Return semantic sample metadata copied when the schema output was collected.")
      .def_prop_ro(
          "tensor_fields",
          [](const RegisteredSnapshot& value) {
            std::vector<std::string> names;
            names.reserve(value.tensors.size());
            for (const auto& [name, tensor] : value.tensors)
              names.push_back(name);
            return names;
          },
          "Return all admitted Tensor field names, including absent optional fields.")
      .def(
          "has_tensor",
          [](const RegisteredSnapshot& value, const std::string& name) {
            for (const auto& [field, tensor] : value.tensors)
              if (field == name)
                return tensor.has_value();
            throw nb::value_error("unknown snapshot Tensor field");
          },
          "Return whether a known Tensor field was present. Raise ValueError for an unknown field.")
      .def(
          "tensor",
          [](const RegisteredSnapshot& value, const std::string& name) {
            for (const auto& [field, tensor] : value.tensors)
              if (field == name) {
                if (!tensor)
                  throw nb::value_error("snapshot Tensor field is absent");
                return *tensor;
              }
            throw nb::value_error("unknown snapshot Tensor field");
          },
          "Return the independent host snapshot of a present Tensor field. Raise ValueError for an "
          "absent or unknown field.");
}

}  // namespace holoscan::testing::python
