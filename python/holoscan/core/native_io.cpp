/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "native_io.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cuda_runtime_api.h>

#include <cstdint>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/authoring/register_payload.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/foreign_operator.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/tensor_output_loan.hpp>

#include "dl_converter.hpp"
#include "errors.hpp"
#include "graph.hpp"
#include "graph_pydoc.hpp"
#include "module_name.hpp"
#include "native_operator.hpp"
#include "payload.hpp"
#include "py_object_payload.hpp"
#include "schema_value.hpp"
#include "tensor_authoring.hpp"

namespace holoscan {

// Pairs one immutable declaration with its run-specific reference and optional Tensor converter.
struct NativePortBinding {
  NativePortDefinition definition;
  std::optional<authoring::OpaquePortRef> reference;
  GilSafeObject view_converter;
};

namespace {

// Python declaration descriptor for one typed input and its optional array-library view.
class PyInputDescriptor {
 public:
  PyInputDescriptor(std::string name, const authoring::RegisteredPayloadType* payload,
                    nb::handle view, nb::handle schema_type)
      : name_(std::move(name)),
        payload_(payload),
        view_(nb::borrow<nb::object>(view)),
        schema_type_(nb::borrow<nb::object>(schema_type)) {}

  [[nodiscard]] const std::string& name() const noexcept { return name_; }
  [[nodiscard]] const authoring::RegisteredPayloadType& payload() const noexcept {
    return *payload_;
  }
  [[nodiscard]] nb::object view() const { return view_; }
  [[nodiscard]] nb::object schema_type() const { return schema_type_; }

 private:
  std::string name_;
  const authoring::RegisteredPayloadType* payload_{};
  nb::object view_;
  nb::object schema_type_;
};

// Python declaration descriptor for one typed output port on a hosted operator class.
class PyOutputDescriptor {
 public:
  PyOutputDescriptor(std::string name, const authoring::RegisteredPayloadType* payload,
                     nb::handle schema_type)
      : name_(std::move(name)),
        payload_(payload),
        schema_type_(nb::borrow<nb::object>(schema_type)) {}

  [[nodiscard]] const std::string& name() const noexcept { return name_; }
  [[nodiscard]] const authoring::RegisteredPayloadType& payload() const noexcept {
    return *payload_;
  }
  [[nodiscard]] nb::object schema_type() const { return schema_type_; }

 private:
  std::string name_;
  const authoring::RegisteredPayloadType* payload_{};
  nb::object schema_type_;
};

// Configures required and optional lifecycle methods for one Python-defined operator.
class PyNativeLifecycleBuilder {
 public:
  explicit PyNativeLifecycleBuilder(PyCallbackBorrow<PyLifecycleHookBindings> bindings)
      : bindings_(std::move(bindings)) {}

  PyNativeLifecycleBuilder& stage(LifecycleStage stage, std::string method) {
    bindings_.with_value("AUTHORING_OPERATOR_SPEC_EXPIRED",
                         [&](auto& bindings) { bindings.bind(stage, method, true); });
    return *this;
  }

  PyNativeLifecycleBuilder& optional(LifecycleStage stage, std::string method) {
    bindings_.with_value("AUTHORING_OPERATOR_SPEC_EXPIRED",
                         [&](auto& bindings) { bindings.bind(stage, method, false); });
    return *this;
  }

 private:
  PyCallbackBorrow<PyLifecycleHookBindings> bindings_;
};

// Raises one field-attributed authoring error for invalid Python publication options.
[[noreturn]] void throw_emit_option_error(std::string_view token, std::string_view field) {
  throw RuntimeError{
      Error{ErrorCode::kInvalidArgument, std::string{token} + " field_name=" + std::string{field}}};
}

// Uses Python's repr so optional fields retain native None and enum formatting.
[[nodiscard]] std::string python_repr(nb::handle value) {
  return nb::cast<std::string>(nb::repr(value));
}

// Owns publication overrides and merges omitted fields from forwarded sample metadata.
class PyEmitOptions {
 public:
  PyEmitOptions(nb::handle capture_time, nb::handle frame_id, nb::handle trace_id,
                nb::handle flags) {
    if (!capture_time.is_none()) {
      capture_time_ = nb::cast<TimePoint>(capture_time);
    }
    if (!frame_id.is_none()) {
      frame_id_ = python_uint64(frame_id, "frame_id");
    }
    if (!trace_id.is_none()) {
      trace_id_ = python_uint64(trace_id, "trace_id");
    }
    if (!flags.is_none()) {
      flags_ = nb::cast<SampleFlags>(flags);
    }
  }

  explicit PyEmitOptions(const EmitOptions& options, bool authoritative)
      : capture_time_(options.capture_time),
        frame_id_(options.frame_id),
        trace_id_(options.trace ? std::optional{options.trace->trace_id} : std::nullopt),
        flags_(options.flags),
        authoritative_(authoritative) {}

  [[nodiscard]] nb::object capture_time() const {
    return capture_time_ ? nb::cast(*capture_time_) : nb::none();
  }
  [[nodiscard]] nb::object frame_id() const {
    return frame_id_ ? nb::cast(*frame_id_) : nb::none();
  }
  [[nodiscard]] nb::object trace_id() const {
    return trace_id_ ? nb::cast(*trace_id_) : nb::none();
  }
  [[nodiscard]] nb::object flags() const { return flags_ ? nb::cast(*flags_) : nb::none(); }
  [[nodiscard]] bool authoritative() const noexcept { return authoritative_; }

  [[nodiscard]] bool equals(const PyEmitOptions& other) const noexcept {
    return capture_time_ == other.capture_time_ && frame_id_ == other.frame_id_ &&
           trace_id_ == other.trace_id_ && flags_ == other.flags_ &&
           authoritative_ == other.authoritative_;
  }

  [[nodiscard]] std::string repr() const {
    return "EmitOptions(capture_time=" + python_repr(capture_time()) +
           ", frame_id=" + python_repr(frame_id()) + ", trace_id=" + python_repr(trace_id()) +
           ", flags=" + python_repr(flags()) +
           ", authoritative=" + (authoritative_ ? "True" : "False") + ")";
  }

  [[nodiscard]] Py_hash_t hash() const {
    return nb::hash(
        nb::make_tuple(capture_time(), frame_id(), trace_id(), flags(), authoritative_));
  }

  [[nodiscard]] EmitOptions value() const noexcept {
    return EmitOptions{
        .capture_time = capture_time_,
        .frame_id = frame_id_,
        .trace = trace_id_ ? std::optional{TraceContext::root(*trace_id_)} : std::nullopt,
        .flags = flags_.value_or(SampleFlags::kNone),
    };
  }

  [[nodiscard]] EmitOptions merge(const SampleMetadata& metadata) const {
    if (authoritative_) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument,
                               "PYTHON_FORWARD_AUTHORITATIVE_OPTIONS_UNSUPPORTED use=Output.emit"}};
    }
    const std::optional<TimePoint> forwarded_capture =
        metadata.source_clock_id.valid() ? std::optional{metadata.capture_time()} : std::nullopt;
    const std::optional<std::uint64_t> forwarded_frame =
        metadata.frame_id != 0U ? std::optional{metadata.frame_id} : std::nullopt;
    const std::optional<TraceContext> forwarded_trace =
        metadata.trace_id != 0U ? std::optional{TraceContext::root(metadata.trace_id)}
                                : std::nullopt;
    return EmitOptions{
        .capture_time = capture_time_.has_value() ? capture_time_ : forwarded_capture,
        .frame_id = frame_id_.has_value() ? frame_id_ : forwarded_frame,
        .trace =
            trace_id_.has_value() ? std::optional{TraceContext::root(*trace_id_)} : forwarded_trace,
        .flags = flags_.value_or(metadata.flags),
    };
  }

 private:
  [[nodiscard]] static std::uint64_t python_uint64(nb::handle value, std::string_view field) {
    if (nb::isinstance<nb::bool_>(value)) {
      throw_emit_option_error("AUTHORING_PARAMETER_TYPE_MISMATCH", field);
    }
    PyObject* const indexed = PyNumber_Index(value.ptr());
    if (indexed == nullptr) {
      PyErr_Clear();
      throw_emit_option_error("AUTHORING_PARAMETER_TYPE_MISMATCH", field);
    }
    const nb::object owner = nb::steal<nb::object>(indexed);
    const unsigned long long converted =  // NOLINT(runtime/int)
        PyLong_AsUnsignedLongLong(owner.ptr());
    if (converted == static_cast<unsigned long long>(-1) && PyErr_Occurred() != nullptr) {
      PyErr_Clear();
      throw_emit_option_error("PARAM_VALIDATION_FAILED category=out_of_range", field);
    }
    return static_cast<std::uint64_t>(converted);
  }

  std::optional<TimePoint> capture_time_;
  std::optional<std::uint64_t> frame_id_;
  std::optional<std::uint64_t> trace_id_;
  std::optional<SampleFlags> flags_;
  bool authoritative_{};
};

// Normalizes None or validates the public EmitOptions wrapper before publication.
[[nodiscard]] PyEmitOptions python_emit_options(nb::handle value) {
  if (value.is_none()) {
    return PyEmitOptions{nb::none(), nb::none(), nb::none(), nb::none()};
  }
  if (!nb::isinstance<PyEmitOptions>(value)) {
    throw_emit_option_error("PYTHON_EMIT_OPTIONS_REQUIRED", "options");
  }
  return nb::cast<const PyEmitOptions&>(value);
}

[[nodiscard]] const authoring::OpaquePortRef& required_port_reference(
    const NativePortBinding& binding) {
  if (!binding.reference.has_value()) {
    throw RuntimeError{Error{ErrorCode::kNotReady, "FOREIGN_OPERATOR_PORT_NOT_BOUND"}};
  }
  return binding.reference.value();
}

}  // namespace

nb::object python_authoritative_emit_options_from_native(const EmitOptions& options) {
  return nb::cast(PyEmitOptions{options, true});
}

// Owns scalar or Tensor sample data together with copied selection metadata.
class PySample {
 public:
  explicit PySample(authoring::ForeignScalarSample sample)
      : data_(sample.has_data ? payload_scalar_to_python(sample.data) : nb::none()),
        metadata_(sample.metadata),
        stale_(sample.is_stale),
        missing_(sample.is_missing) {}

  PySample(authoring::ForeignTensorSample sample, const GilSafeObject& converter)
      : metadata_(sample.metadata), stale_(sample.is_stale), missing_(sample.is_missing) {
    if (!sample.has_data) {
      data_ = nb::none();
      return;
    }
    nb::object tensor = nb::cast(std::make_shared<Tensor>(std::move(sample.data)));
    data_ = converter ? converter.borrow()(tensor) : std::move(tensor);
  }

  explicit PySample(const authoring::ForeignObjectSample& sample)
      : metadata_(sample.metadata), stale_(sample.is_stale), missing_(sample.is_missing) {
    if (!sample.has_data) {
      data_ = nb::none();
      return;
    }
    const PyObjectPayload* const payload = sample.data.get<PyObjectPayload>();
    if (payload == nullptr) {
      throw RuntimeError{Error{ErrorCode::kProtocolError, "AUTHORING_PAYLOAD_VALUE_TYPE_MISMATCH"}};
    }
    data_ = payload->borrow();
  }

  PySample(const authoring::ForeignSchemaSample& sample, const GilSafeObject& schema_type)
      : metadata_(sample.metadata), stale_(sample.is_stale), missing_(sample.is_missing) {
    data_ = sample.has_data ? materialize_python_schema(sample.data, schema_type) : nb::none();
  }

  [[nodiscard]] nb::object data() const { return data_; }
  [[nodiscard]] const SampleMetadata& metadata() const noexcept { return metadata_; }
  [[nodiscard]] bool is_stale() const noexcept { return stale_; }
  [[nodiscard]] bool is_missing() const noexcept { return missing_; }

 private:
  nb::object data_;
  SampleMetadata metadata_{};
  bool stale_{};
  bool missing_{};
};

nb::object python_sample(authoring::ForeignScalarSample sample) {
  return nb::cast(PySample{std::move(sample)});
}

nb::object python_sample(const authoring::ForeignSchemaSample& sample, nb::handle schema_type) {
  return nb::cast(PySample{std::move(sample), GilSafeObject{schema_type}});
}

nb::object python_sample(authoring::ForeignTensorSample sample) {
  return nb::cast(PySample{std::move(sample), GilSafeObject{}});
}

// Owns per-instance ports, conversion callbacks, parameter projection, and active execution state.
class PyNativeOperatorBindings {
 public:
  PyNativeOperatorBindings(std::vector<NativePortDefinition> definitions,
                           HostedParameterSnapshot parameters,
                           ParameterByteSpanResolver parameter_byte_span_resolver,
                           ParameterMemoryKindResolver parameter_memory_kind_resolver)
      : parameters_(std::move(parameters)),
        parameter_byte_span_resolver_(std::move(parameter_byte_span_resolver)),
        parameter_memory_kind_resolver_(std::move(parameter_memory_kind_resolver)) {
    for (auto& definition : definitions) {
      std::string port_name = definition.port_name;
      order_.push_back(port_name);
      GilSafeObject converter = resolve_view_converter(definition.view);
      ports_.emplace(std::move(port_name),
                     NativePortBinding{.definition = std::move(definition),
                                       .view_converter = std::move(converter)});
    }
  }

  [[nodiscard]] NativePortBinding& port(std::string_view name) {
    const auto found = ports_.find(name);
    if (found == ports_.end()) {
      throw RuntimeError{
          Error{ErrorCode::kNotFound, "AUTHORING_OPERATOR_PORT_UNKNOWN name=" + std::string{name}}};
    }
    return found->second;
  }

  [[nodiscard]] const std::vector<std::string>& order() const noexcept { return order_; }

  [[nodiscard]] TensorByteSpanExpression parameter_byte_span(nb::handle parameter,
                                                             bool shape) const {
    if (!parameter_byte_span_resolver_) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "GRAPH_TENSOR_BYTE_SPAN_DEPENDENCY_INVALID"}};
    }
    return parameter_byte_span_resolver_(parameter, shape);
  }

  [[nodiscard]] TensorMemoryKindExpression parameter_memory_kind(nb::handle parameter,
                                                                 MemoryKind when_false,
                                                                 MemoryKind when_true) const {
    if (!parameter_memory_kind_resolver_) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "GRAPH_TENSOR_MEMORY_KIND_DEPENDENCY_INVALID"}};
    }
    return parameter_memory_kind_resolver_(parameter, when_false, when_true);
  }

  void activate(ExecutionContext& context) noexcept {
    const std::lock_guard lock(activation_mutex_);
    ++activation_generation_;
    execution_ = &context;
    activation_thread_ = std::this_thread::get_id();
  }
  void deactivate() noexcept {
    const std::lock_guard lock(activation_mutex_);
    execution_ = nullptr;
    activation_thread_ = {};
  }
  [[nodiscard]] std::uint64_t activation_generation(std::string_view unavailable_token) const {
    const std::lock_guard lock(activation_mutex_);
    if (execution_ == nullptr || activation_thread_ != std::this_thread::get_id()) {
      throw RuntimeError{Error{ErrorCode::kNotReady, std::string{unavailable_token}}};
    }
    return activation_generation_;
  }
  void require_activation(std::uint64_t generation, std::string_view unavailable_token) const {
    const std::lock_guard lock(activation_mutex_);
    if (execution_ == nullptr || activation_generation_ != generation ||
        activation_thread_ != std::this_thread::get_id()) {
      throw RuntimeError{Error{ErrorCode::kNotReady, std::string{unavailable_token}}};
    }
  }
  [[nodiscard]] ExecutionContext& execution() const {
    const std::lock_guard lock(activation_mutex_);
    if (execution_ == nullptr || activation_thread_ != std::this_thread::get_id()) {
      throw RuntimeError{Error{ErrorCode::kNotReady, "PYTHON_EXECUTION_CONTEXT_UNAVAILABLE"}};
    }
    return *execution_;
  }

 private:
  [[nodiscard]] static GilSafeObject resolve_view_converter(const GilSafeObject& declaration) {
    if (!declaration || declaration.borrow().is_none()) {
      return {};
    }
    const nb::object value = declaration.borrow();
    PyObject* const from_dlpack_attribute = PyObject_GetAttrString(value.ptr(), "from_dlpack");
    if (from_dlpack_attribute != nullptr) {
      nb::object from_dlpack = nb::steal<nb::object>(from_dlpack_attribute);
      if (PyCallable_Check(from_dlpack.ptr()) != 0) {
        return GilSafeObject{std::move(from_dlpack)};
      }
    } else if (PyErr_ExceptionMatches(PyExc_AttributeError) != 0) {
      PyErr_Clear();
    } else {
      throw nb::python_error();
    }
    if (PyType_Check(value.ptr()) != 0 && nb::hasattr(value, "__module__")) {
      const std::string module_name = nb::cast<std::string>(value.attr("__module__"));
      const std::string root = module_name.substr(0U, module_name.find('.'));
      if (root == "cupy" || root == "numpy" || root == "torch") {
        const nb::object module = nb::module_::import_(root.c_str());
        if (nb::hasattr(module, "from_dlpack")) {
          return GilSafeObject{module.attr("from_dlpack")};
        }
      }
    }
    if (PyCallable_Check(value.ptr()) != 0) {
      return declaration;
    }
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "PYTHON_TENSOR_INPUT_VIEW_INVALID"}};
  }

  std::map<std::string, NativePortBinding, std::less<>> ports_;
  std::vector<std::string> order_;
  HostedParameterSnapshot parameters_;
  ParameterByteSpanResolver parameter_byte_span_resolver_;
  ParameterMemoryKindResolver parameter_memory_kind_resolver_;
  mutable std::mutex activation_mutex_;
  std::uint64_t activation_generation_{};
  ExecutionContext* execution_{};
  std::thread::id activation_thread_{};
};

class PySelectedBatch;

// Advances lazily through one callback-scoped selected-batch view.
class PySelectedBatchIterator {
 public:
  explicit PySelectedBatchIterator(std::shared_ptr<PySelectedBatch> batch)
      : batch_(std::move(batch)) {}

  [[nodiscard]] PySelectedBatchIterator& iter() noexcept { return *this; }
  [[nodiscard]] PySample next();

 private:
  std::shared_ptr<PySelectedBatch> batch_;
  std::size_t index_{};
};

// Owns selected payload snapshots while enforcing the source callback's view lifetime.
class PySelectedBatch : public std::enable_shared_from_this<PySelectedBatch> {
 public:
  PySelectedBatch(std::shared_ptr<PyNativeOperatorBindings> bindings, std::uint64_t generation,
                  NativePortBinding binding, std::size_t size)
      : bindings_(std::move(bindings)),
        generation_(generation),
        binding_(std::move(binding)),
        size_(size) {}

  [[nodiscard]] std::size_t size() const {
    require_live();
    return size_;
  }

  [[nodiscard]] PySample item(std::int64_t index) const {
    require_live();
    const std::int64_t size = static_cast<std::int64_t>(size_);
    const std::int64_t normalized = index < 0 ? index + size : index;
    if (normalized < 0 || normalized >= size) {
      throw nb::index_error("selected batch index out of range");
    }
    return materialize(static_cast<std::size_t>(normalized));
  }

  [[nodiscard]] PySample item_or_stop(std::size_t index) const {
    require_live();
    if (index >= size_) {
      throw nb::stop_iteration();
    }
    return materialize(index);
  }

  [[nodiscard]] std::vector<PySample> items(const nb::slice& slice) const {
    require_live();
    auto [start, stop, step, slice_length] = slice.compute(size_);
    (void)stop;
    std::vector<PySample> result;
    result.reserve(slice_length);
    for (std::size_t offset = 0; offset < slice_length; ++offset) {
      const auto index = start + static_cast<std::int64_t>(offset) * step;
      result.push_back(materialize(static_cast<std::size_t>(index)));
    }
    return result;
  }

  [[nodiscard]] std::shared_ptr<PySelectedBatchIterator> iter() {
    require_live();
    return std::make_shared<PySelectedBatchIterator>(shared_from_this());
  }

 private:
  [[nodiscard]] PySample materialize(std::size_t index) const {
    require_live();
    if (!binding_.reference) {
      throw RuntimeError{Error{ErrorCode::kNotReady, "PYTHON_SELECTED_BATCH_EXPIRED"}};
    }
    const authoring::RegisteredPayloadType* payload = binding_.definition.payload;
    const authoring::OpaquePortRef& reference = *binding_.reference;
    if (payload->identity.kind == PayloadIdentityKind::kSchema &&
        !detail::is_py_object_payload_identity(payload->identity)) {
      if (payload->selected_schema_batch_item == nullptr) {
        throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
      }
      auto result = [&] {
        const nb::gil_scoped_release release;
        return payload->selected_schema_batch_item(reference, index);
      }();
      require_live();
      if (!result) {
        throw RuntimeError{result.error()};
      }
      return PySample{std::move(*result), binding_.definition.schema_type};
    }
    if (detail::is_py_object_payload_identity(payload->identity)) {
      if (payload->selected_object_batch_item == nullptr) {
        throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
      }
      auto result = [&] {
        const nb::gil_scoped_release release;
        return payload->selected_object_batch_item(reference, index);
      }();
      require_live();
      if (!result) {
        throw RuntimeError{result.error()};
      }
      return PySample{std::move(*result)};
    }
    if (payload->identity.kind == PayloadIdentityKind::kTensor) {
      if (payload->selected_tensor_batch_item == nullptr) {
        throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
      }
      auto result = [&] {
        const nb::gil_scoped_release release;
        return payload->selected_tensor_batch_item(reference, index);
      }();
      require_live();
      if (!result) {
        throw RuntimeError{result.error()};
      }
      return PySample{std::move(*result), binding_.view_converter};
    }
    if (!payload->identity.scalar || payload->selected_scalar_batch_item == nullptr) {
      throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
    }
    auto result = [&] {
      const nb::gil_scoped_release release;
      return payload->selected_scalar_batch_item(reference, index);
    }();
    require_live();
    if (!result) {
      throw RuntimeError{result.error()};
    }
    return PySample{std::move(*result)};
  }

  void require_live() const {
    bindings_->require_activation(generation_, "PYTHON_SELECTED_BATCH_EXPIRED");
  }

  std::shared_ptr<PyNativeOperatorBindings> bindings_;
  std::uint64_t generation_{};
  NativePortBinding binding_;
  std::size_t size_{};
};

PySample PySelectedBatchIterator::next() {
  PySample sample = batch_->item_or_stop(index_);
  ++index_;
  return sample;
}

namespace {

[[nodiscard]] PyNativeOperatorBindings*& active_setup_bindings() noexcept {
  static thread_local PyNativeOperatorBindings* bindings{};
  return bindings;
}

[[nodiscard]] bool setup_active(const PyNativeOperatorBindings* bindings) noexcept {
  return active_setup_bindings() == bindings;
}

// Publishes one binding table only for the lexical duration of its setup callback.
class ScopedNativeSetupBindings {
 public:
  explicit ScopedNativeSetupBindings(PyNativeOperatorBindings& bindings) noexcept
      : previous_(std::exchange(active_setup_bindings(), &bindings)) {}
  ~ScopedNativeSetupBindings() { active_setup_bindings() = previous_; }

 private:
  PyNativeOperatorBindings* previous_{};
};

}  // namespace

PyNativePort::PyNativePort(std::shared_ptr<PyNativeOperatorBindings> bindings, std::string name)
    : bindings_(std::move(bindings)), name_(std::move(name)) {}

const std::string& PyNativePort::name() const noexcept {
  return name_;
}

PortDirection PyNativePort::direction() const {
  return binding().definition.direction;
}

PyPayloadType PyNativePort::payload_type() const {
  return PyPayloadType{binding().definition.payload};
}

InputSelection PyNativePort::selection() const {
  const std::uint64_t generation =
      bindings_->activation_generation("PYTHON_INPUT_SELECTION_UNAVAILABLE");
  const NativePortBinding& value = bound(PortDirection::kInput);
  if (value.definition.payload->input_selection == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  auto result = [&] {
    const nb::gil_scoped_release release;
    return value.definition.payload->input_selection(required_port_reference(value));
  }();
  bindings_->require_activation(generation, "PYTHON_INPUT_SELECTION_UNAVAILABLE");
  if (!result) {
    throw RuntimeError{result.error()};
  }
  return *result;
}

std::shared_ptr<PySelectedBatch> PyNativePort::selected_batch() const {
  const std::uint64_t generation =
      bindings_->activation_generation("PYTHON_SELECTED_BATCH_UNAVAILABLE");
  const NativePortBinding& value = bound(PortDirection::kInput);
  if (value.definition.payload->selected_batch_size == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  auto size = value.definition.payload->selected_batch_size(required_port_reference(value));
  bindings_->require_activation(generation, "PYTHON_SELECTED_BATCH_UNAVAILABLE");
  if (!size) {
    throw RuntimeError{size.error()};
  }
  return std::make_shared<PySelectedBatch>(bindings_, generation, value, *size);
}

PySample PyNativePort::receive() const {
  const std::uint64_t generation =
      bindings_->activation_generation("PYTHON_OPERATOR_PORT_UNAVAILABLE");
  const NativePortBinding& value = bound(PortDirection::kInput);
  if (value.definition.payload->identity.kind == PayloadIdentityKind::kSchema &&
      !detail::is_py_object_payload_identity(value.definition.payload->identity)) {
    if (value.definition.payload->receive_schema == nullptr) {
      throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
    }
    expected<authoring::ForeignSchemaSample, Error> result;
    {
      const nb::gil_scoped_release release;
      result = value.definition.payload->receive_schema(required_port_reference(value));
    }
    if (!result) {
      throw RuntimeError{result.error()};
    }
    return PySample{std::move(*result), value.definition.schema_type};
  }
  if (detail::is_py_object_payload_identity(value.definition.payload->identity)) {
    expected<authoring::ForeignObjectSample, Error> result;
    {
      const nb::gil_scoped_release release;
      result = value.definition.payload->receive_object(required_port_reference(value));
    }
    if (!result) {
      throw RuntimeError{result.error()};
    }
    return PySample{std::move(*result)};
  }
  if (value.definition.payload->identity.kind == PayloadIdentityKind::kTensor) {
    expected<authoring::ForeignTensorSample, Error> result;
    {
      const nb::gil_scoped_release release;
      result = value.definition.payload->receive_tensor(required_port_reference(value));
    }
    bindings_->require_activation(generation, "PYTHON_OPERATOR_PORT_UNAVAILABLE");
    if (!result) {
      throw RuntimeError{result.error()};
    }
    return PySample{std::move(*result), value.view_converter};
  }
  if (value.definition.payload->receive_scalar == nullptr ||
      !value.definition.payload->identity.scalar) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  expected<authoring::ForeignScalarSample, Error> result;
  {
    const nb::gil_scoped_release release;
    result = value.definition.payload->receive_scalar(required_port_reference(value));
  }
  bindings_->require_activation(generation, "PYTHON_OPERATOR_PORT_UNAVAILABLE");
  if (!result) {
    throw RuntimeError{result.error()};
  }
  return PySample{std::move(*result)};
}

nb::object PyNativePort::receive_data() const {
  const std::uint64_t generation =
      bindings_->activation_generation("PYTHON_OPERATOR_PORT_UNAVAILABLE");
  const NativePortBinding& value = bound(PortDirection::kInput);
  if (value.definition.payload->identity.kind == PayloadIdentityKind::kSchema &&
      !detail::is_py_object_payload_identity(value.definition.payload->identity)) {
    if (value.definition.payload->receive_schema_data == nullptr) {
      throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
    }
    expected<authoring::ForeignSchemaValue, Error> result;
    {
      const nb::gil_scoped_release release;
      result = value.definition.payload->receive_schema_data(required_port_reference(value));
    }
    if (!result) {
      throw RuntimeError{result.error()};
    }
    return materialize_python_schema(*result, value.definition.schema_type);
  }
  if (detail::is_py_object_payload_identity(value.definition.payload->identity)) {
    expected<authoring::ForeignPayloadValue, Error> result;
    {
      const nb::gil_scoped_release release;
      result = value.definition.payload->receive_object_data(required_port_reference(value));
    }
    if (!result) {
      throw RuntimeError{result.error()};
    }
    const PyObjectPayload* const payload = result->get<PyObjectPayload>();
    if (payload == nullptr) {
      throw RuntimeError{Error{ErrorCode::kProtocolError, "AUTHORING_PAYLOAD_VALUE_TYPE_MISMATCH"}};
    }
    return payload->borrow();
  }
  if (value.definition.payload->identity.kind == PayloadIdentityKind::kTensor) {
    expected<Tensor, Error> result;
    {
      const nb::gil_scoped_release release;
      result = value.definition.payload->receive_tensor_data(required_port_reference(value));
    }
    bindings_->require_activation(generation, "PYTHON_OPERATOR_PORT_UNAVAILABLE");
    if (!result) {
      throw RuntimeError{result.error()};
    }
    nb::object tensor = nb::cast(std::make_shared<Tensor>(std::move(*result)));
    return value.view_converter ? value.view_converter.borrow()(tensor) : tensor;
  }
  if (value.definition.payload->receive_scalar_data == nullptr ||
      !value.definition.payload->identity.scalar) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  expected<ParamScalarValue, Error> result;
  {
    const nb::gil_scoped_release release;
    result = value.definition.payload->receive_scalar_data(required_port_reference(value));
  }
  bindings_->require_activation(generation, "PYTHON_OPERATOR_PORT_UNAVAILABLE");
  if (!result) {
    throw RuntimeError{result.error()};
  }
  return payload_scalar_to_python(*result);
}

void PyNativePort::emit(nb::handle data, nb::handle options) const {
  emit_value(data, python_emit_options(options).value());
}

void PyNativePort::forward(const PySample& sample, nb::handle data, nb::handle options) const {
  if (sample.is_missing()) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "PYTHON_FORWARD_SAMPLE_MISSING"}};
  }
  emit_value(data, python_emit_options(options).merge(sample.metadata()));
}

void PyNativePort::emit_value(nb::handle data, EmitOptions options) const {
  const std::uint64_t generation =
      bindings_->activation_generation("PYTHON_OPERATOR_PORT_UNAVAILABLE");
  const NativePortBinding& value = bound(PortDirection::kOutput);
  if (value.definition.payload->identity.kind == PayloadIdentityKind::kSchema &&
      !detail::is_py_object_payload_identity(value.definition.payload->identity)) {
    if (value.definition.payload->emit_schema == nullptr) {
      throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
    }
    authoring::ForeignSchemaValue payload = python_schema_value(data);
    expected<void, Error> result;
    {
      const nb::gil_scoped_release release;
      result = value.definition.payload->emit_schema(
          required_port_reference(value), std::move(payload), options);
    }
    bindings_->require_activation(generation, "PYTHON_OPERATOR_PORT_UNAVAILABLE");
    if (!result) {
      throw RuntimeError{result.error()};
    }
    return;
  }
  if (detail::is_py_object_payload_identity(value.definition.payload->identity)) {
    authoring::ForeignPayloadValue payload =
        authoring::ForeignPayloadValue::from(PyObjectPayload{data});
    expected<void, Error> result;
    {
      const nb::gil_scoped_release release;
      result = value.definition.payload->emit_object(
          required_port_reference(value), std::move(payload), options);
    }
    bindings_->require_activation(generation, "PYTHON_OPERATOR_PORT_UNAVAILABLE");
    if (!result) {
      throw RuntimeError{result.error()};
    }
    return;
  }
  if (value.definition.payload->identity.kind == PayloadIdentityKind::kTensor) {
    Tensor tensor = [&] {
      if (nb::isinstance<Tensor>(data)) {
        return Tensor{nb::cast<const Tensor&>(data)};
      }
      const std::shared_ptr<Tensor> converted = as_tensor(nb::borrow<nb::object>(data));
      return Tensor{std::move(*converted)};
    }();
    const auto stream = [&] {
      const nb::gil_scoped_release release;
      return bindings_->execution().cuda_stream().get();
    }();
    bindings_->require_activation(generation, "PYTHON_OPERATOR_PORT_UNAVAILABLE");
    if ((tensor.device().device_type == kDLCUDA || tensor.device().device_type == kDLCUDAManaged) &&
        !tensor.has_readiness_stream_contract() && !tensor.set_readiness_stream(stream)) {
      throw RuntimeError{
          Error{ErrorCode::kNotSupported, "PYTHON_TENSOR_READINESS_STREAM_UNAVAILABLE"}};
    }
    expected<void, Error> result;
    {
      const nb::gil_scoped_release release;
      result = value.definition.payload->emit_tensor(
          required_port_reference(value), std::move(tensor), options);
    }
    bindings_->require_activation(generation, "PYTHON_OPERATOR_PORT_UNAVAILABLE");
    if (!result) {
      throw RuntimeError{result.error()};
    }
    return;
  }
  if (!value.definition.payload->identity.scalar ||
      value.definition.payload->emit_scalar == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  const ParamScalarValue scalar =
      payload_scalar_from_python(data, value.definition.payload->identity.scalar->kind);
  expected<void, Error> result;
  {
    const nb::gil_scoped_release release;
    result = value.definition.payload->emit_scalar(required_port_reference(value), scalar, options);
  }
  bindings_->require_activation(generation, "PYTHON_OPERATOR_PORT_UNAVAILABLE");
  if (!result) {
    throw RuntimeError{result.error()};
  }
}

TensorByteSpanExpression PyNativePort::input_max_byte_span() const {
  if (!setup_active(bindings_.get())) {
    throw RuntimeError{Error{ErrorCode::kNotReady, "PYTHON_OPERATOR_SETUP_CONTEXT_REQUIRED"}};
  }
  const NativePortBinding& value = bound(PortDirection::kInput);
  if (value.definition.payload->identity.kind != PayloadIdentityKind::kTensor ||
      value.definition.payload->tensor_input_max_byte_span == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  auto result =
      value.definition.payload->tensor_input_max_byte_span(required_port_reference(value));
  if (!result) {
    throw RuntimeError{result.error()};
  }
  return std::move(*result);
}

PyNativePort& PyNativePort::produces_tensor(const PyTensorOutputSpec& spec,
                                            std::optional<TensorByteSpanExpression> byte_span,
                                            std::optional<TensorMemoryKindExpression> memory_kind,
                                            std::optional<std::string> field) {
  if (!setup_active(bindings_.get())) {
    throw RuntimeError{Error{ErrorCode::kNotReady, "PYTHON_OPERATOR_SETUP_CONTEXT_REQUIRED"}};
  }
  const NativePortBinding& value = bound(PortDirection::kOutput);
  if (field) {
    if (memory_kind) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "GRAPH_TENSOR_MEMORY_KIND_DERIVATION_INVALID"}};
    }
    if (!byte_span) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "GRAPH_TENSOR_BYTE_SPAN_DERIVATION_INVALID"}};
    }
    if (!value.definition.schema_type ||
        value.definition.payload->schema_tensor_output_produces == nullptr) {
      throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
    }
    auto result = value.definition.payload->schema_tensor_output_produces(
        required_port_reference(value), *field, spec.value(), std::move(*byte_span));
    if (!result) {
      throw RuntimeError{result.error()};
    }
    return *this;
  }
  if (value.definition.payload->identity.kind != PayloadIdentityKind::kTensor ||
      value.definition.payload->tensor_output_produces == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  auto result = value.definition.payload->tensor_output_produces(
      required_port_reference(value),
      spec.value(),
      TensorOutputDerivations{.max_byte_span = std::move(byte_span),
                              .memory_kind = std::move(memory_kind)});
  if (!result) {
    throw RuntimeError{result.error()};
  }
  return *this;
}

PyNativePort& PyNativePort::expects_tensor(const PyTensorInputSpec& spec) {
  if (!setup_active(bindings_.get())) {
    throw RuntimeError{Error{ErrorCode::kNotReady, "PYTHON_OPERATOR_SETUP_CONTEXT_REQUIRED"}};
  }
  const NativePortBinding& value = bound(PortDirection::kInput);
  if (value.definition.payload->identity.kind != PayloadIdentityKind::kTensor ||
      value.definition.payload->tensor_input_expects == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  auto result =
      value.definition.payload->tensor_input_expects(required_port_reference(value), spec.value());
  if (!result) {
    throw RuntimeError{result.error()};
  }
  return *this;
}

PyNativePort& PyNativePort::accepts_projected_tensor() {
  if (!setup_active(bindings_.get())) {
    throw RuntimeError{Error{ErrorCode::kNotReady, "PYTHON_OPERATOR_SETUP_CONTEXT_REQUIRED"}};
  }
  NativePortBinding& value = bound(PortDirection::kInput);
  if (value.definition.payload->identity.kind != PayloadIdentityKind::kTensor ||
      value.definition.payload->tensor_input_accepts_projection == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  auto result = value.definition.payload->tensor_input_accepts_projection(*value.reference);
  if (!result) {
    throw RuntimeError{result.error()};
  }
  return *this;
}

PyNativePort& PyNativePort::expected_rate(std::uint64_t numerator, std::uint64_t denominator) {
  if (!setup_active(bindings_.get())) {
    throw RuntimeError{Error{ErrorCode::kNotReady, "PYTHON_OPERATOR_SETUP_CONTEXT_REQUIRED"}};
  }
  const NativePortBinding& value = bound(PortDirection::kInput);
  if (value.definition.payload->input_expected_rate == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  auto result = value.definition.payload->input_expected_rate(required_port_reference(value),
                                                              Rate::hz(numerator, denominator));
  if (!result) {
    throw RuntimeError{result.error()};
  }
  return *this;
}

PyNativePort& PyNativePort::nominal_rate(std::uint64_t numerator, std::uint64_t denominator) {
  if (!setup_active(bindings_.get())) {
    throw RuntimeError{Error{ErrorCode::kNotReady, "PYTHON_OPERATOR_SETUP_CONTEXT_REQUIRED"}};
  }
  const NativePortBinding& value = bound(PortDirection::kOutput);
  if (value.definition.payload->output_nominal_rate == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  auto result = value.definition.payload->output_nominal_rate(required_port_reference(value),
                                                              Rate::hz(numerator, denominator));
  if (!result) {
    throw RuntimeError{result.error()};
  }
  return *this;
}

PyNativePort& PyNativePort::follows(const PyNativePort& input, std::uint32_t numerator,
                                    std::uint32_t denominator) {
  return follows(input, RateRatio{numerator, denominator});
}

PyNativePort& PyNativePort::follows(const PyNativePort& input, RateRatio ratio) {
  if (!setup_active(bindings_.get()) || bindings_.get() != input.bindings_.get()) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_RATE_FOLLOWS_INPUT_INVALID"}};
  }
  const NativePortBinding& output_value = bound(PortDirection::kOutput);
  const NativePortBinding& input_value = input.bound(PortDirection::kInput);
  if (input_value.definition.payload->output_follows_input == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  auto result = input_value.definition.payload->output_follows_input(
      required_port_reference(output_value), required_port_reference(input_value), ratio);
  if (!result) {
    throw RuntimeError{result.error()};
  }
  return *this;
}

PortRef PyNativePort::contract_input_ref() const {
  const NativePortBinding& value = bound(PortDirection::kInput);
  if (value.definition.payload->contract_input_ref == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  return value.definition.payload->contract_input_ref(required_port_reference(value));
}

Contract& PyNativePort::contract_input(Contract& contract, Exact selector) const {
  const NativePortBinding& value = bound(PortDirection::kInput);
  if (value.definition.payload->contract_input_exact == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  return value.definition.payload->contract_input_exact(
      contract, required_port_reference(value), std::move(selector));
}

Contract& PyNativePort::contract_input(Contract& contract, LatestWithin selector) const {
  const NativePortBinding& value = bound(PortDirection::kInput);
  if (value.definition.payload->contract_input_latest_within == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  return value.definition.payload->contract_input_latest_within(
      contract, required_port_reference(value), std::move(selector));
}

Contract& PyNativePort::contract_input(Contract& contract, DrainAll selector) const {
  const NativePortBinding& value = bound(PortDirection::kInput);
  if (value.definition.payload->contract_input_drain_all == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  return value.definition.payload->contract_input_drain_all(
      contract, required_port_reference(value), std::move(selector));
}

Contract& PyNativePort::contract_input(Contract& contract, BracketedWindow selector) const {
  const NativePortBinding& value = bound(PortDirection::kInput);
  if (value.definition.payload->contract_input_bracketed_window == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  return value.definition.payload->contract_input_bracketed_window(
      contract, required_port_reference(value), std::move(selector));
}

NativePortBinding& PyNativePort::binding() const {
  return bindings_->port(name_);
}

NativePortBinding& PyNativePort::bound(PortDirection direction) const {
  NativePortBinding& result = binding();
  if (result.definition.direction != direction) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "GRAPH_PORT_DIRECTION_MISMATCH"}};
  }
  if (!result.reference) {
    throw RuntimeError{Error{ErrorCode::kNotReady, "FOREIGN_OPERATOR_PORT_NOT_BOUND"}};
  }
  return result;
}

// Callback-scoped Python facade used by setup() to bind declared ports into ForeignOperatorSpec.
class PyNativeOperatorSpec {
 public:
  PyNativeOperatorSpec(ForeignOperatorSpec& spec,
                       std::shared_ptr<PyNativeOperatorBindings> bindings,
                       PyLifecycleHookBindings& lifecycle)
      : spec_(spec), bindings_(std::move(bindings)), lifecycle_(lifecycle) {}

  PyNativePort input(const PyNativePort& port, std::optional<std::uint32_t> queue_depth,
                     bool optional) {
    return bind(port, PortDirection::kInput, queue_depth, optional);
  }
  PyNativePort output(const PyNativePort& port, std::optional<std::uint32_t> max_emits) {
    return bind(port, PortDirection::kOutput, max_emits, false);
  }
  [[nodiscard]] PyNativeLifecycleBuilder lifecycle() {
    return PyNativeLifecycleBuilder{lifecycle_};
  }

  void invalidate() noexcept {
    spec_.invalidate();
    lifecycle_.invalidate();
  }

 private:
  PyNativePort bind(const PyNativePort& port, PortDirection direction,
                    std::optional<std::uint32_t> bound, bool optional) {
    return spec_.with_value("AUTHORING_OPERATOR_SPEC_EXPIRED", [&](ForeignOperatorSpec& spec) {
      NativePortBinding& binding = bindings_->port(port.name());
      if (binding.definition.direction != direction || binding.reference) {
        throw RuntimeError{
            Error{ErrorCode::kInvalidArgument, "FOREIGN_OPERATOR_PORT_BINDING_INVALID"}};
      }
      auto result =
          direction == PortDirection::kInput
              ? spec.input(
                    *binding.definition.payload, binding.definition.port_name, bound, optional)
              : spec.output(*binding.definition.payload, binding.definition.port_name, bound);
      if (!result) {
        throw RuntimeError{result.error()};
      }
      binding.reference = std::move(*result);
      return port;
    });
  }

  PyCallbackBorrow<ForeignOperatorSpec> spec_;
  std::shared_ptr<PyNativeOperatorBindings> bindings_;
  PyCallbackBorrow<PyLifecycleHookBindings> lifecycle_;
};

// Shared liveness cell used by ExecutionContext and any CUDA stream facade derived from it.
struct PyExecutionContextState {
  ExecutionContext* context{};
  std::thread::id callback_thread{std::this_thread::get_id()};
};

// Adapts Core's activation stream to Python's CUDA array-library context-manager protocols.
class PyCudaStream {
 public:
  explicit PyCudaStream(std::shared_ptr<PyExecutionContextState> state)
      : state_(std::move(state)) {}
  PyCudaStream(cudaStream_t stream, int device) : test_stream_(stream), test_device_(device) {}
  PyCudaStream(const PyCudaStream&) = delete;
  PyCudaStream& operator=(const PyCudaStream&) = delete;
  PyCudaStream(PyCudaStream&& other) noexcept
      : state_(std::move(other.state_)),
        test_stream_(other.test_stream_),
        test_device_(other.test_device_),
        contexts_(std::move(other.contexts_)),
        entered_(std::exchange(other.entered_, false)) {}
  PyCudaStream& operator=(PyCudaStream&&) = delete;

  ~PyCudaStream() {
    if (!entered_ || !nb::is_alive()) {
      return;
    }
    const nb::gil_scoped_acquire gil;
    if (gil.is_valid()) {
      close_contexts(nb::none(), nb::none(), nb::none(), false);
    }
  }

  [[nodiscard]] nb::tuple cuda_stream() const {
    return nb::make_tuple(0U, reinterpret_cast<std::uintptr_t>(stream()));
  }

  PyCudaStream& enter() {
    const auto native_stream = stream();
    if (entered_) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "PYTHON_CUDA_STREAM_ALREADY_ENTERED"}};
    }
    entered_ = true;
    const std::uintptr_t pointer = reinterpret_cast<std::uintptr_t>(native_stream);
    if (pointer == 0U) {
      return *this;
    }
    try {
      const nb::dict modules = nb::cast<nb::dict>(nb::module_::import_("sys").attr("modules"));
      const auto enter_context = [&](const nb::object& context) {
        context.attr("__enter__")();
        contexts_.emplace_back(context);
      };
      const nb::str cupy_name{"cupy"};
      if (modules.contains(cupy_name)) {
        const nb::object cupy = nb::borrow<nb::object>(modules[cupy_name]);
        enter_context(
            cupy.attr("cuda").attr("ExternalStream")(pointer, stream_device(native_stream)));
      }
      const nb::str torch_name{"torch"};
      if (modules.contains(torch_name)) {
        const nb::object torch = nb::borrow<nb::object>(modules[torch_name]);
        const nb::object external = torch.attr("cuda").attr("ExternalStream")(
            pointer, nb::arg("device") = stream_device(native_stream));
        enter_context(torch.attr("cuda").attr("stream")(external));
      }
    } catch (...) {
      const std::exception_ptr original = std::current_exception();
      close_contexts(nb::none(), nb::none(), nb::none(), false);
      std::rethrow_exception(original);
    }
    return *this;
  }

  bool exit(nb::handle exception_type, nb::handle exception, nb::handle traceback) {
    close_contexts(exception_type, exception, traceback, true);
    return false;
  }

 private:
  void close_contexts(nb::handle exception_type, nb::handle exception, nb::handle traceback,
                      bool propagate) {
    std::exception_ptr first_error;
    for (auto iterator = contexts_.rbegin(); iterator != contexts_.rend(); ++iterator) {
      try {
        iterator->borrow().attr("__exit__")(exception_type, exception, traceback);
      } catch (...) {
        if (first_error == nullptr) {
          first_error = std::current_exception();
        }
      }
    }
    contexts_.clear();
    entered_ = false;
    if (propagate && first_error != nullptr) {
      std::rethrow_exception(first_error);
    }
  }

  [[nodiscard]] cudaStream_t stream() const {
    if (test_stream_.has_value()) {
      return *test_stream_;
    }
    if (state_ == nullptr || state_->context == nullptr ||
        state_->callback_thread != std::this_thread::get_id()) {
      throw RuntimeError{Error{ErrorCode::kNotReady, "PYTHON_EXECUTION_CONTEXT_EXPIRED"}};
    }
    return state_->context->cuda_stream().get();
  }

  [[nodiscard]] int stream_device(cudaStream_t native_stream) const {
    if (test_device_.has_value()) {
      return *test_device_;
    }
    int device{};
#if CUDART_VERSION >= 12080
    const cudaError_t status = cudaStreamGetDevice(native_stream, &device);
#else
    // cudaStreamGetDevice was added in CUDA 12.8. Before then, callback entry has already selected
    // the activation stream's device, so use the Runtime API's current-device query.
    (void)native_stream;
    const cudaError_t status = cudaGetDevice(&device);
#endif
    if (status != cudaSuccess) {
      throw RuntimeError{Error{ErrorCode::kFailure, "PYTHON_CUDA_STREAM_DEVICE_QUERY_FAILED"}};
    }
    return device;
  }

  std::shared_ptr<PyExecutionContextState> state_;
  std::optional<cudaStream_t> test_stream_;
  std::optional<int> test_device_;
  std::vector<GilSafeObject> contexts_;
  bool entered_{};
};

// Callback-scoped Python facade exposing execution metadata and the active CUDA stream.
class PyExecutionContext {
 public:
  PyExecutionContext(ExecutionContext& context, std::shared_ptr<PyNativeOperatorBindings> bindings)
      : state_(std::make_shared<PyExecutionContextState>(
            PyExecutionContextState{.context = &context})),
        bindings_(std::move(bindings)) {
    bindings_->activate(context);
  }

  ~PyExecutionContext() { invalidate(); }

  [[nodiscard]] PyCudaStream stream() const {
    ensure_live();
    return PyCudaStream{state_};
  }

  [[nodiscard]] std::string operator_path() const {
    ensure_live();
    return std::string{state_->context->operator_path()};
  }

  [[nodiscard]] ClockRef clock() const {
    ensure_live();
    return state_->context->clock();
  }

  [[nodiscard]] TimePoint activation_time() const {
    ensure_live();
    return state_->context->activation_time();
  }

  [[nodiscard]] TagTime tag_time() const {
    ensure_live();
    return state_->context->tag_time();
  }

  [[nodiscard]] std::uint64_t trace_id() const {
    ensure_live();
    return state_->context->trace_id();
  }

  [[nodiscard]] ActivationStatus activation() const {
    ensure_live();
    return state_->context->activation();
  }

  [[nodiscard]] bool stop_requested() const {
    ensure_live();
    return state_->context->cancellation().stop_requested();
  }

  void request_completion() const {
    ensure_live();
    state_->context->request_completion();
  }

  void request_stop() const {
    ensure_live();
    state_->context->request_stop();
  }

  void invalidate() noexcept {
    const bool was_live = state_ != nullptr && state_->context != nullptr;
    if (was_live) {
      state_->context = nullptr;
    }
    if (was_live && bindings_ != nullptr) {
      bindings_->deactivate();
    }
  }

 private:
  void ensure_live() const {
    if (state_ == nullptr || state_->context == nullptr) {
      throw RuntimeError{Error{ErrorCode::kNotReady, "PYTHON_EXECUTION_CONTEXT_EXPIRED"}};
    }
    static_cast<void>(bindings_->activation_generation("PYTHON_EXECUTION_CONTEXT_EXPIRED"));
  }

  std::shared_ptr<PyExecutionContextState> state_;
  std::shared_ptr<PyNativeOperatorBindings> bindings_;
};

// Owns the shape, dtype, and optional strides used to request one generated-schema Tensor field.
class PyTensorLoanRequest {
 public:
  PyTensorLoanRequest(std::vector<std::int64_t> shape, nb::handle dtype, nb::handle strides)
      : shape_(std::move(shape)), dtype_(python_tensor_dtype(dtype)) {
    if (!strides.is_none()) {
      strides_ = nb::cast<std::vector<std::int64_t>>(strides);
    }
  }

  [[nodiscard]] const std::vector<std::int64_t>& shape() const noexcept { return shape_; }
  [[nodiscard]] DLDataType dtype() const noexcept { return dtype_; }
  [[nodiscard]] nb::object strides() const { return strides_ ? nb::cast(*strides_) : nb::none(); }
  [[nodiscard]] TensorLoanRequest value() const noexcept {
    return TensorLoanRequest{
        .shape = shape_,
        .dtype = dtype_,
        .byte_strides =
            strides_ ? TensorByteStridesView{*strides_} : TensorByteStridesView{std::nullopt},
    };
  }

 private:
  std::vector<std::int64_t> shape_;
  DLDataType dtype_{};
  std::optional<std::vector<std::int64_t>> strides_;
};

// Owns a runtime Tensor loan, write guard, and borrowed Tensor view through commit or abandonment.
class PyTensorLoan {
 public:
  PyTensorLoan(TensorOutputLoan loan, const authoring::RegisteredPayloadType& payload,
               authoring::OpaquePortRef output, cudaStream_t stream, EmitOptions options)
      : loan_(std::make_unique<TensorOutputLoan>(std::move(loan))),
        payload_(&payload),
        output_(std::move(output)),
        options_(options) {
    auto writer = [&] {
      const nb::gil_scoped_release release;
      return loan_->memory_kind() == MemoryKind::kCudaDevice ||
                     loan_->memory_kind() == MemoryKind::kCudaManaged
                 ? loan_->write(::cuda::stream_ref{stream})
                 : loan_->write_host();
    }();
    if (!writer) {
      throw RuntimeError{writer.error()};
    }
    guard_ = std::make_unique<TensorOutputWriteGuard>(std::move(*writer));
    tensor_ = std::make_shared<Tensor>(guard_->tensor());
  }
  PyTensorLoan(const PyTensorLoan&) = delete;
  PyTensorLoan& operator=(const PyTensorLoan&) = delete;
  PyTensorLoan(PyTensorLoan&&) noexcept = default;
  PyTensorLoan& operator=(PyTensorLoan&&) = delete;

  ~PyTensorLoan() {
    invalidate_tensor();
    if (guard_ == nullptr && loan_ == nullptr) {
      return;
    }
    // Nanobind destroys this Python-owned wrapper with the GIL held. Release it while the
    // potentially blocking runtime pool objects execute their no-throw abandonment paths.
    if (nb::is_alive()) {
      const nb::gil_scoped_release release;
      abandon();
      return;
    }
    abandon();
  }

  [[nodiscard]] std::shared_ptr<Tensor> tensor() const {
    ensure_live();
    return tensor_;
  }

  PyTensorLoan& enter() {
    ensure_live();
    if (entered_) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "PYTHON_TENSOR_LOAN_ALREADY_ENTERED"}};
    }
    entered_ = true;
    return *this;
  }

  bool exit(nb::handle exception_type, nb::handle, nb::handle) {
    ensure_live();
    if (!exception_type.is_none()) {
      invalidate_tensor();
      {
        const nb::gil_scoped_release release;
        abandon();
      }
      return false;
    }
    expected<void, Error> committed;
    {
      const nb::gil_scoped_release release;
      committed = std::move(*guard_).commit();
    }
    guard_.reset();
    invalidate_tensor();
    entered_ = false;
    if (!committed) {
      {
        const nb::gil_scoped_release release;
        abandon();
      }
      throw RuntimeError{committed.error()};
    }
    TensorOutputLoan loan = std::move(*loan_);
    loan_.reset();
    expected<void, Error> emitted;
    {
      const nb::gil_scoped_release release;
      emitted = payload_->emit_tensor_loan(output_, std::move(loan), options_);
    }
    if (!emitted) {
      throw RuntimeError{emitted.error()};
    }
    return false;
  }

 private:
  void ensure_live() const {
    if (loan_ == nullptr || guard_ == nullptr || tensor_ == nullptr) {
      throw_consumed_loan_error();
    }
  }

  void invalidate_tensor() noexcept {
    if (tensor_ != nullptr) {
      *tensor_ = Tensor{};
      tensor_.reset();
    }
  }

  void abandon() noexcept {
    guard_.reset();
    // Both an active guard and a live loan have no-throw abandonment destructors. Rely on those
    // terminal paths so an exception leaving the Python block does not add a second cancel attempt.
    loan_.reset();
    entered_ = false;
  }

  std::unique_ptr<TensorOutputLoan> loan_;
  std::unique_ptr<TensorOutputWriteGuard> guard_;
  std::shared_ptr<Tensor> tensor_;
  const authoring::RegisteredPayloadType* payload_{};
  authoring::OpaquePortRef output_;
  EmitOptions options_{};
  bool entered_{};
};

// Owns a descriptor Tensor loan, its field guards and views, and publication options.
class PyTensorSchemaLoan {
 public:
  PyTensorSchemaLoan(authoring::ForeignTensorSchemaLoan loan,
                     const authoring::RegisteredPayloadType& payload,
                     authoring::OpaquePortRef output, authoring::ForeignSchemaValue descriptor,
                     cudaStream_t stream, EmitOptions options)
      : loan_(std::make_unique<authoring::ForeignTensorSchemaLoan>(std::move(loan))),
        payload_(&payload),
        output_(std::move(output)),
        descriptor_(std::move(descriptor)),
        options_(options) {
    if (loan_->size() != payload.descriptor_tensor_fields.size()) {
      throw RuntimeError{
          Error{ErrorCode::kProtocolError, "PYTHON_SCHEMA_TENSOR_LOAN_INVENTORY_MISMATCH"}};
    }
    guards_.reserve(loan_->size());
    tensors_.reserve(loan_->size());
    for (std::size_t index = 0U; index < loan_->size(); ++index) {
      auto writer = [&] {
        const nb::gil_scoped_release release;
        const MemoryKind kind = loan_->memory_kind(index);
        return kind == MemoryKind::kCudaDevice || kind == MemoryKind::kCudaManaged
                   ? loan_->write(index, ::cuda::stream_ref{stream})
                   : loan_->write_host(index);
      }();
      if (!writer) {
        throw RuntimeError{writer.error()};
      }
      guards_.push_back(std::make_unique<TensorOutputWriteGuard>(std::move(*writer)));
      tensors_.push_back(std::make_shared<Tensor>(guards_.back()->tensor()));
    }
  }

  ~PyTensorSchemaLoan() {
    invalidate_tensors();
    if (guards_.empty() && loan_ == nullptr) {
      return;
    }
    // Invalidate every escaped Tensor while Python still owns the GIL, then release it while the
    // native bundle follows its potentially blocking no-throw abandonment path.
    if (nb::is_alive()) {
      const nb::gil_scoped_release release;
      abandon();
      return;
    }
    abandon();
  }

  [[nodiscard]] nb::object tensors() const {
    ensure_live();
    nb::dict fields;
    for (std::size_t index = 0U; index < tensors_.size(); ++index) {
      fields[nb::str(payload_->descriptor_tensor_fields[index].name.data(),
                     payload_->descriptor_tensor_fields[index].name.size())] =
          nb::cast(tensors_[index]);
    }
    return nb::module_::import_("types").attr("MappingProxyType")(fields);
  }

  PyTensorSchemaLoan& enter() {
    ensure_live();
    if (entered_) {
      throw RuntimeError{
          Error{ErrorCode::kInvalidArgument, "PYTHON_TENSOR_SCHEMA_LOAN_ALREADY_ENTERED"}};
    }
    entered_ = true;
    return *this;
  }

  bool exit(nb::handle exception_type, nb::handle, nb::handle) {
    ensure_live();
    if (!exception_type.is_none()) {
      abandon();
      return false;
    }
    for (auto& guard : guards_) {
      auto committed = std::move(*guard).commit();
      if (!committed) {
        abandon();
        throw RuntimeError{committed.error()};
      }
    }
    invalidate_tensors();
    guards_.clear();
    entered_ = false;
    authoring::ForeignTensorSchemaLoan loan = std::move(*loan_);
    loan_.reset();
    expected<void, Error> emitted;
    {
      const nb::gil_scoped_release release;
      emitted =
          payload_->emit_schema_tensors(output_, std::move(loan), std::move(descriptor_), options_);
    }
    if (!emitted) {
      throw RuntimeError{emitted.error()};
    }
    return false;
  }

 private:
  void ensure_live() const {
    if (loan_ == nullptr || guards_.empty() || tensors_.empty()) {
      throw_consumed_loan_error();
    }
  }

  void invalidate_tensors() noexcept {
    for (const auto& tensor : tensors_) {
      if (tensor != nullptr) {
        *tensor = Tensor{};
      }
    }
    tensors_.clear();
  }

  void abandon() noexcept {
    invalidate_tensors();
    guards_.clear();
    loan_.reset();
    entered_ = false;
  }

  std::unique_ptr<authoring::ForeignTensorSchemaLoan> loan_;
  std::vector<std::unique_ptr<TensorOutputWriteGuard>> guards_;
  std::vector<std::shared_ptr<Tensor>> tensors_;
  const authoring::RegisteredPayloadType* payload_{};
  authoring::OpaquePortRef output_;
  authoring::ForeignSchemaValue descriptor_;
  EmitOptions options_{};
  bool entered_{};
};

std::shared_ptr<PyTensorLoan> PyNativePort::allocate_tensor(const std::vector<std::int64_t>& shape,
                                                            nb::handle dtype, nb::handle strides,
                                                            nb::handle options) const {
  const NativePortBinding& value = bound(PortDirection::kOutput);
  const EmitOptions native_options = python_emit_options(options).value();
  if (value.definition.payload->identity.kind != PayloadIdentityKind::kTensor ||
      value.definition.payload->allocate_tensor == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  std::optional<std::vector<std::int64_t>> copied_strides;
  if (!strides.is_none()) {
    copied_strides = nb::cast<std::vector<std::int64_t>>(strides);
  }
  const TensorByteStridesView stride_view =
      copied_strides ? TensorByteStridesView{*copied_strides} : TensorByteStridesView{std::nullopt};
  const TensorLoanRequest request{
      .shape = shape, .dtype = python_tensor_dtype(dtype), .byte_strides = stride_view};
  auto allocated = [&] {
    const nb::gil_scoped_release release;
    return value.definition.payload->allocate_tensor(required_port_reference(value), request);
  }();
  if (!allocated) {
    throw RuntimeError{allocated.error()};
  }
  const auto stream = bindings_->execution().cuda_stream().get();
  return std::make_shared<PyTensorLoan>(std::move(*allocated),
                                        *value.definition.payload,
                                        required_port_reference(value),
                                        stream,
                                        native_options);
}

std::shared_ptr<PyTensorSchemaLoan> PyNativePort::allocate_tensors(nb::handle descriptor,
                                                                   const nb::dict& requests,
                                                                   nb::handle options) const {
  const NativePortBinding& value = bound(PortDirection::kOutput);
  const EmitOptions native_options = python_emit_options(options).value();
  if (!value.definition.schema_type || value.definition.payload->validate_schema == nullptr ||
      value.definition.payload->allocate_schema_tensors == nullptr ||
      value.definition.payload->emit_schema_tensors == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotSupported, "AUTHORING_PAYLOAD_TYPE_UNSUPPORTED"}};
  }
  auto native_descriptor = python_schema_value(descriptor);
  auto valid_descriptor = value.definition.payload->validate_schema(native_descriptor);
  if (!valid_descriptor) {
    throw RuntimeError{valid_descriptor.error()};
  }
  std::vector<std::string> names;
  std::vector<authoring::ForeignTensorFieldLoanRequest> native_requests;
  names.reserve(requests.size());
  native_requests.reserve(requests.size());
  for (const auto& [name, request] : requests) {
    if (!nb::isinstance<nb::str>(name) || !nb::isinstance<PyTensorLoanRequest>(request)) {
      throw nb::type_error(
          "allocate_tensors requests must map field names to holoscan.tensor.LoanRequest");
    }
    names.push_back(nb::cast<std::string>(name));
    native_requests.push_back(authoring::ForeignTensorFieldLoanRequest{
        .field_name = names.back(),
        .tensor = nb::cast<const PyTensorLoanRequest&>(request).value(),
    });
  }
  auto allocated = [&] {
    const nb::gil_scoped_release release;
    return value.definition.payload->allocate_schema_tensors(required_port_reference(value),
                                                             native_requests);
  }();
  if (!allocated) {
    throw RuntimeError{allocated.error()};
  }
  const auto stream = bindings_->execution().cuda_stream().get();
  return std::make_shared<PyTensorSchemaLoan>(std::move(*allocated),
                                              *value.definition.payload,
                                              required_port_reference(value),
                                              std::move(native_descriptor),
                                              stream,
                                              native_options);
}

std::optional<NativePortDefinition> native_port_definition(nb::handle value,
                                                           std::string attribute_name) {
  if (nb::isinstance<PyInputDescriptor>(value)) {
    auto& descriptor = nb::cast<PyInputDescriptor&>(value);
    const nb::object schema_type = descriptor.schema_type();
    return NativePortDefinition{
        std::move(attribute_name),
        descriptor.name(),
        PortDirection::kInput,
        &descriptor.payload(),
        GilSafeObject{descriptor.view()},
        schema_type.is_none() ? GilSafeObject{} : GilSafeObject{schema_type}};
  }
  if (nb::isinstance<PyOutputDescriptor>(value)) {
    auto& descriptor = nb::cast<PyOutputDescriptor&>(value);
    const nb::object schema_type = descriptor.schema_type();
    return NativePortDefinition{
        std::move(attribute_name),
        descriptor.name(),
        PortDirection::kOutput,
        &descriptor.payload(),
        {},
        schema_type.is_none() ? GilSafeObject{} : GilSafeObject{schema_type}};
  }
  return std::nullopt;
}

std::shared_ptr<PyNativeOperatorBindings> make_native_operator_bindings(
    const std::vector<NativePortDefinition>& definitions, HostedParameterSnapshot parameters,
    ParameterByteSpanResolver parameter_byte_span_resolver,
    ParameterMemoryKindResolver parameter_memory_kind_resolver) {
  return std::make_shared<PyNativeOperatorBindings>(definitions,
                                                    std::move(parameters),
                                                    std::move(parameter_byte_span_resolver),
                                                    std::move(parameter_memory_kind_resolver));
}

// Resolves a Python Param to a byte-span expression only while its operator setup is active.
TensorByteSpanExpression active_parameter_byte_span(nb::handle parameter, bool shape) {
  PyNativeOperatorBindings* const bindings = active_setup_bindings();
  if (bindings == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotReady, "PYTHON_OPERATOR_SETUP_CONTEXT_REQUIRED"}};
  }
  return bindings->parameter_byte_span(parameter, shape);
}

TensorMemoryKindExpression active_parameter_memory_kind(nb::handle parameter, MemoryKind when_false,
                                                        MemoryKind when_true) {
  PyNativeOperatorBindings* const bindings = active_setup_bindings();
  if (bindings == nullptr) {
    throw RuntimeError{Error{ErrorCode::kNotReady, "PYTHON_OPERATOR_SETUP_CONTEXT_REQUIRED"}};
  }
  return bindings->parameter_memory_kind(parameter, when_false, when_true);
}

void attach_native_operator_bindings(nb::handle instance,
                                     const std::shared_ptr<PyNativeOperatorBindings>& bindings) {
  instance.attr("_holoscan_operator_bindings") = nb::cast(bindings);
}

void setup_native_operator(nb::handle callback, ForeignOperatorSpec& spec,
                           const std::shared_ptr<PyNativeOperatorBindings>& bindings,
                           PyLifecycleHookBindings& lifecycle) {
  const ScopedNativeSetupBindings setup_scope{*bindings};
  lifecycle.reset();
  PyNativeOperatorSpec python_spec{spec, bindings, lifecycle};
  try {
    if (!callback.is_none()) {
      callback(nb::cast(&python_spec, nb::rv_policy::reference));
    } else {
      for (const std::string& name : bindings->order()) {
        const NativePortBinding& binding = bindings->port(name);
        const PyNativePort port{bindings, name};
        if (binding.definition.direction == PortDirection::kInput) {
          (void)python_spec.input(port, std::nullopt, false);
        } else {
          (void)python_spec.output(port, std::nullopt);
        }
      }
    }
  } catch (...) {
    python_spec.invalidate();
    throw;
  }
  python_spec.invalidate();
}

void compute_native_operator(nb::handle callback, ExecutionContext& context,
                             const std::shared_ptr<PyNativeOperatorBindings>& bindings) {
  auto python_context = std::make_shared<PyExecutionContext>(context, bindings);
  try {
    callback(nb::cast(python_context));
  } catch (...) {
    python_context->invalidate();
    throw;
  }
  python_context->invalidate();
}

void init_native_io(nb::module_& module) {
  (void)python::bind_class_in_module<PyNativeOperatorBindings>(
      module, "holoscan._core", "_NativeOperatorBindings");
  auto native_port_type =
      python::bind_class_in_module<PyNativePort>(module, "holoscan", "OperatorPort");
  native_port_type.def_prop_ro("name", &PyNativePort::name)
      .def_prop_ro("direction", &PyNativePort::direction)
      .def_prop_ro("payload_type", &PyNativePort::payload_type)
      .def("selection", &PyNativePort::selection)
      .def("selected_batch", &PyNativePort::selected_batch)
      .def("receive", &PyNativePort::receive)
      .def("receive_data", &PyNativePort::receive_data)
      .def("emit",
           &PyNativePort::emit,
           nb::arg("data"),
           nb::kw_only(),
           nb::arg("options") = nb::none())
      .def(
          "forward",
          [](const PyNativePort& port, const PySample& sample, nb::handle options) {
            port.forward(sample, sample.data(), options);
          },
          nb::arg("sample"),
          nb::kw_only(),
          nb::arg("options") = nb::none(),
          "Forward one received sample without replacing its data.\n\n"
          "Every field absent from options inherits from the source sample metadata. Use emit() "
          "for a fused result whose absent provenance fields must remain absent.")
      .def("forward",
           &PyNativePort::forward,
           nb::arg("sample"),
           nb::arg("data"),
           nb::kw_only(),
           nb::arg("options") = nb::none(),
           "Forward one received sample while replacing its data.\n\n"
           "Every field absent from options inherits from the source sample metadata. Use emit() "
           "for a fused result whose absent provenance fields must remain absent.")
      .def("allocate_tensor",
           &PyNativePort::allocate_tensor,
           nb::kw_only(),
           nb::arg("shape"),
           nb::arg("dtype"),
           nb::arg("strides") = nb::none(),
           nb::arg("options") = nb::none())
      .def("allocate_tensors",
           &PyNativePort::allocate_tensors,
           nb::arg("descriptor"),
           nb::arg("requests"),
           nb::kw_only(),
           nb::arg("options") = nb::none(),
           "Allocate and publish every Tensor field of a descriptor-backed schema.\n\n"
           "The requests mapping must contain every canonical field name. The returned context "
           "manager exposes provider-owned Tensors through its read-only tensors mapping and "
           "publishes them atomically with descriptor metadata on successful exit. This API "
           "supports schemas with one or multiple Tensor fields.")
      .def("produces_tensor",
           &PyNativePort::produces_tensor,
           nb::arg("spec"),
           nb::arg("byte_span") = nb::none(),
           nb::kw_only(),
           nb::arg("memory_kind") = nb::none(),
           nb::arg("field") = nb::none(),
           nb::rv_policy::reference_internal)
      .def("expects_tensor",
           &PyNativePort::expects_tensor,
           nb::arg("spec"),
           nb::rv_policy::reference_internal)
      .def("accepts_projected_tensor",
           &PyNativePort::accepts_projected_tensor,
           nb::rv_policy::reference_internal,
           "Allow one direct, same-process descriptor-schema Tensor projection to connect to "
           "this Tensor input. Projection fanout and mixed full-schema/projected delivery are "
           "not supported.")
      .def("expected_rate",
           &PyNativePort::expected_rate,
           nb::arg("numerator_hz"),
           nb::arg("denominator_hz") = 1U,
           nb::rv_policy::reference_internal)
      .def("nominal_rate",
           &PyNativePort::nominal_rate,
           nb::arg("numerator_hz"),
           nb::arg("denominator_hz") = 1U,
           nb::rv_policy::reference_internal)
      .def("follows",
           nb::overload_cast<const PyNativePort&, std::uint32_t, std::uint32_t>(
               &PyNativePort::follows),
           nb::arg("input"),
           nb::kw_only(),
           nb::arg("numerator") = 1U,
           nb::arg("denominator") = 1U,
           nb::rv_policy::reference_internal)
      .def("follows",
           nb::overload_cast<const PyNativePort&, RateRatio>(&PyNativePort::follows),
           nb::arg("input"),
           nb::arg("ratio"),
           nb::rv_policy::reference_internal);
  auto native_spec_type =
      python::bind_class_in_module<PyNativeOperatorSpec>(module, "holoscan", "OperatorSpec");
  native_spec_type
      .def("input",
           &PyNativeOperatorSpec::input,
           nb::arg("port"),
           nb::kw_only(),
           nb::arg("queue_depth") = nb::none(),
           nb::arg("optional") = false,
           "Declare an input port; optional=True permits an unwired Exact/SafeDefault side input.")
      .def("output",
           &PyNativeOperatorSpec::output,
           nb::arg("port"),
           nb::kw_only(),
           nb::arg("max_emits_per_compute") = nb::none())
      .def("lifecycle", &PyNativeOperatorSpec::lifecycle);
  auto native_lifecycle_builder_type = python::bind_class_in_module<PyNativeLifecycleBuilder>(
      module, "holoscan.lifecycle", "OperatorLifecycleBuilder");
  native_lifecycle_builder_type
      .def("stage", &PyNativeLifecycleBuilder::stage, nb::arg("stage"), nb::arg("method"))
      .def("optional", &PyNativeLifecycleBuilder::optional, nb::arg("stage"), nb::arg("method"));
  auto on_violation_type =
      python::bind_enum_in_module<OnViolation>(module, "holoscan", "OnViolation");
  on_violation_type.value("SAFE_DEFAULT", OnViolation::kSafeDefault)
      .value("EMIT_WITNESS", OnViolation::kEmitWitness)
      .value("FAIL_RUN", OnViolation::kFailRun);
  auto activation_reason_type =
      python::bind_enum_in_module<ActivationReason>(module, "holoscan", "ActivationReason");
  activation_reason_type.value("NOMINAL", ActivationReason::kNominal)
      .value("DEADLINE_EXCEEDED", ActivationReason::kDeadlineExceeded)
      .value("INVARIANT_VIOLATED", ActivationReason::kInvariantViolated);
  auto activation_status_type =
      python::bind_class_in_module<ActivationStatus>(module, "holoscan", "ActivationStatus");
  activation_status_type
      .def(nb::init<ActivationReason, OnViolation>(),
           nb::kw_only(),
           nb::arg("reason") = ActivationReason::kNominal,
           nb::arg("policy") = OnViolation::kEmitWitness)
      .def_ro("reason", &ActivationStatus::reason)
      .def_ro("policy", &ActivationStatus::policy)
      .def(
          "__eq__",
          [](ActivationStatus lhs, ActivationStatus rhs) {
            return lhs.reason == rhs.reason && lhs.policy == rhs.policy;
          },
          nb::is_operator())
      .def("__hash__",
           [](ActivationStatus value) {
             return nb::hash(nb::make_tuple(value.reason, value.policy));
           })
      .def("__repr__", [](ActivationStatus value) {
        return "ActivationStatus(reason=" + python_repr(nb::cast(value.reason)) +
               ", policy=" + python_repr(nb::cast(value.policy)) + ")";
      });
  auto execution_context_type =
      python::bind_class_in_module<PyExecutionContext>(module, "holoscan", "ExecutionContext");
  execution_context_type
      .def_prop_ro("operator_path",
                   &PyExecutionContext::operator_path,
                   "Canonical percent-encoded path of the active operator callback.")
      .def("stream", &PyExecutionContext::stream, nb::sig("def stream(self) -> object"))
      .def("clock", &PyExecutionContext::clock)
      .def_prop_ro("tag_time", &PyExecutionContext::tag_time)
      .def_prop_ro("activation_time", &PyExecutionContext::activation_time)
      .def_prop_ro("trace_id", &PyExecutionContext::trace_id)
      .def_prop_ro("activation", &PyExecutionContext::activation)
      .def_prop_ro("stop_requested", &PyExecutionContext::stop_requested)
      .def("request_completion", &PyExecutionContext::request_completion)
      .def("request_stop", &PyExecutionContext::request_stop);
  auto cuda_stream_type =
      python::bind_class_in_module<PyCudaStream>(module, "holoscan._core", "_CudaStream");
  cuda_stream_type.def("__cuda_stream__", &PyCudaStream::cuda_stream)
      .def("__enter__", &PyCudaStream::enter, nb::rv_policy::reference_internal)
      .def("__exit__",
           &PyCudaStream::exit,
           nb::arg("exc_type").none(),
           nb::arg("exc_value").none(),
           nb::arg("traceback").none());
  module.def(
      "_test_cuda_stream_scope",
      [](std::uintptr_t pointer, int device) {
        return PyCudaStream{reinterpret_cast<cudaStream_t>(pointer), device};
      },
      nb::arg("pointer"),
      nb::arg("device") = 0);
  auto tensor_loan_type =
      python::bind_class_in_module<PyTensorLoan>(module, "holoscan.tensor", "TensorLoan");
  tensor_loan_type.def_prop_ro("tensor", &PyTensorLoan::tensor)
      .def("__enter__", &PyTensorLoan::enter, nb::rv_policy::reference_internal)
      .def("__exit__",
           &PyTensorLoan::exit,
           nb::arg("exc_type").none(),
           nb::arg("exc_value").none(),
           nb::arg("traceback").none());
  auto tensor_schema_loan_type = python::bind_class_in_module<PyTensorSchemaLoan>(
      module, "holoscan.tensor", "TensorSchemaLoan");
  tensor_schema_loan_type.def_prop_ro("tensors", &PyTensorSchemaLoan::tensors)
      .def("__enter__", &PyTensorSchemaLoan::enter, nb::rv_policy::reference_internal)
      .def("__exit__",
           &PyTensorSchemaLoan::exit,
           nb::arg("exc_type").none(),
           nb::arg("exc_value").none(),
           nb::arg("traceback").none());
  auto tensor_loan_request_type =
      python::bind_class_in_module<PyTensorLoanRequest>(module, "holoscan.tensor", "LoanRequest");
  tensor_loan_request_type
      .def(nb::init<std::vector<std::int64_t>, nb::handle, nb::handle>(),
           nb::kw_only(),
           nb::arg("shape"),
           nb::arg("dtype"),
           nb::arg("strides") = nb::none())
      .def_prop_ro("shape", &PyTensorLoanRequest::shape)
      .def_prop_ro("dtype", &PyTensorLoanRequest::dtype)
      .def_prop_ro("strides", &PyTensorLoanRequest::strides);
  auto sample_metadata_version_type = python::bind_class_in_module<SampleMetadataVersion>(
      module, "holoscan", "SampleMetadataVersion");
  sample_metadata_version_type.def_ro("major", &SampleMetadataVersion::major)
      .def_ro("minor", &SampleMetadataVersion::minor);
  auto sample_flags_type = python::bind_enum_in_module<SampleFlags>(
      module, "holoscan", "SampleFlags", nb::is_arithmetic(), nb::is_flag());
  sample_flags_type.value("NONE", SampleFlags::kNone)
      .value("INVALID", SampleFlags::kInvalid)
      .value("DEGRADED", SampleFlags::kDegraded);
  auto payload_kind_type =
      python::bind_enum_in_module<PayloadKind>(module, "holoscan", "PayloadKind");
  payload_kind_type.value("INLINE_TYPED", PayloadKind::kInlineTyped)
      .value("SERIALIZED_BYTES", PayloadKind::kSerializedBytes)
      .value("DESCRIPTOR_BACKED", PayloadKind::kDescriptorBacked)
      .value("COMPATIBILITY_PAYLOAD", PayloadKind::kCompatibilityPayload)
      .value("CONTROL_EMPTY", PayloadKind::kControlEmpty);
  auto sample_metadata_type =
      python::bind_class_in_module<SampleMetadata>(module, "holoscan", "SampleMetadata");
  sample_metadata_type.def_prop_ro("version",
                                   [](const SampleMetadata& value) { return value.version; });
  sample_metadata_type.def_ro("sequence", &SampleMetadata::sequence)
      .def_ro("capture_timestamp_ns", &SampleMetadata::capture_timestamp_ns)
      .def_prop_ro("capture_time", &SampleMetadata::capture_time)
      .def_prop_ro("source_clock_id",
                   [](const SampleMetadata& value) { return value.source_clock_id.value; })
      .def_ro("publish_timestamp_ns", &SampleMetadata::publish_timestamp_ns)
      .def_prop_ro("publish_time", &SampleMetadata::publish_time)
      .def_prop_ro("publish_clock_id",
                   [](const SampleMetadata& value) { return value.publish_clock_id.value; })
      .def_ro("trace_id", &SampleMetadata::trace_id)
      .def_ro("frame_id", &SampleMetadata::frame_id)
      .def_ro("byte_size", &SampleMetadata::byte_size)
      .def_ro("payload_kind", &SampleMetadata::payload_kind)
      .def_ro("flags", &SampleMetadata::flags)
      .def_prop_ro("valid", &SampleMetadata::valid)
      .def_prop_ro("degraded", &SampleMetadata::degraded);
  auto emit_options_type = python::bind_class_in_module<PyEmitOptions>(
      module,
      "holoscan",
      "EmitOptions",
      "Publication metadata overrides. Fields omitted from user-created options inherit from "
      "the source when passed to Output.forward(). Options returned by ProvenanceAccumulator "
      "are authoritative and can only be passed to Output.emit().");
  emit_options_type
      .def(nb::init<nb::handle, nb::handle, nb::handle, nb::handle>(),
           nb::kw_only(),
           nb::arg("capture_time") = nb::none(),
           nb::arg("frame_id") = nb::none(),
           nb::arg("trace_id") = nb::none(),
           nb::arg("flags") = nb::none())
      .def_prop_ro("capture_time", &PyEmitOptions::capture_time)
      .def_prop_ro("frame_id", &PyEmitOptions::frame_id)
      .def_prop_ro("trace_id", &PyEmitOptions::trace_id)
      .def_prop_ro("flags", &PyEmitOptions::flags)
      .def_prop_ro(
          "authoritative",
          &PyEmitOptions::authoritative,
          "Whether absent fields are authoritative rather than forwarding inheritance requests.")
      .def("__eq__", &PyEmitOptions::equals, nb::is_operator())
      .def("__hash__", &PyEmitOptions::hash)
      .def("__repr__", &PyEmitOptions::repr);
  auto scalar_sample_type =
      python::bind_class_in_module<PySample>(module, "holoscan", "Sample", nb::is_generic());
  scalar_sample_type.def_prop_ro("data", &PySample::data)
      .def_prop_ro("metadata", &PySample::metadata, nb::rv_policy::reference_internal)
      .def_prop_ro("is_stale", &PySample::is_stale)
      .def_prop_ro("is_missing", &PySample::is_missing);
  auto input_state_type = python::bind_enum_in_module<InputState>(module, "holoscan", "InputState");
  input_state_type.value("SELECTED", InputState::kSelected)
      .value("REUSED", InputState::kReused)
      .value("SAFE_DEFAULT", InputState::kSafeDefault)
      .value("ABSENT", InputState::kAbsent)
      .value("NOT_SELECTED", InputState::kNotSelected);
  auto selection_reason_type =
      python::bind_enum_in_module<SelectionReason>(module, "holoscan", "SelectionReason");
  selection_reason_type.value("NONE", SelectionReason::kNone)
      .value("NEVER_SEEN", SelectionReason::kNeverSeen)
      .value("TOO_STALE", SelectionReason::kTooStale)
      .value("INSUFFICIENT_HISTORY", SelectionReason::kInsufficientHistory)
      .value("WINDOW_TIMEOUT", SelectionReason::kWindowTimeout)
      .value("WINDOW_SAMPLE_LIMIT", SelectionReason::kWindowSampleLimit)
      .value("WINDOW_PENDING_LIMIT", SelectionReason::kWindowPendingLimit)
      .value("END_OF_STREAM", SelectionReason::kEndOfStream)
      .value("NOT_CONNECTED", SelectionReason::kNotConnected)
      .value("ALIGNMENT_WINDOW_MISS", SelectionReason::kAlignmentWindowMiss)
      .value("NOT_SELECTED_FOR_ACTIVATION", SelectionReason::kNotSelectedForActivation)
      .value("CAPTURE_TIME_ABSENT", SelectionReason::kCaptureTimeAbsent)
      .value("CLOCK_INCOMPARABLE", SelectionReason::kClockIncomparable)
      .value("CONVERSION_UNAVAILABLE", SelectionReason::kConversionUnavailable);
  auto input_selection_type =
      python::bind_class_in_module<InputSelection>(module, "holoscan", "InputSelection");
  input_selection_type
      .def(nb::init<InputState, SelectionReason>(),
           nb::kw_only(),
           nb::arg("state"),
           nb::arg("reason"))
      .def_ro("state", &InputSelection::state)
      .def_ro("reason", &InputSelection::reason)
      .def(
          "__eq__",
          [](InputSelection lhs, InputSelection rhs) {
            return lhs.state == rhs.state && lhs.reason == rhs.reason;
          },
          nb::is_operator())
      .def("__hash__",
           [](InputSelection value) { return nb::hash(nb::make_tuple(value.state, value.reason)); })
      .def("__repr__", [](InputSelection value) {
        return "InputSelection(state=" + python_repr(nb::cast(value.state)) +
               ", reason=" + python_repr(nb::cast(value.reason)) + ")";
      });
  auto selected_batch_iterator_type = python::bind_class_in_module<PySelectedBatchIterator>(
      module, "holoscan._core", "_SelectedBatchIterator");
  selected_batch_iterator_type
      .def("__iter__", &PySelectedBatchIterator::iter, nb::rv_policy::reference_internal)
      .def("__next__", &PySelectedBatchIterator::next);
  auto selected_batch_type = python::bind_class_in_module<PySelectedBatch>(
      module, "holoscan", "SelectedBatch", nb::is_generic());
  selected_batch_type.def("__len__", &PySelectedBatch::size)
      .def("__getitem__",
           [](const PySelectedBatch& batch, std::int64_t index) { return batch.item(index); })
      .def("__getitem__", &PySelectedBatch::items)
      .def("__iter__", &PySelectedBatch::iter);

  auto input_type = python::bind_class_in_module<PyInputDescriptor>(
      module, "holoscan", "Input", doc::graph::kInputDescriptor);
  auto input_descriptor_get = [](const PyInputDescriptor& descriptor,
                                 const nb::args& args) -> nb::object {
    if (args.size() != 2U) {
      throw nb::type_error("Input.__get__ expects an instance and owner");
    }
    const nb::handle instance = args[0];
    if (instance.is_none()) {
      return nb::cast(&descriptor, nb::rv_policy::reference);
    }
    if (nb::hasattr(instance, "_holoscan_operator_bindings")) {
      auto bindings = nb::cast<std::shared_ptr<PyNativeOperatorBindings>>(
          instance.attr("_holoscan_operator_bindings"));
      return nb::cast(PyNativePort{std::move(bindings), descriptor.name()});
    }
    if (nb::isinstance<PyNativeOperatorBase>(instance)) {
      PyErr_Format(PyExc_TypeError,
                   "Input '%s' is unavailable during hosted operator __init__; access it from "
                   "setup or a runtime callback",
                   descriptor.name().c_str());
      throw nb::python_error();
    }
    return python_graph_interface_port(
        instance, descriptor.name(), PortDirection::kInput, descriptor.payload());
  };
  input_type
      .def(
          "__init__",
          [](PyInputDescriptor* self, std::string name, nb::handle payload, nb::handle view) {
            const nb::object schema_type = nb::hasattr(payload, "_from_holoscan_schema")
                                               ? nb::borrow<nb::object>(payload)
                                               : nb::none();
            new (self) PyInputDescriptor(
                std::move(name), &python_payload_registration(payload), view, schema_type);
          },
          nb::arg("name"),
          nb::arg("payload_type"),
          nb::kw_only(),
          nb::arg("as_") = nb::none())
      .def_prop_ro("name", &PyInputDescriptor::name)
      .def_prop_ro("payload_type",
                   [](const PyInputDescriptor& value) { return PyPayloadType{&value.payload()}; })
      .def_prop_ro("as_", &PyInputDescriptor::view)
      .def("__get__",
           input_descriptor_get,
           nb::sig("def __get__(self, instance: None, owner: type[object], /) -> typing.Self"))
      .def("__get__",
           input_descriptor_get,
           nb::sig("def __get__(self, instance: holoscan.Graph, "
                   "owner: type[holoscan.Graph], /) -> holoscan.Port"))
      .def("__get__",
           input_descriptor_get,
           nb::sig("def __get__(self, instance: holoscan.Operator, "
                   "owner: type[holoscan.Operator], /) -> holoscan.OperatorPort"))
      .def("__set__", [](const PyInputDescriptor&, nb::handle, nb::handle) {
        PyErr_SetString(PyExc_AttributeError, "reusable Graph interface descriptors are read-only");
        throw nb::python_error();
      });

  auto output_type = python::bind_class_in_module<PyOutputDescriptor>(
      module, "holoscan", "Output", doc::graph::kOutputDescriptor);
  auto output_descriptor_get = [](const PyOutputDescriptor& descriptor,
                                  const nb::args& args) -> nb::object {
    if (args.size() != 2U) {
      throw nb::type_error("Output.__get__ expects an instance and owner");
    }
    const nb::handle instance = args[0];
    if (instance.is_none()) {
      return nb::cast(&descriptor, nb::rv_policy::reference);
    }
    if (nb::hasattr(instance, "_holoscan_operator_bindings")) {
      auto bindings = nb::cast<std::shared_ptr<PyNativeOperatorBindings>>(
          instance.attr("_holoscan_operator_bindings"));
      return nb::cast(PyNativePort{std::move(bindings), descriptor.name()});
    }
    if (nb::isinstance<PyNativeOperatorBase>(instance)) {
      PyErr_Format(PyExc_TypeError,
                   "Output '%s' is unavailable during hosted operator __init__; access it from "
                   "setup or a runtime callback",
                   descriptor.name().c_str());
      throw nb::python_error();
    }
    return python_graph_interface_port(
        instance, descriptor.name(), PortDirection::kOutput, descriptor.payload());
  };
  output_type
      .def(
          "__init__",
          [](PyOutputDescriptor* self, std::string name, nb::handle payload) {
            const nb::object schema_type = nb::hasattr(payload, "_from_holoscan_schema")
                                               ? nb::borrow<nb::object>(payload)
                                               : nb::none();
            new (self) PyOutputDescriptor(
                std::move(name), &python_payload_registration(payload), schema_type);
          },
          nb::arg("name"),
          nb::arg("payload_type"))
      .def_prop_ro("name", &PyOutputDescriptor::name)
      .def_prop_ro("payload_type",
                   [](const PyOutputDescriptor& value) { return PyPayloadType{&value.payload()}; })
      .def("__get__",
           output_descriptor_get,
           nb::sig("def __get__(self, instance: None, owner: type[object], /) -> typing.Self"))
      .def("__get__",
           output_descriptor_get,
           nb::sig("def __get__(self, instance: holoscan.Graph, "
                   "owner: type[holoscan.Graph], /) -> holoscan.Port"))
      .def("__get__",
           output_descriptor_get,
           nb::sig("def __get__(self, instance: holoscan.Operator, "
                   "owner: type[holoscan.Operator], /) -> holoscan.OperatorPort"))
      .def("__set__", [](const PyOutputDescriptor&, nb::handle, nb::handle) {
        PyErr_SetString(PyExc_AttributeError, "reusable Graph interface descriptors are read-only");
        throw nb::python_error();
      });
}

}  // namespace holoscan
