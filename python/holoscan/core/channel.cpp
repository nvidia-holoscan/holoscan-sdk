/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "channel.hpp"
#include "channel_tensor.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/authoring/channel_registry.hpp>
#include <holoscan/channel/channel.hpp>
#include <holoscan/channel/readiness_handle.hpp>
#include <holoscan/core/errors.hpp>

#include "../timeout.hpp"
#include "errors.hpp"
#include "graph.hpp"
#include "module_name.hpp"
#include "native_io.hpp"
#include "native_operator.hpp"
#include "payload.hpp"
#include "schema_value.hpp"
#include "time.hpp"

namespace holoscan {
namespace {

using authoring::OpaqueChannelPublisher;
using authoring::OpaqueChannelSubscriber;
using authoring::RegisteredPayloadType;

constexpr auto kConnectionPollInterval = std::chrono::milliseconds{5};

// Every caller passes an rvalue; keeping the sink parameter owning avoids copying the error into
// RuntimeError while preserving the single translation point.
[[noreturn]] void throw_channel_error(Error error) {  // NOLINT(performance-unnecessary-value-param)
  throw RuntimeError{std::move(error)};
}

// Unwraps exact pass-through Core results without obscuring their stable error identity.
template <typename T>
T value_or_throw(expected<T, Error> result) {
  if (!result) {
    throw_channel_error(std::move(result).error());
  }
  return std::move(*result);
}

void value_or_throw(expected<void, Error> result) {
  if (!result) {
    throw_channel_error(std::move(result).error());
  }
}

// Accepts Python seconds or the shared explicit-duration vocabulary at channel wait boundaries.
std::chrono::nanoseconds python_timeout(nb::handle value, std::string_view field) {
  if ((nb::isinstance<nb::float_>(value) || nb::isinstance<nb::int_>(value)) &&
      !nb::isinstance<nb::bool_>(value)) {
    const double seconds = PyFloat_AsDouble(value.ptr());
    if (PyErr_Occurred() != nullptr) {
      throw nb::python_error();
    }
    return python::detail::seconds_to_timeout(seconds);
  }
  return std::chrono::nanoseconds{
      python_duration_nanoseconds(value,
                                  field,
                                  "PYTHON_CHANNEL_DURATION_TYPE_MISMATCH",
                                  "PYTHON_CHANNEL_DURATION_OUT_OF_RANGE")};
}

std::optional<std::chrono::nanoseconds> optional_python_timeout(nb::handle value,
                                                                std::string_view field) {
  return value.is_none() ? std::nullopt
                         : std::optional<std::chrono::nanoseconds>{python_timeout(value, field)};
}

std::chrono::steady_clock::time_point deadline_after(std::chrono::nanoseconds timeout) noexcept {
  const auto now = std::chrono::steady_clock::now();
  const auto remaining = std::chrono::steady_clock::time_point::max() - now;
  return timeout >= remaining ? std::chrono::steady_clock::time_point::max() : now + timeout;
}

std::uint64_t python_uint64(nb::handle value, std::string_view field) {
  if (nb::isinstance<nb::bool_>(value)) {
    throw_channel_error(
        Error{ErrorCode::kInvalidArgument,
              "PYTHON_CHANNEL_UNSIGNED_INTEGER_REQUIRED field=" + std::string{field}});
  }
  PyObject* const indexed = PyNumber_Index(value.ptr());
  if (indexed == nullptr) {
    PyErr_Clear();
    throw_channel_error(
        Error{ErrorCode::kInvalidArgument,
              "PYTHON_CHANNEL_UNSIGNED_INTEGER_REQUIRED field=" + std::string{field}});
  }
  const nb::object owner = nb::steal<nb::object>(indexed);
  const unsigned long long converted =
      PyLong_AsUnsignedLongLong(owner.ptr());  // NOLINT(runtime/int)
  if (converted == static_cast<unsigned long long>(-1) && PyErr_Occurred() != nullptr) {
    PyErr_Clear();
    throw_channel_error(
        Error{ErrorCode::kInvalidArgument,
              "PYTHON_CHANNEL_UNSIGNED_INTEGER_OUT_OF_RANGE field=" + std::string{field}});
  }
  return static_cast<std::uint64_t>(converted);
}

EmitOptions python_emit_options(nb::handle capture_time, nb::handle frame_id, nb::handle flags) {
  EmitOptions options;
  if (!capture_time.is_none()) {
    options.capture_time = nb::cast<TimePoint>(capture_time);
  }
  if (!frame_id.is_none()) {
    options.frame_id = python_uint64(frame_id, "frame_id");
  }
  if (!flags.is_none()) {
    options.flags = nb::cast<SampleFlags>(flags);
  }
  return options;
}

QueuePolicy python_queue_policy(ChannelQueuePolicy policy) noexcept {
  return policy == ChannelQueuePolicy::kBlocking ? QueuePolicy::kBlocking
                                                 : QueuePolicy::kDropOldest;
}

ChannelQueuePolicy channel_queue_policy(QueuePolicy policy) noexcept {
  return policy == QueuePolicy::kBlocking ? ChannelQueuePolicy::kBlocking
                                          : ChannelQueuePolicy::kDropOldest;
}

// Provides the common runtime-owned retirement contract for publisher and subscriber adapters.
class PyChannelEndpoint {
 public:
  virtual ~PyChannelEndpoint() = default;
  virtual void retire_for_runtime(bool publish_eos) noexcept = 0;
  [[nodiscard]] virtual bool is_publisher() const noexcept = 0;
};

// Owns one Core runtime and coordinates deterministic publisher-before-subscriber retirement.
class PyChannelRuntimeState {
 public:
  explicit PyChannelRuntimeState(ChannelRuntime runtime)
      : runtime_(std::make_shared<ChannelRuntime>(std::move(runtime))) {}

  [[nodiscard]] std::shared_ptr<ChannelRuntime> runtime() const {
    const std::lock_guard lock(mutex_);
    if (closed_) {
      throw_channel_error(Error{ErrorCode::kClosed, "PYTHON_CHANNEL_RUNTIME_CLOSED"});
    }
    return runtime_;
  }

  [[nodiscard]] bool closed() const noexcept {
    const std::lock_guard lock(mutex_);
    return closed_;
  }

  void add(const std::shared_ptr<PyChannelEndpoint>& endpoint) {
    bool close_endpoint{};
    bool publish_eos{};
    {
      const std::lock_guard lock(mutex_);
      close_endpoint = closed_;
      publish_eos = publish_eos_;
      if (!closed_) {
        endpoints_.push_back(endpoint);
      }
    }
    if (close_endpoint) {
      endpoint->retire_for_runtime(publish_eos);
      throw_channel_error(Error{ErrorCode::kClosed, "PYTHON_CHANNEL_RUNTIME_CLOSED"});
    }
  }

  void close() noexcept { retire(true); }

  void abandon() noexcept { retire(false); }

 private:
  void retire(bool publish_eos) noexcept {
    std::vector<std::shared_ptr<PyChannelEndpoint>> endpoints;
    {
      const std::lock_guard lock(mutex_);
      if (closed_) {
        return;
      }
      closed_ = true;
      publish_eos_ = publish_eos;
      endpoints.reserve(endpoints_.size());
      for (const auto& endpoint : endpoints_) {
        if (auto retained = endpoint.lock()) {
          endpoints.push_back(std::move(retained));
        }
      }
      endpoints_.clear();
    }
    for (const auto& endpoint : endpoints) {
      if (endpoint->is_publisher()) {
        endpoint->retire_for_runtime(publish_eos);
      }
    }
    for (const auto& endpoint : endpoints) {
      if (!endpoint->is_publisher()) {
        endpoint->retire_for_runtime(publish_eos);
      }
    }
  }

  mutable std::mutex mutex_;
  std::shared_ptr<ChannelRuntime> runtime_;
  std::vector<std::weak_ptr<PyChannelEndpoint>> endpoints_;
  bool closed_{};
  bool publish_eos_{};
};

// Owns the stop source shared with one blocking subscriber call.
class PyCancelToken {
 public:
  void cancel() noexcept { (void)source_.request_stop(); }
  [[nodiscard]] bool cancelled() const noexcept { return source_.stop_requested(); }
  [[nodiscard]] std::stop_token token() const noexcept { return source_.get_token(); }

 private:
  std::stop_source source_;
};

// Serializes one subscriber's native operations and records its latched terminal state.
struct PySubscriberOperationState {
  std::mutex mutex;
  std::atomic<bool> wait_owned{};
  std::atomic<bool> remote_closed{};
};

// Owns a pollable native readiness claim and releases the subscriber's single-wait slot on detach.
class PyReadinessHandle {
 public:
  PyReadinessHandle(std::shared_ptr<PySubscriberOperationState> operations,
                    ChannelReadinessHandle handle)
      : operations_(std::move(operations)), handle_(std::move(handle)) {}

  PyReadinessHandle(const PyReadinessHandle&) = delete;
  PyReadinessHandle& operator=(const PyReadinessHandle&) = delete;

  ~PyReadinessHandle() { detach(); }

  [[nodiscard]] PyReadinessHandle& enter() noexcept { return *this; }
  void exit(nb::handle, nb::handle, nb::handle) noexcept { detach(); }

  [[nodiscard]] int fileno() const {
    const std::lock_guard lock(operations_->mutex);
    const int descriptor = handle_.descriptor();
    if (descriptor < 0) {
      throw_channel_error(Error{ErrorCode::kClosed, "PYTHON_CHANNEL_READINESS_DETACHED"});
    }
    return descriptor;
  }

  void acknowledge() {
    auto result = [&]() {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(operations_->mutex);
      return handle_.acknowledge();
    }();
    value_or_throw(std::move(result));
  }

  void detach() noexcept {
    const std::lock_guard lock(operations_->mutex);
    if (handle_.descriptor() >= 0) {
      handle_.detach();
      operations_->wait_owned.store(false);
    }
  }

 private:
  std::shared_ptr<PySubscriberOperationState> operations_;
  ChannelReadinessHandle handle_;
};

// Infers the registered payload row from immutable endpoint-catalog identity evidence.
const RegisteredPayloadType& payload_for_endpoint(ChannelEndpointCatalog::EndpointView endpoint) {
  if (endpoint.tensor_requirements()) {
    auto payload = authoring::authoring_registry().payload_type("holoscan.Tensor");
    if (!payload) {
      throw_channel_error(std::move(payload).error());
    }
    return payload->get();
  }
  PayloadIdentityRecord identity;
  if (const auto scalar = endpoint.scalar_identity()) {
    identity.kind = PayloadIdentityKind::kScalar;
    identity.scalar = ScalarIdentityRecord{.kind = *scalar};
  } else if (const auto schema = endpoint.schema_identity()) {
    const std::uint32_t transport_profile_tag =
        endpoint.schema_transport_profile_tag().value_or(0U);
    identity.kind = PayloadIdentityKind::kSchema;
    identity.payload_kind = transport_profile_tag == 0U ? PayloadKind::kSerializedBytes
                                                        : PayloadKind::kDescriptorBacked;
    identity.transport_profile_tag = transport_profile_tag;
    identity.schema = SchemaIdentityRecord{
        .canonical_name = std::string{schema->canonical_name()},
        .compatibility_epoch = schema->compatibility_epoch(),
        .family_id = schema->family_id(),
        .file_identifier = schema->file_identifier(),
    };
  } else if (endpoint.pod_type_identity()) {
    throw_channel_error(Error{ErrorCode::kNotSupported, "PYTHON_CHANNEL_POD_PAYLOAD_UNSUPPORTED"});
  } else {
    throw_channel_error(
        Error{ErrorCode::kIncompatible, "PYTHON_CHANNEL_ENDPOINT_PAYLOAD_IDENTITY_MISSING"});
  }

  auto payload = authoring::authoring_registry().payload_type(identity);
  if (!payload) {
    throw_channel_error(std::move(payload).error());
  }
  return payload->get();
}

nb::object schema_type_for(nb::handle payload_type) {
  return !payload_type.is_none() && nb::hasattr(payload_type, "_from_holoscan_schema")
             ? nb::borrow<nb::object>(payload_type)
             : nb::none();
}

nb::object inferred_schema_type(const RegisteredPayloadType& payload) {
  if (payload.identity.kind != PayloadIdentityKind::kSchema || !payload.identity.schema) {
    return nb::none();
  }
  const nb::object registry = nb::module_::import_("holoscan.payload.schema");
  return registry.attr("_resolve_type")(payload.identity.schema->canonical_name);
}

const RegisteredPayloadType& checked_catalog_payload(ChannelEndpointCatalog::EndpointView endpoint,
                                                     nb::handle payload_type) {
  const RegisteredPayloadType& inferred = payload_for_endpoint(endpoint);
  if (!payload_type.is_none()) {
    const RegisteredPayloadType& explicit_payload = python_payload_registration(payload_type);
    if (!authoring::payload_identities_equal(inferred.identity, explicit_payload.identity)) {
      throw_channel_error(
          Error{ErrorCode::kIncompatible, "PYTHON_CHANNEL_EXPLICIT_PAYLOAD_MISMATCH"});
    }
  }
  return inferred;
}

// Owns the erased publisher and converts Python values through its exact registered payload row.
class PyChannelPublisher : public PyChannelEndpoint {
 public:
  PyChannelPublisher(std::shared_ptr<PyChannelRuntimeState> state,
                     const RegisteredPayloadType* payload, OpaqueChannelPublisher publisher)
      : state_(std::move(state)), payload_(payload), publisher_(std::move(publisher)) {}

  [[nodiscard]] nb::object payload_type() const { return nb::cast(PyPayloadType{payload_}); }

  [[nodiscard]] nb::object loan(const std::vector<std::int64_t>& shape, nb::handle dtype,
                                nb::handle strides) {
    (void)state_->runtime();
    if (payload_->identity.kind != PayloadIdentityKind::kTensor) {
      throw_channel_error(
          Error{ErrorCode::kNotSupported, "PYTHON_CHANNEL_TENSOR_ENDPOINT_REQUIRED"});
    }
    return python::detail::channel_tensor_loan(
        *payload_, publisher_, mutex_, shape, dtype, strides);
  }

  std::uint32_t emit(nb::handle value, nb::handle capture_time, nb::handle frame_id,
                     nb::handle flags, nb::handle timeout) {
    (void)state_->runtime();
    EmitOptions options = python_emit_options(capture_time, frame_id, flags);
    const auto retry_timeout = optional_python_timeout(timeout, "timeout");
    const auto retry_deadline = retry_timeout
                                    ? std::optional{deadline_after(*retry_timeout)}
                                    : std::optional<std::chrono::steady_clock::time_point>{};
    if (payload_->identity.kind == PayloadIdentityKind::kTensor) {
      return python::detail::emit_channel_tensor(
          *payload_, publisher_, mutex_, value, std::move(options), retry_timeout, retry_deadline);
    }

    std::optional<ParamScalarValue> scalar;
    std::optional<authoring::ForeignSchemaValue> schema;
    if (payload_->identity.kind == PayloadIdentityKind::kScalar && payload_->identity.scalar) {
      scalar = payload_scalar_from_python(value, payload_->identity.scalar->kind);
    } else if (payload_->identity.kind == PayloadIdentityKind::kSchema) {
      schema = python_schema_value(value);
    } else {
      throw_channel_error(
          Error{ErrorCode::kNotSupported, "PYTHON_CHANNEL_PAYLOAD_TYPE_UNSUPPORTED"});
    }

    auto result = [&]() {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(mutex_);
      while (true) {
        auto attempted = scalar ? payload_->channel->emit_scalar(publisher_, *scalar, options)
                                : payload_->channel->emit_schema(publisher_, *schema, options);
        if (attempted || !retry_timeout ||
            (attempted.error().code != ErrorCode::kNotReady &&
             attempted.error().code != ErrorCode::kBackpressured)) {
          return attempted;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= *retry_deadline) {
          return expected<std::uint32_t, Error>{
              make_unexpected(Error{ErrorCode::kTimeout, "PYTHON_CHANNEL_EMIT_TIMEOUT"})};
        }
        std::this_thread::sleep_until(std::min(*retry_deadline, now + kConnectionPollInterval));
      }
    }();
    return value_or_throw(std::move(result));
  }

  std::uint32_t emit_captured(nb::handle value, std::int64_t capture_timestamp_ns,
                              nb::handle frame_id, nb::handle flags) {
    (void)state_->runtime();
    EmitOptions options = python_emit_options(nb::none(), frame_id, flags);
    if (payload_->identity.kind == PayloadIdentityKind::kTensor) {
      options.capture_time = make_capture_time(capture_timestamp_ns);
      return python::detail::emit_captured_channel_tensor(
          *payload_, publisher_, mutex_, value, std::move(options));
    }
    std::optional<ParamScalarValue> scalar;
    std::optional<authoring::ForeignSchemaValue> schema;
    if (payload_->identity.kind == PayloadIdentityKind::kScalar && payload_->identity.scalar) {
      scalar = payload_scalar_from_python(value, payload_->identity.scalar->kind);
    } else if (payload_->identity.kind == PayloadIdentityKind::kSchema) {
      schema = python_schema_value(value);
    } else {
      throw_channel_error(
          Error{ErrorCode::kNotSupported, "PYTHON_CHANNEL_PAYLOAD_TYPE_UNSUPPORTED"});
    }
    auto result = [&]() {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(mutex_);
      return scalar ? payload_->channel->emit_captured_scalar(
                          publisher_, *scalar, capture_timestamp_ns, options)
                    : payload_->channel->emit_captured_schema(
                          publisher_, *schema, capture_timestamp_ns, options);
    }();
    return value_or_throw(std::move(result));
  }

  [[nodiscard]] TimePoint make_capture_time(std::int64_t timestamp_ns) const {
    (void)state_->runtime();
    auto result = [&]() {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(mutex_);
      return payload_->channel->make_capture_time(publisher_, timestamp_ns);
    }();
    return value_or_throw(std::move(result));
  }

  void report_clock_health(ClockHealthReport report) {
    (void)state_->runtime();
    auto result = [&]() {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(mutex_);
      return payload_->channel->report_clock_health(publisher_, report);
    }();
    value_or_throw(std::move(result));
  }

  void update_connections() {
    (void)state_->runtime();
    auto result = [&]() {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(mutex_);
      return payload_->channel->update_connections(publisher_);
    }();
    value_or_throw(std::move(result));
  }

  [[nodiscard]] bool has_subscribers() noexcept {
    if (state_->closed()) {
      return false;
    }
    const nb::gil_scoped_release release;
    const std::lock_guard lock(mutex_);
    return payload_->channel->has_subscribers(publisher_);
  }

  void wait_for_subscribers(nb::handle timeout) {
    (void)state_->runtime();
    const auto wait_timeout = optional_python_timeout(timeout, "timeout");
    const auto deadline =
        wait_timeout ? deadline_after(*wait_timeout) : std::chrono::steady_clock::time_point::max();
    auto result = [&]() -> expected<void, Error> {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(mutex_);
      while (true) {
        auto refreshed = payload_->channel->update_connections(publisher_);
        if (!refreshed) {
          return refreshed;
        }
        if (payload_->channel->has_subscribers(publisher_)) {
          return {};
        }
        const auto now = std::chrono::steady_clock::now();
        if (now >= deadline) {
          return make_unexpected(
              Error{ErrorCode::kTimeout, "PYTHON_CHANNEL_SUBSCRIBER_WAIT_TIMEOUT"});
        }
        std::this_thread::sleep_until(std::min(deadline, now + kConnectionPollInterval));
      }
    }();
    value_or_throw(std::move(result));
  }

  void close() {
    (void)state_->runtime();
    auto result = [&]() {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(mutex_);
      return payload_->channel->publisher_close(publisher_);
    }();
    value_or_throw(std::move(result));
  }

  void retire_for_runtime(bool publish_eos) noexcept override {
    const std::lock_guard lock(mutex_);
    if (publish_eos) {
      (void)payload_->channel->publisher_close(publisher_);
    } else {
      publisher_ = OpaqueChannelPublisher{};
    }
  }

  [[nodiscard]] bool is_publisher() const noexcept override { return true; }

 private:
  std::shared_ptr<PyChannelRuntimeState> state_;
  const RegisteredPayloadType* payload_{};
  mutable std::mutex mutex_;
  OpaqueChannelPublisher publisher_;
};

// Owns the erased subscriber, its wait authority, and schema materialization identity.
class PyChannelSubscriber : public PyChannelEndpoint {
 public:
  PyChannelSubscriber(std::shared_ptr<PyChannelRuntimeState> state,
                      const RegisteredPayloadType* payload, OpaqueChannelSubscriber subscriber,
                      nb::handle schema_type)
      : state_(std::move(state)),
        payload_(payload),
        operations_(std::make_shared<PySubscriberOperationState>()),
        subscriber_(std::move(subscriber)),
        schema_type_(schema_type) {}

  [[nodiscard]] nb::object payload_type() const { return nb::cast(PyPayloadType{payload_}); }

  [[nodiscard]] nb::object try_receive() {
    (void)state_->runtime();
    require_not_remote_closed();
    if (payload_->identity.kind == PayloadIdentityKind::kScalar) {
      auto result = [&]() {
        const nb::gil_scoped_release release;
        const std::lock_guard lock(operations_->mutex);
        return payload_->channel->try_receive_scalar(subscriber_);
      }();
      if (!result && result.error().code == ErrorCode::kNotReady) {
        return nb::none();
      }
      latch_remote_close(result);
      return python_sample(value_or_throw(std::move(result)));
    }
    if (payload_->identity.kind == PayloadIdentityKind::kSchema) {
      auto result = [&]() {
        const nb::gil_scoped_release release;
        const std::lock_guard lock(operations_->mutex);
        return payload_->channel->try_receive_schema(subscriber_);
      }();
      if (!result && result.error().code == ErrorCode::kNotReady) {
        return nb::none();
      }
      latch_remote_close(result);
      return python_sample(value_or_throw(std::move(result)), schema_type_.borrow());
    }
    if (payload_->identity.kind == PayloadIdentityKind::kTensor) {
      auto result =
          python::detail::try_receive_channel_tensor(*payload_, subscriber_, operations_->mutex);
      if (!result && result.error().code == ErrorCode::kNotReady) {
        return nb::none();
      }
      latch_remote_close(result);
      return python_sample(value_or_throw(std::move(result)));
    }
    throw_channel_error(Error{ErrorCode::kNotSupported, "PYTHON_CHANNEL_PAYLOAD_TYPE_UNSUPPORTED"});
  }

  [[nodiscard]] nb::object receive(nb::handle timeout,
                                   const std::shared_ptr<PyCancelToken>& cancel) {
    (void)state_->runtime();
    require_not_remote_closed();
    bool expected = false;
    if (!operations_->wait_owned.compare_exchange_strong(expected, true)) {
      throw_channel_error(Error{ErrorCode::kInvalidArgument, "PYTHON_CHANNEL_WAIT_ALREADY_ACTIVE"});
    }
    struct WaitReset {
      std::atomic<bool>& active;
      ~WaitReset() { active.store(false); }
    };
    const WaitReset reset{operations_->wait_owned};
    const auto wait_timeout = optional_python_timeout(timeout, "timeout");
    const auto deadline =
        wait_timeout ? deadline_after(*wait_timeout) : std::chrono::steady_clock::time_point::max();
    const std::stop_token token = cancel ? cancel->token() : std::stop_token{};
    if (payload_->identity.kind == PayloadIdentityKind::kScalar) {
      auto result = [&]() {
        const nb::gil_scoped_release release;
        const std::lock_guard lock(operations_->mutex);
        return payload_->channel->wait_and_receive_scalar(subscriber_, deadline, token);
      }();
      latch_remote_close(result);
      return python_sample(value_or_throw(std::move(result)));
    }
    if (payload_->identity.kind == PayloadIdentityKind::kSchema) {
      auto result = [&]() {
        const nb::gil_scoped_release release;
        const std::lock_guard lock(operations_->mutex);
        return payload_->channel->wait_and_receive_schema(subscriber_, deadline, token);
      }();
      latch_remote_close(result);
      return python_sample(value_or_throw(std::move(result)), schema_type_.borrow());
    }
    if (payload_->identity.kind == PayloadIdentityKind::kTensor) {
      auto result = python::detail::receive_channel_tensor(
          *payload_, subscriber_, operations_->mutex, deadline, token);
      latch_remote_close(result);
      return python_sample(value_or_throw(std::move(result)));
    }
    throw_channel_error(Error{ErrorCode::kNotSupported, "PYTHON_CHANNEL_PAYLOAD_TYPE_UNSUPPORTED"});
  }

  void wait(nb::handle timeout, const std::shared_ptr<PyCancelToken>& cancel) {
    (void)state_->runtime();
    require_not_remote_closed();
    bool expected = false;
    if (!operations_->wait_owned.compare_exchange_strong(expected, true)) {
      throw_channel_error(Error{ErrorCode::kInvalidArgument, "PYTHON_CHANNEL_WAIT_ALREADY_ACTIVE"});
    }
    struct WaitReset {
      std::atomic<bool>& active;
      ~WaitReset() { active.store(false); }
    };
    const WaitReset reset{operations_->wait_owned};
    const auto wait_timeout = optional_python_timeout(timeout, "timeout");
    const auto deadline =
        wait_timeout ? deadline_after(*wait_timeout) : std::chrono::steady_clock::time_point::max();
    const std::stop_token token = cancel ? cancel->token() : std::stop_token{};
    auto result = [&]() {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(operations_->mutex);
      return payload_->channel->wait(subscriber_, deadline, token);
    }();
    latch_remote_close(result);
    value_or_throw(std::move(result));
  }

  [[nodiscard]] std::shared_ptr<PyReadinessHandle> readiness() {
    (void)state_->runtime();
    require_not_remote_closed();
    bool expected = false;
    if (!operations_->wait_owned.compare_exchange_strong(expected, true)) {
      throw_channel_error(Error{ErrorCode::kInvalidArgument, "PYTHON_CHANNEL_WAIT_ALREADY_ACTIVE"});
    }
    auto result = [&]() {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(operations_->mutex);
      return payload_->channel->subscriber_readiness(subscriber_);
    }();
    if (!result) {
      operations_->wait_owned.store(false);
      throw_channel_error(std::move(result).error());
    }
    return std::make_shared<PyReadinessHandle>(operations_, std::move(*result));
  }

  [[nodiscard]] bool has_samples() const {
    (void)state_->runtime();
    auto result = [&]() {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(operations_->mutex);
      return payload_->channel->has_samples(subscriber_);
    }();
    return value_or_throw(std::move(result));
  }

  [[nodiscard]] bool has_publishers() const noexcept {
    if (state_->closed()) {
      return false;
    }
    const nb::gil_scoped_release release;
    const std::lock_guard lock(operations_->mutex);
    return payload_->channel->has_publishers(subscriber_);
  }

  void refresh_connections() {
    (void)state_->runtime();
    auto result = [&]() {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(operations_->mutex);
      return payload_->channel->refresh_connections(subscriber_);
    }();
    value_or_throw(std::move(result));
  }

  void close() {
    (void)state_->runtime();
    auto result = [&]() {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(operations_->mutex);
      return payload_->channel->subscriber_close(subscriber_);
    }();
    value_or_throw(std::move(result));
  }

  void retire_for_runtime(bool) noexcept override {
    const std::lock_guard lock(operations_->mutex);
    (void)payload_->channel->subscriber_close(subscriber_);
  }

  [[nodiscard]] bool is_publisher() const noexcept override { return false; }

 private:
  void require_not_remote_closed() const {
    if (operations_->remote_closed.load()) {
      throw_channel_error(Error{ErrorCode::kClosed, "PYTHON_CHANNEL_REMOTE_CLOSED"});
    }
  }

  template <typename T>
  void latch_remote_close(const expected<T, Error>& result) noexcept {
    if (!result && result.error().code == ErrorCode::kClosed) {
      operations_->remote_closed.store(true);
    }
  }

  std::shared_ptr<PyChannelRuntimeState> state_;
  const RegisteredPayloadType* payload_{};
  std::shared_ptr<PySubscriberOperationState> operations_;
  OpaqueChannelSubscriber subscriber_;
  GilSafeObject schema_type_;
};

// Presents one graph-free ChannelRuntime and retains every endpoint created from it.
class PyChannelRuntime {
 public:
  PyChannelRuntime() {
    auto created = [] {
      const nb::gil_scoped_release release;
      return ChannelRuntime::create();
    }();
    if (!created) {
      throw_channel_error(std::move(created).error());
    }
    state_ = std::make_shared<PyChannelRuntimeState>(std::move(*created));
  }

  PyChannelRuntime(const PyChannelRuntime&) = delete;
  PyChannelRuntime& operator=(const PyChannelRuntime&) = delete;

  ~PyChannelRuntime() {
    if (state_) {
      if (nb::is_alive()) {
        const nb::gil_scoped_release release;
        state_->abandon();
      } else {
        state_->abandon();
      }
    }
  }

  [[nodiscard]] PyChannelRuntime& enter() noexcept { return *this; }
  void exit(nb::handle exc_type, nb::handle, nb::handle) {
    if (exc_type.is_none()) {
      close();
    } else {
      abandon();
    }
  }

  void close() {
    const nb::gil_scoped_release release;
    state_->close();
  }

  void abandon() {
    const nb::gil_scoped_release release;
    state_->abandon();
  }

  [[nodiscard]] std::shared_ptr<PyChannelPublisher> catalog_publisher(
      const ChannelEndpointCatalog& catalog, std::string_view endpoint_name, nb::handle timeout,
      nb::handle payload_type) {
    const auto endpoint = value_or_throw(catalog.endpoint(endpoint_name));
    const RegisteredPayloadType& payload = checked_catalog_payload(endpoint, payload_type);
    ChannelAttachOptions options{.timeout = optional_python_timeout(timeout, "timeout")};
    auto runtime = state_->runtime();
    auto publisher = [&]() {
      const nb::gil_scoped_release release;
      return payload.channel->create_publisher(*runtime, endpoint, std::move(options));
    }();
    auto result = std::make_shared<PyChannelPublisher>(
        state_, &payload, value_or_throw(std::move(publisher)));
    state_->add(result);
    return result;
  }

  [[nodiscard]] std::shared_ptr<PyChannelSubscriber> catalog_subscriber(
      const ChannelEndpointCatalog& catalog, std::string_view endpoint_name, nb::handle timeout,
      nb::handle payload_type) {
    const auto endpoint = value_or_throw(catalog.endpoint(endpoint_name));
    const RegisteredPayloadType& payload = checked_catalog_payload(endpoint, payload_type);
    nb::object schema_type = schema_type_for(payload_type);
    if (schema_type.is_none()) {
      schema_type = inferred_schema_type(payload);
    }
    ChannelAttachOptions options{.timeout = optional_python_timeout(timeout, "timeout")};
    auto runtime = state_->runtime();
    auto subscriber = [&]() {
      const nb::gil_scoped_release release;
      return payload.channel->create_subscriber(*runtime, endpoint, std::move(options));
    }();
    auto result = std::make_shared<PyChannelSubscriber>(
        state_, &payload, value_or_throw(std::move(subscriber)), schema_type);
    state_->add(result);
    return result;
  }

  [[nodiscard]] std::shared_ptr<PyChannelPublisher> name_publisher(
      std::string_view channel_name, nb::handle payload_type,
      std::optional<ChannelOptions> options) {
    const RegisteredPayloadType& payload = python_payload_registration(payload_type);
    auto runtime = state_->runtime();
    auto publisher = [&]() {
      const nb::gil_scoped_release release;
      return payload.channel->create_name_publisher(
          *runtime, channel_name, options.value_or(ChannelOptions{}));
    }();
    auto result = std::make_shared<PyChannelPublisher>(
        state_, &payload, value_or_throw(std::move(publisher)));
    state_->add(result);
    return result;
  }

  [[nodiscard]] std::shared_ptr<PyChannelSubscriber> name_subscriber(
      std::string_view channel_name, nb::handle payload_type,
      std::optional<ChannelOptions> options) {
    const RegisteredPayloadType& payload = python_payload_registration(payload_type);
    const nb::object schema_type = schema_type_for(payload_type);
    auto runtime = state_->runtime();
    auto subscriber = [&]() {
      const nb::gil_scoped_release release;
      return payload.channel->create_name_subscriber(
          *runtime, channel_name, options.value_or(ChannelOptions{}));
    }();
    auto result = std::make_shared<PyChannelSubscriber>(
        state_, &payload, value_or_throw(std::move(subscriber)), schema_type);
    state_->add(result);
    return result;
  }

 private:
  std::shared_ptr<PyChannelRuntimeState> state_;
};

}  // namespace

void init_channel(nb::module_& module) {
  module.def("_raise_channel_async_receive_timeout", [] {
    throw_channel_error(Error{ErrorCode::kTimeout, "PYTHON_CHANNEL_ASYNC_RECEIVE_TIMEOUT"});
  });

  auto transfer_mode = nb::enum_<ChannelTensorTransferMode>(module, "TensorTransferMode");
  transfer_mode.attr("__module__") = "holoscan.channel";
  transfer_mode.value("SAME_PROCESS_SIDECAR", ChannelTensorTransferMode::kSameProcessSidecar)
      .value("BUFFER_HANDLE", ChannelTensorTransferMode::kBufferHandle);

  auto channel_options = nb::class_<ChannelOptions>(module, "ChannelOptions");
  channel_options.attr("__module__") = "holoscan.channel";
  channel_options
      .def(
          "__init__",
          [](ChannelOptions* self,
             QueuePolicy queue_policy,
             std::uint32_t max_subscriber_queue_depth,
             std::uint32_t subscriber_queue_depth,
             std::uint32_t max_subscriber_borrowed_samples,
             std::uint32_t max_borrowed_samples,
             std::uint32_t publisher_history_depth,
             std::uint32_t max_loaned_samples,
             std::uint32_t expected_recipient_count,
             bool enable_event_wake,
             bool replacement_attribution_required,
             ChannelTensorTransferMode tensor_transfer_mode) {
            new (self) ChannelOptions{
                .queue_policy = channel_queue_policy(queue_policy),
                .max_subscriber_queue_depth = max_subscriber_queue_depth,
                .subscriber_queue_depth = subscriber_queue_depth,
                .max_subscriber_borrowed_samples = max_subscriber_borrowed_samples,
                .max_borrowed_samples = max_borrowed_samples,
                .publisher_history_depth = publisher_history_depth,
                .max_loaned_samples = max_loaned_samples,
                .expected_recipient_count = expected_recipient_count,
                .enable_event_wake = enable_event_wake,
                .replacement_attribution_required = replacement_attribution_required,
                .tensor_transfer_mode = tensor_transfer_mode,
            };
          },
          nb::kw_only(),
          nb::arg("queue_policy") = QueuePolicy::kBlocking,
          nb::arg("max_subscriber_queue_depth") = 0U,
          nb::arg("subscriber_queue_depth") = 8U,
          nb::arg("max_subscriber_borrowed_samples") = 0U,
          nb::arg("max_borrowed_samples") = 0U,
          nb::arg("publisher_history_depth") = 0U,
          nb::arg("max_loaned_samples") = 0U,
          nb::arg("expected_recipient_count") = 1U,
          nb::arg("enable_event_wake") = false,
          nb::arg("replacement_attribution_required") = false,
          nb::arg("tensor_transfer_mode") = ChannelTensorTransferMode::kSameProcessSidecar)
      .def_prop_rw(
          "queue_policy",
          [](const ChannelOptions& options) { return python_queue_policy(options.queue_policy); },
          [](ChannelOptions& options, QueuePolicy policy) {
            options.queue_policy = channel_queue_policy(policy);
          })
      .def_rw("max_subscriber_queue_depth", &ChannelOptions::max_subscriber_queue_depth)
      .def_rw("subscriber_queue_depth", &ChannelOptions::subscriber_queue_depth)
      .def_rw("max_subscriber_borrowed_samples", &ChannelOptions::max_subscriber_borrowed_samples)
      .def_rw("max_borrowed_samples", &ChannelOptions::max_borrowed_samples)
      .def_rw("publisher_history_depth", &ChannelOptions::publisher_history_depth)
      .def_rw("max_loaned_samples", &ChannelOptions::max_loaned_samples)
      .def_rw("expected_recipient_count", &ChannelOptions::expected_recipient_count)
      .def_rw("enable_event_wake", &ChannelOptions::enable_event_wake)
      .def_rw("replacement_attribution_required", &ChannelOptions::replacement_attribution_required)
      .def_rw("tensor_transfer_mode", &ChannelOptions::tensor_transfer_mode);

  auto assessment = nb::enum_<ClockHealthAssessment>(module, "ClockHealthAssessment");
  assessment.attr("__module__") = "holoscan.channel";
  assessment.value("HEALTHY", ClockHealthAssessment::kHealthy)
      .value("DEGRADED", ClockHealthAssessment::kDegraded)
      .value("UNHEALTHY", ClockHealthAssessment::kUnhealthy);
  auto reason = nb::enum_<ClockHealthReason>(module, "ClockHealthReason");
  reason.attr("__module__") = "holoscan.channel";
  reason.value("NONE", ClockHealthReason::kNone)
      .value("SYNCHRONIZATION_LOCK_LOST", ClockHealthReason::kSynchronizationLockLost)
      .value("PHASE_ERROR_LIMIT_EXCEEDED", ClockHealthReason::kPhaseErrorLimitExceeded)
      .value("RATE_ERROR_LIMIT_EXCEEDED", ClockHealthReason::kRateErrorLimitExceeded)
      .value("CALIBRATION_EXPIRED", ClockHealthReason::kCalibrationExpired)
      .value("CLOCK_STALLED", ClockHealthReason::kClockStalled)
      .value("TIMESTAMP_REGRESSION", ClockHealthReason::kTimestampRegression)
      .value("TIMESTAMP_SOURCE_FAULT", ClockHealthReason::kTimestampSourceFault)
      .value("MONITOR_FAULT", ClockHealthReason::kMonitorFault)
      .value("UNSPECIFIED", ClockHealthReason::kUnspecified);
  auto health_report = nb::class_<ClockHealthReport>(module, "ClockHealthReport");
  health_report.attr("__module__") = "holoscan.channel";
  health_report
      .def(
          "__init__",
          [](ClockHealthReport* self, ClockHealthAssessment assessment, ClockHealthReason reason) {
            new (self) ClockHealthReport{.assessment = assessment, .reason = reason};
          },
          nb::arg("assessment"),
          nb::arg("reason") = ClockHealthReason::kNone)
      .def_rw("assessment", &ClockHealthReport::assessment)
      .def_rw("reason", &ClockHealthReport::reason)
      .def_static("healthy",
                  &ClockHealthReport::healthy,
                  nb::sig("def healthy() -> holoscan.channel.ClockHealthReport"))
      .def_static("degraded",
                  &ClockHealthReport::degraded,
                  nb::arg("reason"),
                  nb::sig("def degraded(reason: holoscan.channel.ClockHealthReason) -> "
                          "holoscan.channel.ClockHealthReport"))
      .def_static("unhealthy",
                  &ClockHealthReport::unhealthy,
                  nb::arg("reason"),
                  nb::sig("def unhealthy(reason: holoscan.channel.ClockHealthReason) -> "
                          "holoscan.channel.ClockHealthReport"));

  auto cancel_token = nb::class_<PyCancelToken>(module, "CancelToken");
  cancel_token.attr("__module__") = "holoscan.channel";
  cancel_token.def(nb::init<>())
      .def("cancel", &PyCancelToken::cancel)
      .def_prop_ro("cancelled", &PyCancelToken::cancelled);

  auto readiness_handle = nb::class_<PyReadinessHandle>(module, "ReadinessHandle");
  readiness_handle.attr("__module__") = "holoscan.channel";
  readiness_handle.def("__enter__", &PyReadinessHandle::enter, nb::rv_policy::reference_internal)
      .def("__exit__",
           &PyReadinessHandle::exit,
           nb::arg("exc_type").none(),
           nb::arg("exc_value").none(),
           nb::arg("traceback").none())
      .def("fileno", &PyReadinessHandle::fileno)
      .def("acknowledge", &PyReadinessHandle::acknowledge)
      .def("detach", &PyReadinessHandle::detach);

  python::detail::init_channel_tensor(module);

  auto publisher = nb::class_<PyChannelPublisher>(module, "Publisher");
  publisher.attr("__module__") = "holoscan.channel";
  publisher
      .def("loan",
           &PyChannelPublisher::loan,
           nb::arg("shape"),
           nb::arg("dtype"),
           nb::kw_only(),
           nb::arg("strides").none() = nb::none(),
           nb::sig("def loan(self, shape: list[int] | tuple[int, ...], dtype: object, *, "
                   "strides: list[int] | tuple[int, ...] | None = None) -> "
                   "holoscan.channel.TensorLoan"))
      .def("emit",
           &PyChannelPublisher::emit,
           nb::arg("value"),
           nb::kw_only(),
           nb::arg("capture_time").none() = nb::none(),
           nb::arg("frame_id").none() = nb::none(),
           nb::arg("flags").none() = nb::none(),
           nb::arg("timeout").none() = nb::none())
      .def("emit_captured",
           &PyChannelPublisher::emit_captured,
           nb::arg("value"),
           nb::arg("capture_timestamp_ns"),
           nb::kw_only(),
           nb::arg("frame_id").none() = nb::none(),
           nb::arg("flags").none() = nb::none())
      .def("make_capture_time", &PyChannelPublisher::make_capture_time, nb::arg("timestamp_ns"))
      .def("report_clock_health", &PyChannelPublisher::report_clock_health, nb::arg("report"))
      .def("update_connections", &PyChannelPublisher::update_connections)
      .def("wait_for_subscribers",
           &PyChannelPublisher::wait_for_subscribers,
           nb::arg("timeout").none() = nb::none())
      .def_prop_ro("has_subscribers", &PyChannelPublisher::has_subscribers)
      .def_prop_ro("payload_type", &PyChannelPublisher::payload_type)
      .def("close", &PyChannelPublisher::close);

  auto subscriber = nb::class_<PyChannelSubscriber>(module, "Subscriber");
  subscriber.attr("__module__") = "holoscan.channel";
  subscriber.def("try_receive", &PyChannelSubscriber::try_receive)
      .def("receive",
           &PyChannelSubscriber::receive,
           nb::arg("timeout").none() = nb::none(),
           nb::kw_only(),
           nb::arg("cancel").none() = nb::none())
      .def("wait",
           &PyChannelSubscriber::wait,
           nb::arg("timeout").none() = nb::none(),
           nb::kw_only(),
           nb::arg("cancel").none() = nb::none())
      .def("readiness", &PyChannelSubscriber::readiness)
      .def_prop_ro("has_samples", &PyChannelSubscriber::has_samples)
      .def_prop_ro("has_publishers", &PyChannelSubscriber::has_publishers)
      .def_prop_ro("payload_type", &PyChannelSubscriber::payload_type)
      .def("refresh_connections", &PyChannelSubscriber::refresh_connections)
      .def("close", &PyChannelSubscriber::close);

  auto runtime = nb::class_<PyChannelRuntime>(module, "Runtime");
  runtime.attr("__module__") = "holoscan.channel";
  runtime.def(nb::init<>())
      .def("__enter__", &PyChannelRuntime::enter, nb::rv_policy::reference_internal)
      .def("__exit__",
           &PyChannelRuntime::exit,
           nb::arg("exc_type").none(),
           nb::arg("exc_value").none(),
           nb::arg("traceback").none())
      .def("close", &PyChannelRuntime::close)
      .def("publisher",
           &PyChannelRuntime::catalog_publisher,
           nb::arg("catalog"),
           nb::arg("name"),
           nb::kw_only(),
           nb::arg("timeout").none() = nb::none(),
           nb::arg("payload_type").none() = nb::none())
      .def("publisher",
           &PyChannelRuntime::name_publisher,
           nb::arg("channel_name"),
           nb::arg("payload_type"),
           nb::kw_only(),
           nb::arg("options").none() = nb::none())
      .def("subscriber",
           &PyChannelRuntime::catalog_subscriber,
           nb::arg("catalog"),
           nb::arg("name"),
           nb::kw_only(),
           nb::arg("timeout").none() = nb::none(),
           nb::arg("payload_type").none() = nb::none())
      .def("subscriber",
           &PyChannelRuntime::name_subscriber,
           nb::arg("channel_name"),
           nb::arg("payload_type"),
           nb::kw_only(),
           nb::arg("options").none() = nb::none());
}

}  // namespace holoscan
