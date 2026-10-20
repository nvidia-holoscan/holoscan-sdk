/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "channel_tensor.hpp"

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/core/domain/dlpack.hpp>
#include <holoscan/core/domain/dlpack_utils.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>

#include "dl_converter.hpp"
#include "integer_tuple.hpp"
#include "tensor_authoring.hpp"

namespace holoscan {
namespace {

constexpr auto kConnectionPollInterval = std::chrono::milliseconds{5};

// Every caller passes an rvalue; keeping the sink parameter owning avoids copying the error into
// RuntimeError while preserving the single translation point.
[[noreturn]] void throw_channel_error(Error error) {  // NOLINT(performance-unnecessary-value-param)
  throw RuntimeError{std::move(error)};
}

// Unwraps pass-through channel results while preserving their stable Core error identity.
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

// Normalizes CUDA stream protocol objects and the established CuPy pointer surface.
cudaStream_t python_cuda_stream(nb::handle value) {
  nb::object stream = nb::borrow<nb::object>(value);
  if (nb::hasattr(stream, "__cuda_stream__")) {
    const nb::object protocol = stream.attr("__cuda_stream__");
    if (PyCallable_Check(protocol.ptr()) == 0) {
      throw nb::type_error("__cuda_stream__ must be callable");
    }
    stream = protocol();
    if (PySequence_Size(stream.ptr()) != 2) {
      throw nb::value_error("__cuda_stream__ must return (version, stream_pointer)");
    }
    const nb::object version = nb::steal<nb::object>(PySequence_GetItem(stream.ptr(), 0));
    const nb::object pointer = nb::steal<nb::object>(PySequence_GetItem(stream.ptr(), 1));
    if (nb::cast<std::int64_t>(version) != 0) {
      throw nb::value_error("__cuda_stream__ protocol version must be zero");
    }
    stream = pointer;
  } else if (nb::hasattr(stream, "ptr")) {
    // CuPy Stream and ExternalStream expose the native stream through this property rather than
    // the emerging __cuda_stream__ protocol. Accept the same narrow pointer-shaped surface that
    // array libraries use when interoperating with an external CUDA producer.
    stream = stream.attr("ptr");
  }
  if (!nb::isinstance<nb::int_>(stream) || nb::isinstance<nb::bool_>(stream)) {
    throw nb::type_error("stream must be an integer pointer or expose __cuda_stream__ or ptr");
  }
  const std::int64_t pointer = nb::cast<std::int64_t>(stream);
  if (pointer < 0) {
    throw nb::value_error("CUDA stream pointer must be nonnegative");
  }
  if (pointer == 0 || pointer == 1) {
    return cudaStreamLegacy;
  }
  if (pointer == 2) {
    return cudaStreamPerThread;
  }
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast,performance-no-int-to-ptr)
  return reinterpret_cast<cudaStream_t>(static_cast<std::uintptr_t>(pointer));
}

cudaStream_t normalized_cuda_stream(cudaStream_t stream) noexcept {
  return stream == cudaStreamLegacy ? nullptr : stream;
}

bool requests_unsynchronized_dlpack(nb::handle stream) {
  return nb::isinstance<nb::int_>(stream) && !nb::isinstance<nb::bool_>(stream) &&
         nb::cast<std::int64_t>(stream) == -1;
}

nb::tuple integer_tuple(std::span<const std::int64_t> values, const std::int64_t scale = 1) {
  return python::detail::scaled_integer_tuple(values, scale);
}

DLDevice loan_device(MemoryKind memory_kind, DeviceId device_id) noexcept {
  switch (memory_kind) {
    case MemoryKind::kCudaDevice:
      return DLDevice{kDLCUDA, device_id.value};
    case MemoryKind::kPinnedHost:
      return DLDevice{kDLCUDAHost, 0};
    case MemoryKind::kCudaManaged:
      return DLDevice{kDLCUDAManaged, device_id.value};
    case MemoryKind::kHost:
    case MemoryKind::kUnknown:
    default:
      return DLDevice{kDLCPU, 0};
  }
}

// Tracks the one-way native-loan lifecycle across Python write, export, and emit operations.
enum class PyChannelTensorLoanState : std::uint8_t {
  kLoaned,
  kWriting,
  kReady,
  kFailed,
  kConsumed,
};

class PyChannelTensorLoan;
class PyChannelTensorArrayExport;

class PyTensorWritePhase;

// Owns one tracked host array-interface lease and keeps its native reservation alive.
class PyChannelTensorArrayExport {
 public:
  PyChannelTensorArrayExport(std::shared_ptr<PyChannelTensorLoan> owner, void* data,
                             std::vector<std::int64_t> shape,
                             std::optional<std::vector<std::int64_t>> byte_strides,
                             std::string typestr);
  ~PyChannelTensorArrayExport();

  PyChannelTensorArrayExport(const PyChannelTensorArrayExport&) = delete;
  PyChannelTensorArrayExport& operator=(const PyChannelTensorArrayExport&) = delete;

  [[nodiscard]] nb::dict array_interface() const;

 private:
  std::shared_ptr<PyChannelTensorLoan> owner_;
  void* data_{};
  std::vector<std::int64_t> shape_;
  std::optional<std::vector<std::int64_t>> byte_strides_;
  std::string typestr_;
};

// Owns one endpoint reservation and exposes writable views only during its explicit write phase.
class PyChannelTensorLoan : public std::enable_shared_from_this<PyChannelTensorLoan> {
 public:
  explicit PyChannelTensorLoan(TensorPublishLoan loan)
      : loan_(std::make_unique<TensorPublishLoan>(std::move(loan))),
        shape_(loan_->shape().begin(), loan_->shape().end()),
        dtype_(loan_->dtype()),
        device_(loan_device(loan_->memory_kind(), loan_->device_id())),
        memory_kind_(loan_->memory_kind()),
        byte_size_(loan_->byte_size()) {
    const TensorByteStridesView strides = loan_->byte_strides();
    if (strides) {
      byte_strides_.emplace(strides.begin(), strides.end());
    }
  }

  PyChannelTensorLoan(const PyChannelTensorLoan&) = delete;
  PyChannelTensorLoan& operator=(const PyChannelTensorLoan&) = delete;

  [[nodiscard]] std::shared_ptr<PyTensorWritePhase> write(nb::handle stream);

  void cancel() {
    auto cancelled = [&] {
      const nb::gil_scoped_release release;
      std::unique_ptr<TensorPublishLoan> loan;
      {
        const std::lock_guard lock(mutex_);
        if (exports_.load(std::memory_order_acquire) != 0U) {
          return expected<void, Error>{make_unexpected(
              Error{ErrorCode::kPermissionDenied, "PYTHON_TENSOR_LOAN_VIEW_STILL_EXPORTED"})};
        }
        if (loan_ == nullptr || state_ == PyChannelTensorLoanState::kConsumed) {
          return expected<void, Error>{
              make_unexpected(Error{ErrorCode::kInvalidArgument, "PYTHON_TENSOR_LOAN_CONSUMED"})};
        }
        writer_.reset();
        loan = std::move(loan_);
        state_ = PyChannelTensorLoanState::kConsumed;
      }
      return std::move(*loan).cancel();
    }();
    value_or_throw(std::move(cancelled));
  }

  [[nodiscard]] nb::tuple shape() const { return integer_tuple(shape_); }
  [[nodiscard]] DLDataType dtype() const noexcept { return dtype_; }
  [[nodiscard]] nb::object strides() const {
    return byte_strides_ ? nb::object{integer_tuple(*byte_strides_)} : nb::none();
  }
  [[nodiscard]] MemoryKind memory_kind() const noexcept { return memory_kind_; }
  [[nodiscard]] std::size_t byte_size() const noexcept { return byte_size_; }

  [[nodiscard]] nb::tuple dlpack_device() const {
    return nb::make_tuple(static_cast<int>(device_.device_type), device_.device_id);
  }

  [[nodiscard]] nb::object array(nb::handle dtype, nb::handle copy);

  [[nodiscard]] nb::capsule dlpack(nb::handle stream, nb::handle max_version, nb::handle dl_device,
                                   nb::handle copy);

  [[nodiscard]] TensorPublishLoan take() {
    auto result = [&]() -> expected<TensorPublishLoan, Error> {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(mutex_);
      if (exports_.load(std::memory_order_acquire) != 0U) {
        return make_unexpected(
            Error{ErrorCode::kPermissionDenied, "PYTHON_TENSOR_LOAN_VIEW_STILL_EXPORTED"});
      }
      if (state_ != PyChannelTensorLoanState::kReady || loan_ == nullptr || writer_ != nullptr) {
        return make_unexpected(Error{ErrorCode::kInvalidArgument, "PYTHON_TENSOR_LOAN_NOT_READY"});
      }
      state_ = PyChannelTensorLoanState::kConsumed;
      TensorPublishLoan taken = std::move(*loan_);
      loan_.reset();
      return taken;
    }();
    return value_or_throw(std::move(result));
  }

  void retain_export() noexcept { exports_.fetch_add(1U, std::memory_order_release); }
  void release_export() noexcept { exports_.fetch_sub(1U, std::memory_order_acq_rel); }

 private:
  friend class PyTensorWritePhase;
  friend class PyChannelTensorArrayExport;

  void begin_write(const std::optional<cudaStream_t> stream) {
    auto result = [&]() -> expected<void, Error> {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(mutex_);
      if (state_ != PyChannelTensorLoanState::kLoaned || loan_ == nullptr) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "PYTHON_TENSOR_LOAN_NOT_WRITABLE"});
      }
      auto writer = stream ? loan_->write(::cuda::stream_ref{*stream}) : loan_->write_host();
      if (!writer) {
        return make_unexpected(std::move(writer).error());
      }
      writer_ = std::make_unique<TensorLoanWriteGuard>(std::move(*writer));
      state_ = PyChannelTensorLoanState::kWriting;
      return {};
    }();
    value_or_throw(std::move(result));
  }

  void finish_write(const bool exception_raised) {
    auto result = [&]() -> expected<void, Error> {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(mutex_);
      if (state_ != PyChannelTensorLoanState::kWriting || writer_ == nullptr) {
        return make_unexpected(
            Error{ErrorCode::kInvalidArgument, "PYTHON_TENSOR_LOAN_WRITE_INACTIVE"});
      }
      if (exception_raised) {
        writer_.reset();
        state_ = PyChannelTensorLoanState::kFailed;
        return {};
      }
      if (exports_.load(std::memory_order_acquire) != 0U) {
        writer_.reset();
        state_ = PyChannelTensorLoanState::kFailed;
        return make_unexpected(
            Error{ErrorCode::kPermissionDenied, "PYTHON_TENSOR_LOAN_VIEW_STILL_EXPORTED"});
      }
      auto committed = std::move(*writer_).commit();
      writer_.reset();
      if (!committed) {
        state_ = PyChannelTensorLoanState::kFailed;
        return make_unexpected(std::move(committed).error());
      }
      state_ = PyChannelTensorLoanState::kReady;
      return {};
    }();
    value_or_throw(std::move(result));
  }

  void abandon_write() noexcept {
    const nb::gil_scoped_release release;
    const std::lock_guard lock(mutex_);
    if (state_ == PyChannelTensorLoanState::kWriting) {
      writer_.reset();
      state_ = PyChannelTensorLoanState::kFailed;
    }
  }

  [[nodiscard]] bool active_writer() const noexcept {
    return state_ == PyChannelTensorLoanState::kWriting && writer_ != nullptr &&
           writer_->data() != nullptr;
  }

  mutable std::mutex mutex_;
  std::unique_ptr<TensorPublishLoan> loan_;
  std::unique_ptr<TensorLoanWriteGuard> writer_;
  std::vector<std::int64_t> shape_;
  std::optional<std::vector<std::int64_t>> byte_strides_;
  DLDataType dtype_{};
  DLDevice device_{};
  MemoryKind memory_kind_{MemoryKind::kUnknown};
  std::size_t byte_size_{};
  std::atomic<std::uint32_t> exports_{};
  PyChannelTensorLoanState state_{PyChannelTensorLoanState::kLoaned};
};

// Presents a writable endpoint reservation through Tensor's canonical DLPack exporter. The shared
// reference keeps the write phase live until the consumer releases its capsule and carries the
// producer stream needed by Tensor's common consumer-stream synchronization path.
class PyChannelTensorLoanMemoryReference final : public TensorMemoryReference {
 public:
  PyChannelTensorLoanMemoryReference(std::shared_ptr<PyChannelTensorLoan> owner,
                                     std::optional<cudaStream_t> producer_stream)
      : owner_(std::move(owner)), producer_stream_(producer_stream) {
    owner_->retain_export();
  }

  ~PyChannelTensorLoanMemoryReference() override { owner_->release_export(); }

  [[nodiscard]] bool has_readiness_stream_contract() const noexcept override { return true; }
  [[nodiscard]] cudaStream_t readiness_stream() const noexcept override {
    return producer_stream_.value_or(nullptr);
  }
  [[nodiscard]] expected<void, Error> synchronize_for_consumer(
      cudaStream_t consumer_stream) override {
    if (producer_stream_ &&
        normalized_cuda_stream(*producer_stream_) != normalized_cuda_stream(consumer_stream)) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "PYTHON_TENSOR_LOAN_DLPACK_STREAM_MISMATCH"});
    }
    return {};
  }
  [[nodiscard]] expected<void, Error> synchronize_for_host() override {
    return producer_stream_ ? synchronize_cuda_stream_for_host(*producer_stream_)
                            : expected<void, Error>{};
  }

 private:
  std::shared_ptr<PyChannelTensorLoan> owner_;
  std::optional<cudaStream_t> producer_stream_;
};

nb::capsule PyChannelTensorLoan::dlpack(nb::handle stream, nb::handle max_version,
                                        nb::handle dl_device, nb::handle copy) {
  if ((memory_kind_ == MemoryKind::kCudaDevice || memory_kind_ == MemoryKind::kCudaManaged) &&
      requests_unsynchronized_dlpack(stream)) {
    throw std::runtime_error("PYTHON_TENSOR_LOAN_DLPACK_STREAM_UNTRACKED");
  }
  const std::optional<bool> copy_requested =
      copy.is_none() ? std::nullopt : std::optional<bool>{nb::cast<bool>(copy)};
  if (copy_requested.value_or(false)) {
    throw nb::buffer_error("copy=True is not supported for an endpoint-owned Tensor loan");
  }

  std::optional<std::tuple<int, int>> requested_device;
  if (!dl_device.is_none()) {
    const nb::tuple requested = nb::cast<nb::tuple>(dl_device);
    if (requested.size() != 2) {
      throw nb::buffer_error("dl_device must be a (device_type, device_id) pair");
    }
    requested_device = std::tuple{nb::cast<int>(requested[0]), nb::cast<int>(requested[1])};
    if (std::get<0>(*requested_device) != device_.device_type ||
        std::get<1>(*requested_device) != device_.device_id) {
      throw nb::buffer_error("a Tensor loan cannot be copied to another DLPack device");
    }
  }
  const std::optional<std::tuple<int, int>> requested_version =
      max_version.is_none()
          ? std::nullopt
          : std::optional<std::tuple<int, int>>{nb::cast<std::tuple<int, int>>(max_version)};

  auto loan_tensor = [&]() -> expected<Tensor, Error> {
    const nb::gil_scoped_release release;
    const std::lock_guard lock(mutex_);
    if (!active_writer()) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "PYTHON_TENSOR_LOAN_WRITE_INACTIVE"});
    }

    std::vector<std::int64_t> element_strides;
    if (byte_strides_) {
      const std::size_t item_size = dlpack_dtype_item_size_bytes(dtype_);
      element_strides.reserve(byte_strides_->size());
      for (const std::int64_t stride : *byte_strides_) {
        element_strides.push_back(stride / static_cast<std::int64_t>(item_size));
      }
    }
    const std::optional<::cuda::stream_ref> producer = writer_->stream();
    auto context = std::make_shared<DLManagedTensorContext>();
    context->memory_reference = std::make_shared<PyChannelTensorLoanMemoryReference>(
        shared_from_this(), producer ? std::optional<cudaStream_t>{producer->get()} : std::nullopt);
    const DLTensor loan_view{
        .data = writer_->data(),
        .device = device_,
        .ndim = static_cast<std::int32_t>(shape_.size()),
        .dtype = dtype_,
        .shape = shape_.empty() ? nullptr : shape_.data(),
        .strides = element_strides.empty() ? nullptr : element_strides.data(),
        .byte_offset = 0U,
    };
    copy_dl_tensor_with_owned_metadata(*context, loan_view);
    return Tensor{std::move(context)};
  }();
  Tensor tensor = value_or_throw(std::move(loan_tensor));
  return py_dlpack(
      &tensor, nb::borrow<nb::object>(stream), requested_version, requested_device, copy_requested);
}

PyChannelTensorArrayExport::PyChannelTensorArrayExport(
    std::shared_ptr<PyChannelTensorLoan> owner, void* data, std::vector<std::int64_t> shape,
    std::optional<std::vector<std::int64_t>> byte_strides, std::string typestr)
    : owner_(std::move(owner)),
      data_(data),
      shape_(std::move(shape)),
      byte_strides_(std::move(byte_strides)),
      typestr_(std::move(typestr)) {
  owner_->retain_export();
}

PyChannelTensorArrayExport::~PyChannelTensorArrayExport() {
  owner_->release_export();
}

nb::dict PyChannelTensorArrayExport::array_interface() const {
  nb::dict result;
  result["shape"] = integer_tuple(shape_);
  result["typestr"] = typestr_.c_str();
  result["data"] = nb::make_tuple(reinterpret_cast<std::uintptr_t>(data_), false);
  result["version"] = 3;
  result["strides"] = byte_strides_ ? nb::object{integer_tuple(*byte_strides_)} : nb::none();
  return result;
}

nb::object PyChannelTensorLoan::array(nb::handle dtype, nb::handle copy) {
  if (!dtype.is_none()) {
    throw nb::type_error("a writable Tensor loan view cannot change dtype");
  }
  if (!copy.is_none()) {
    const int copy_requested = PyObject_IsTrue(copy.ptr());
    if (copy_requested < 0) {
      throw nb::python_error();
    }
    if (copy_requested != 0) {
      throw nb::buffer_error("a writable Tensor loan view cannot be copied");
    }
  }
  const char* const typestr = value_or_throw(numpy_dtype(dtype_));
  auto exported = [&]() -> expected<std::unique_ptr<PyChannelTensorArrayExport>, Error> {
    const nb::gil_scoped_release release;
    const std::lock_guard lock(mutex_);
    if (!active_writer()) {
      return make_unexpected(
          Error{ErrorCode::kInvalidArgument, "PYTHON_TENSOR_LOAN_WRITE_INACTIVE"});
    }
    if (device_.device_type != kDLCPU && device_.device_type != kDLCUDAHost) {
      return make_unexpected(
          Error{ErrorCode::kNotSupported, "PYTHON_TENSOR_LOAN_HOST_ARRAY_REQUIRED"});
    }
    return std::make_unique<PyChannelTensorArrayExport>(
        shared_from_this(), writer_->data(), shape_, byte_strides_, typestr);
  }();
  auto exporter = value_or_throw(std::move(exported));
  const nb::object python_exporter = nb::cast(exporter.release(), nb::rv_policy::take_ownership);
  return nb::module_::import_("numpy").attr("asarray")(python_exporter);
}

// Commits or abandons a loan writer at the Python context-manager boundary.
class PyTensorWritePhase {
 public:
  PyTensorWritePhase(std::shared_ptr<PyChannelTensorLoan> loan, std::optional<cudaStream_t> stream)
      : loan_(std::move(loan)), stream_(stream) {}

  ~PyTensorWritePhase() {
    if (entered_) {
      loan_->abandon_write();
    }
  }

  [[nodiscard]] std::shared_ptr<PyChannelTensorLoan> enter() {
    if (entered_) {
      throw_channel_error(
          Error{ErrorCode::kInvalidArgument, "PYTHON_TENSOR_LOAN_WRITE_ALREADY_ENTERED"});
    }
    loan_->begin_write(stream_);
    entered_ = true;
    return loan_;
  }

  bool exit(nb::handle exception_type, nb::handle, nb::handle) {
    if (!entered_) {
      throw_channel_error(Error{ErrorCode::kInvalidArgument, "PYTHON_TENSOR_LOAN_WRITE_INACTIVE"});
    }
    entered_ = false;
    loan_->finish_write(!exception_type.is_none());
    return false;
  }

 private:
  std::shared_ptr<PyChannelTensorLoan> loan_;
  std::optional<cudaStream_t> stream_;
  bool entered_{};
};

std::shared_ptr<PyTensorWritePhase> PyChannelTensorLoan::write(nb::handle stream) {
  if (memory_kind_ == MemoryKind::kCudaDevice || memory_kind_ == MemoryKind::kCudaManaged) {
    if (stream.is_none()) {
      throw_channel_error(
          Error{ErrorCode::kInvalidArgument, "PYTHON_TENSOR_LOAN_CUDA_STREAM_REQUIRED"});
    }
    auto* const writer_stream = python_cuda_stream(stream);
    // CUDA assigns the same sentinel value to a distinct default stream on each host thread, so
    // pointer equality cannot prove that later DLPack writes use the stream committed here.
    if (writer_stream == cudaStreamPerThread) {
      throw_channel_error(
          Error{ErrorCode::kInvalidArgument, "PYTHON_TENSOR_LOAN_THREAD_STREAM_UNSUPPORTED"});
    }
    return std::make_shared<PyTensorWritePhase>(shared_from_this(), writer_stream);
  }
  if (!stream.is_none()) {
    throw_channel_error(
        Error{ErrorCode::kInvalidArgument, "PYTHON_TENSOR_LOAN_HOST_STREAM_FORBIDDEN"});
  }
  return std::make_shared<PyTensorWritePhase>(shared_from_this(), std::nullopt);
}

// Establishes consumer-stream readiness before the runtime copies a Python-authored Tensor.
cudaStream_t prepare_tensor_copy(const Tensor& tensor) {
  const DLDevice device = tensor.device();
  if (device.device_type == kDLCPU || device.device_type == kDLCUDAHost) {
    return nullptr;
  }
  if (device.device_type != kDLCUDA && device.device_type != kDLCUDAManaged) {
    throw_channel_error(
        Error{ErrorCode::kNotSupported, "PYTHON_CHANNEL_TENSOR_DEVICE_UNSUPPORTED"});
  }
  const auto stream = cudaStreamPerThread;
  auto synchronized = [&] {
    const nb::gil_scoped_release release;
    return tensor.synchronize_for_consumer_stream(stream);
  }();
  value_or_throw(std::move(synchronized));
  return stream;
}

}  // namespace

namespace python::detail {

nb::object channel_tensor_loan(const authoring::RegisteredPayloadType& payload,
                               authoring::OpaqueChannelPublisher& publisher, std::mutex& mutex,
                               const std::vector<std::int64_t>& shape, nb::handle dtype,
                               nb::handle strides) {
  std::optional<std::vector<std::int64_t>> copied_strides;
  if (!strides.is_none()) {
    copied_strides = nb::cast<std::vector<std::int64_t>>(strides);
  }
  const TensorLoanRequest request{
      .shape = shape,
      .dtype = python_tensor_dtype(dtype),
      .byte_strides = copied_strides ? TensorByteStridesView{*copied_strides}
                                     : TensorByteStridesView{std::nullopt},
  };
  auto result = [&] {
    const nb::gil_scoped_release release;
    const std::lock_guard lock(mutex);
    return payload.channel->loan_tensor(publisher, request);
  }();
  return nb::cast(std::make_shared<PyChannelTensorLoan>(value_or_throw(std::move(result))));
}

std::uint32_t emit_channel_tensor(
    const authoring::RegisteredPayloadType& payload, authoring::OpaqueChannelPublisher& publisher,
    std::mutex& mutex, nb::handle value, EmitOptions options,
    std::optional<std::chrono::nanoseconds> retry_timeout,
    std::optional<std::chrono::steady_clock::time_point> retry_deadline) {
  if (nb::isinstance<PyChannelTensorLoan>(value)) {
    if (retry_timeout) {
      throw_channel_error(
          Error{ErrorCode::kInvalidArgument, "PYTHON_TENSOR_LOAN_TIMEOUT_UNSUPPORTED"});
    }
    TensorPublishLoan loan = nb::cast<std::shared_ptr<PyChannelTensorLoan>>(value)->take();
    auto result = [&] {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(mutex);
      return payload.channel->emit_tensor_loan(publisher, std::move(loan), std::move(options));
    }();
    return value_or_throw(std::move(result));
  }

  const std::shared_ptr<Tensor> tensor = as_tensor(nb::borrow<nb::object>(value));
  const auto producer_stream = prepare_tensor_copy(*tensor);
  auto result = [&]() {
    const nb::gil_scoped_release release;
    const std::lock_guard lock(mutex);
    while (true) {
      auto attempted = payload.channel->emit_tensor_copy(
          publisher, *tensor, ::cuda::stream_ref{producer_stream}, options);
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

std::uint32_t emit_captured_channel_tensor(const authoring::RegisteredPayloadType& payload,
                                           authoring::OpaqueChannelPublisher& publisher,
                                           std::mutex& mutex, nb::handle value,
                                           EmitOptions options) {
  if (nb::isinstance<PyChannelTensorLoan>(value)) {
    TensorPublishLoan loan = nb::cast<std::shared_ptr<PyChannelTensorLoan>>(value)->take();
    auto result = [&] {
      const nb::gil_scoped_release release;
      const std::lock_guard lock(mutex);
      return payload.channel->emit_tensor_loan(publisher, std::move(loan), std::move(options));
    }();
    return value_or_throw(std::move(result));
  }

  const std::shared_ptr<Tensor> tensor = as_tensor(nb::borrow<nb::object>(value));
  const auto producer_stream = prepare_tensor_copy(*tensor);
  auto result = [&] {
    const nb::gil_scoped_release release;
    const std::lock_guard lock(mutex);
    return payload.channel->emit_tensor_copy(
        publisher, *tensor, ::cuda::stream_ref{producer_stream}, std::move(options));
  }();
  return value_or_throw(std::move(result));
}

expected<authoring::ForeignTensorSample, Error> try_receive_channel_tensor(
    const authoring::RegisteredPayloadType& payload, authoring::OpaqueChannelSubscriber& subscriber,
    std::mutex& mutex) {
  const nb::gil_scoped_release release;
  const std::lock_guard lock(mutex);
  return payload.channel->try_receive_tensor(subscriber);
}

expected<authoring::ForeignTensorSample, Error> receive_channel_tensor(
    const authoring::RegisteredPayloadType& payload, authoring::OpaqueChannelSubscriber& subscriber,
    std::mutex& mutex, std::chrono::steady_clock::time_point deadline,
    const std::stop_token& token) {
  const nb::gil_scoped_release release;
  const std::lock_guard lock(mutex);
  return payload.channel->wait_and_receive_tensor(subscriber, deadline, token);
}

void init_channel_tensor(nb::module_& module) {
  auto tensor_array_export =
      nb::class_<PyChannelTensorArrayExport>(module, "_TensorLoanArrayExport");
  tensor_array_export.attr("__module__") = "holoscan.channel";
  tensor_array_export.def_prop_ro("__array_interface__",
                                  &PyChannelTensorArrayExport::array_interface);

  auto tensor_write_phase = nb::class_<PyTensorWritePhase>(module, "_TensorWritePhase");
  tensor_write_phase.attr("__module__") = "holoscan.channel";
  tensor_write_phase.def("__enter__", &PyTensorWritePhase::enter)
      .def("__exit__",
           &PyTensorWritePhase::exit,
           nb::arg("exc_type").none(),
           nb::arg("exc_value").none(),
           nb::arg("traceback").none());

  auto tensor_loan = nb::class_<PyChannelTensorLoan>(
      module,
      "_ChannelTensorLoan",
      "One-shot endpoint-owned Tensor loan. Every emit attempt consumes the loan, including "
      "attempts that return an error.");
  tensor_loan.attr("__module__") = "holoscan.channel";
  tensor_loan.attr("__name__") = "TensorLoan";
  tensor_loan.attr("__qualname__") = "TensorLoan";
  tensor_loan
      .def("write",
           &PyChannelTensorLoan::write,
           nb::kw_only(),
           nb::arg("stream").none() = nb::none(),
           "Enter the only writable phase. CUDA storage requires a stream whose identity is "
           "stable across host threads, so stream=2 is rejected. Retain any stream owner "
           "through context exit.")
      .def("cancel", &PyChannelTensorLoan::cancel)
      .def_prop_ro("shape", &PyChannelTensorLoan::shape)
      .def_prop_ro("dtype", &PyChannelTensorLoan::dtype)
      .def_prop_ro("strides", &PyChannelTensorLoan::strides)
      .def_prop_ro("memory_kind", &PyChannelTensorLoan::memory_kind)
      .def_prop_ro("nbytes", &PyChannelTensorLoan::byte_size)
      .def("__array__",
           &PyChannelTensorLoan::array,
           nb::arg("dtype").none() = nb::none(),
           nb::kw_only(),
           nb::arg("copy").none() = nb::none())
      .def("__dlpack__",
           &PyChannelTensorLoan::dlpack,
           nb::kw_only(),
           nb::arg("stream").none() = nb::none(),
           nb::arg("max_version").none() = nb::none(),
           nb::arg("dl_device").none() = nb::none(),
           nb::arg("copy").none() = nb::none(),
           "Export a writable DLPack view. CUDA consumers must use the write phase's "
           "normalized producer stream; cross-stream and stream=-1 exports are rejected.")
      .def("__dlpack_device__", &PyChannelTensorLoan::dlpack_device);
}

}  // namespace python::detail
}  // namespace holoscan
