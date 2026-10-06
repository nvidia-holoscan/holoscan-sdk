/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime_api.h>
#include <algorithm>
#include <holoscan/testing/operator_test_harness.hpp>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include "core/domain/tensor_access.hpp"

namespace holoscan::testing::detail {
namespace {
MemoryKind memory_kind(DLDeviceType device) {
  switch (device) {
    case kDLCPU:
      return MemoryKind::kHost;
    case kDLCUDAHost:
      return MemoryKind::kPinnedHost;
    case kDLCUDA:
      return MemoryKind::kCudaDevice;
    case kDLCUDAManaged:
      return MemoryKind::kCudaManaged;
    default:
      throw std::invalid_argument{"operator test Tensor storage type is unsupported"};
  }
}
}  // namespace
SourceTensorSpec derive_tensor_source_spec(std::span<const Tensor* const> values) {
  if (values.empty())
    throw std::invalid_argument{"operator test cannot infer an empty Tensor contract"};
  const auto require_tensor = [](const Tensor* value) -> const Tensor& {
    if (value == nullptr) {
      throw std::invalid_argument{"operator test staging requires an initialized Tensor"};
    }
    const auto layout = holoscan::detail::TensorAccess::snapshot(*value);
    if (!layout.valid || (layout.byte_size != 0U &&
                          (!value->is_contiguous() || value->nbytes() <= 0 ||
                           static_cast<std::uint64_t>(value->nbytes()) != layout.byte_size ||
                           value->data() == nullptr))) {
      throw std::invalid_argument{
          "operator test staging requires a valid contiguous Tensor layout"};
    }
    return *value;
  };
  const auto& first = require_tensor(values.front());
  const auto device = first.device();
  const auto dtype = first.dtype();
  const auto rank = static_cast<std::uint32_t>(first.shape_span().size());
  SourceTensorSpec result;
  result.maximum_shape.assign(rank, 0);
  result.value.representation.memory_kind = memory_kind(device.device_type);
  result.value.representation.dtype = dtype;
  result.value.representation.rank = rank;
  result.value.representation.strides = TensorStridePolicy::kContiguousOnly;
  std::uint64_t maximum_bytes{};
  for (const auto* value : values) {
    const auto& current = require_tensor(value);
    const auto placement = current.device();
    const auto current_dtype = current.dtype();
    if (device.device_type != placement.device_type || device.device_id != placement.device_id ||
        current.shape_span().size() != rank || dtype.code != current_dtype.code ||
        dtype.bits != current_dtype.bits || dtype.lanes != current_dtype.lanes) {
      throw std::invalid_argument{"operator test staged Tensor representations must agree"};
    }
    const auto shape = value->shape_span();
    for (std::size_t axis = 0U; axis < shape.size(); ++axis) {
      result.maximum_shape[axis] = std::max(result.maximum_shape[axis], shape[axis]);
    }
    maximum_bytes = std::max(maximum_bytes, static_cast<std::uint64_t>(value->nbytes()));
  }
  auto bounding_bytes = static_cast<std::uint64_t>(first.itemsize());
  for (const auto extent : result.maximum_shape) {
    // Empty logical dimensions still need positive physical capacity in the native pool.
    const auto physical_extent = static_cast<std::uint64_t>(std::max(std::int64_t{1}, extent));
    if (bounding_bytes > std::numeric_limits<std::uint64_t>::max() / physical_extent) {
      throw std::invalid_argument{"operator test Tensor bounding shape overflows byte capacity"};
    }
    bounding_bytes *= physical_extent;
  }
  result.value.bounds = tensor_bounds(std::max(maximum_bytes, bounding_bytes));
  result.value.storage = TensorOutputStorage::kRuntimePool;
  return result;
}

std::shared_ptr<const void> acquire_staged_tensor_leases(std::span<const Tensor* const> tensors) {
  auto leases = std::make_shared<std::vector<holoscan::detail::TensorReadLease>>();
  leases->reserve(tensors.size());
  for (const auto* tensor : tensors) {
    if (tensor == nullptr)
      throw std::invalid_argument{"operator test staging requires an initialized Tensor"};
    const auto layout = holoscan::detail::TensorAccess::snapshot(*tensor);
    if (!layout.valid || (layout.byte_size != 0U &&
                          (tensor->nbytes() <= 0 ||
                           static_cast<std::uint64_t>(tensor->nbytes()) != layout.byte_size ||
                           tensor->data() == nullptr)))
      throw std::invalid_argument{"operator test staging requires a valid Tensor layout"};
    // A null-backed empty Tensor has metadata but no bytes to claim. The script owns that metadata.
    if (layout.byte_span == 0U && layout.data == nullptr)
      continue;
    auto lease = holoscan::detail::TensorAccess::acquire_read_lease(*tensor);
    if (!lease)
      throw RuntimeError{std::move(lease).error()};
    leases->push_back(std::move(*lease));
  }
  return leases;
}

expected<void, Error> copy_staged_tensor_to_host(void* destination, std::size_t destination_bytes,
                                                 const Tensor& value) {
  const auto layout = holoscan::detail::TensorAccess::snapshot(value);
  if (!layout.valid || layout.byte_size > destination_bytes ||
      (layout.byte_size != 0U &&
       (!value.is_contiguous() || destination == nullptr || value.nbytes() <= 0 ||
        static_cast<std::uint64_t>(value.nbytes()) != layout.byte_size ||
        value.data() == nullptr))) {
    return make_unexpected(Error{ErrorCode::kIncompatible,
                                 "operator test Tensor host staging layout or bound mismatch"});
  }
  if (layout.device.device_type != kDLCPU && layout.device.device_type != kDLCUDAHost) {
    return make_unexpected(
        Error{ErrorCode::kIncompatible, "operator test Tensor host staging placement mismatch"});
  }
  // Empty staged fields carry valid metadata but need not own any backing or copy destination.
  if (layout.byte_size == 0U)
    return {};
  return value.copy_to_host(destination, destination_bytes);
}

expected<void, Error> copy_staged_tensor_to_device(void* destination, std::size_t destination_bytes,
                                                   DeviceId device, const Tensor& value,
                                                   ::cuda::stream_ref stream) {
  const auto layout = holoscan::detail::TensorAccess::snapshot(value);
  const auto source_device = layout.device;
  if (!layout.valid ||
      (source_device.device_type != kDLCUDA && source_device.device_type != kDLCUDAManaged) ||
      device.value != source_device.device_id || layout.byte_size > destination_bytes ||
      (layout.byte_size != 0U &&
       (!value.is_contiguous() || destination == nullptr || value.nbytes() <= 0 ||
        static_cast<std::uint64_t>(value.nbytes()) != layout.byte_size ||
        value.data() == nullptr))) {
    return make_unexpected(
        Error{ErrorCode::kIncompatible, "operator test Tensor copy placement or bound mismatch"});
  }
  if (layout.byte_size == 0U)
    return {};
  auto ready = value.synchronize_for_consumer_stream(stream);
  if (!ready)
    return ready;
  auto status = cudaMemcpyAsync(destination,
                                value.data(),
                                static_cast<std::size_t>(layout.byte_size),
                                cudaMemcpyDefault,
                                stream.get());
  if (status == cudaSuccess)
    status = cudaStreamSynchronize(stream.get());
  if (status != cudaSuccess)
    return make_unexpected(Error{ErrorCode::kFailure, cudaGetErrorString(status)});
  return {};
}

expected<void, Error> fill_staged_tensor_loan(TensorOutputLoan& loan, const Tensor& value,
                                              ExecutionContext& context) {
  const auto layout = holoscan::detail::TensorAccess::snapshot(value);
  if (!layout.valid ||
      (layout.byte_size != 0U && (!value.is_contiguous() || value.nbytes() <= 0 ||
                                  static_cast<std::uint64_t>(value.nbytes()) != layout.byte_size ||
                                  value.data() == nullptr))) {
    return make_unexpected(
        Error{ErrorCode::kIncompatible, "operator test Tensor staging layout mismatch"});
  }
  if (loan.memory_kind() == MemoryKind::kHost || loan.memory_kind() == MemoryKind::kPinnedHost) {
    return loan.write_host_and_commit([&](const auto& write) {
      return copy_staged_tensor_to_host(write.data(), write.byte_size(), value);
    });
  }
  const auto stream = context.cuda_stream();
  return loan.write_and_commit(stream, [&](const auto& write) {
    return copy_staged_tensor_to_device(
        write.data(), write.byte_size(), loan.device_id(), value, stream);
  });
}

EmitOptions staged_emit_options(SampleMetadata metadata) {
  EmitOptions options{.flags = metadata.flags};
  // Preserve malformed pairs too; Runtime must reject them exactly as for value samples.
  if (metadata.source_clock_id.valid() || metadata.capture_timestamp_ns != 0)
    options.capture_time = metadata.capture_time();
  if (metadata.frame_id != 0U)
    options.frame_id = metadata.frame_id;
  if (metadata.trace_id != 0U)
    options.trace = TraceContext{.trace_id = TraceId{metadata.trace_id}};
  return options;
}

expected<void, Error> emit_staged_tensor(Output<Tensor>& output, const Tensor& value,
                                         SampleMetadata metadata, ExecutionContext& context) {
  auto loan = output.allocate_tensor(
      TensorLoanRequest{.shape = value.shape_span(), .dtype = value.dtype()});
  if (!loan)
    return make_unexpected(std::move(loan).error());
  auto copied = fill_staged_tensor_loan(*loan, value, context);
  if (!copied)
    return copied;
  return output.emit(std::move(*loan), staged_emit_options(metadata));
}
}  // namespace holoscan::testing::detail
