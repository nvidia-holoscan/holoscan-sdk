/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <memory>
#include <optional>
#include <tuple>

#include <dlpack/dlpack.h>

#include <holoscan/core/domain/tensor.hpp>

namespace holoscan {

namespace nb = nanobind;

// DLPack capsule names per the array API standard. A capsule is renamed to its "used_*" variant
// once a consumer has taken ownership, so the producer's destructor knows not to free it again.
constexpr const char* dlpack_capsule_name{"dltensor"};
constexpr const char* used_dlpack_capsule_name{"used_dltensor"};
constexpr const char* dlpack_versioned_capsule_name{"dltensor_versioned"};
constexpr const char* used_dlpack_versioned_capsule_name{"used_dltensor_versioned"};

/// @brief Implement the `__dlpack__` protocol: produce a PyCapsule wrapping the tensor's
/// (versioned or unversioned) DLManagedTensor, honoring the consumer stream / max_version /
/// dl_device / copy arguments. Phase 2 of the nanobind port (export); see
/// h5x_plan_holoscan_tensor_python.md.
nb::capsule py_dlpack(Tensor* tensor, const nb::object& stream,
                      std::optional<std::tuple<int, int>> max_version,
                      std::optional<std::tuple<int, int>> dl_device, std::optional<bool> copy);

/// @brief Implement the `__dlpack_device__` protocol: return a (device_type, device_id) tuple.
nb::tuple py_dlpack_device(Tensor* tensor);

/// @brief Import a Tensor from any object supporting the `__dlpack__` protocol.
///
/// The imported DLManagedTensor is owned via a GIL-acquiring memory reference, so the producer's
/// deleter (which touches Python refcounts) runs safely even if the Tensor is released from a
/// non-Python thread. This avoids the async LazyDLManagedTensorDeleter used in the 4.x bindings.
std::shared_ptr<Tensor> tensor_from_dlpack(const nb::object& obj);

/// @brief Import a Tensor from any tensor-like object, preferring the DLPack C exchange API and
/// then Python DLPack over the CUDA/host array interfaces (see the rationale comment in
/// dl_converter.cpp). Backs `Tensor.as_tensor`.
std::shared_ptr<Tensor> as_tensor(const nb::object& obj);

/// @brief Whether @p obj exposes a tensor-like protocol (`__dlpack_c_exchange_api__`,
/// `__dlpack__` + `__dlpack_device__`, or an array interface). Mirrors the duck-typing used by
/// `as_tensor`.
bool is_tensor_like(const nb::object& obj);

/// @brief Build the NumPy `__array_interface__` dict for a host (CPU / pinned-host) Tensor.
/// Raises AttributeError if the tensor is not host-resident, so `hasattr` reports correctly.
nb::dict array_interface(const Tensor& tensor);

/// @brief Build the `__cuda_array_interface__` dict for a device (CUDA / managed) Tensor.
/// Raises AttributeError if the tensor is not device-resident.
nb::dict cuda_array_interface(const Tensor& tensor);

/// @brief PyCapsule wrapping the DLPack 1.3 C exchange API table; bound as
/// `Tensor.__dlpack_c_exchange_api__` (see https://dmlc.github.io/dlpack/latest/python_spec.html).
nb::capsule dlpack_c_exchange_api_capsule();

/// @brief Test-only helper that looks up the C exchange API capsule and invokes its callbacks
/// (managed export, stack DLTensor export, import, stream), returning a dict of results.
nb::dict test_dlpack_c_exchange_api(const nb::object& tensor_obj);

}  // namespace holoscan
