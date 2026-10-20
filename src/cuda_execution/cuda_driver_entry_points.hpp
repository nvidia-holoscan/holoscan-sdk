/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Internal (non-installed) helpers for resolving CUDA driver entry points at runtime via the
// runtime's driver-entry-point query, shared by the CUDA execution pools (stream pool, event pool,
// green context pool). Keeping the loading and diagnostics in one place avoids duplicating the
// resolver and gives every caller the same symbol-missing vs driver-too-old error reporting.

#include <cuda.h>
#include <cuda_runtime.h>

#include <fmt/format.h>

#include <cstdint>
#include <string>
#include <utility>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

// Resolving driver entry points needs cudaGetDriverEntryPointByVersion, a CUDA 12.3+ runtime API.
// Callers gate any feature that relies on these helpers on this macro; the corresponding public
// Config fields stay available on all toolkits so the API surface is stable.
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12030)
#define HOLOSCAN_CUDA_DRIVER_ENTRY_POINTS_AVAILABLE 1
#else
#define HOLOSCAN_CUDA_DRIVER_ENTRY_POINTS_AVAILABLE 0
#endif

namespace holoscan::cuda_execution_detail {

#if HOLOSCAN_CUDA_DRIVER_ENTRY_POINTS_AVAILABLE

inline constexpr std::int32_t kCUDADriverABIVersion12030 = 12030;
inline constexpr std::int32_t kCUDADriverABIVersion12040 = 12040;
inline constexpr std::int32_t kCUDADriverABIVersion12050 = 12050;
inline constexpr std::int32_t kCUDADriverABIVersion13010 = 13010;

// Human-readable explanation of a driver entry-point query result, so a failed lookup can say
// whether the symbol was missing or the installed driver is simply too old for it.
inline const char* driver_entry_point_status_text(cudaDriverEntryPointQueryResult status) {
  switch (status) {
    case cudaDriverEntryPointSuccess:
      return "entry point resolved";
    case cudaDriverEntryPointSymbolNotFound:
      return "symbol not found in the CUDA driver";
    // CUDA's enumerator name is misspelled in the driver header; match that spelling.
    case cudaDriverEntryPointVersionNotSufficent:
      return "the installed CUDA driver is too old for this symbol";
    default:
      return "unknown driver entry-point query result";
  }
}

// Build the error for a failed driver entry-point lookup, naming the symbol and the cause.
inline Error driver_entry_point_error(const char* name, cudaDriverEntryPointQueryResult status,
                                      cudaError_t result) {
  return Error{ErrorCode::kFailure,
               fmt::format("CUDA driver entry point '{}' is unavailable: {} "
                           "(query result {}, cuda error {})",
                           name,
                           driver_entry_point_status_text(status),
                           static_cast<int>(status),
                           static_cast<int>(result))};
}

// Resolve a driver entry point into a caller-owned function-pointer slot. The runtime sets the slot
// to null on any non-success status, so a null slot is reported with the specific cause. Suits
// callers that fill a table of differently-typed pointers (e.g. the green context pool).
template <typename Query>
inline expected<void, Error> load_driver_entry_point_with_query(void** func, const char* name,
                                                                std::int32_t abi_version,
                                                                Query&& query) {
  cudaDriverEntryPointQueryResult status{};
  const cudaError_t result = std::forward<Query>(query)(name, func, abi_version, &status);
  if (result != cudaSuccess) {
    return make_unexpected(driver_entry_point_error(name, status, result));
  }
  if (status != cudaDriverEntryPointSuccess) {
    return make_unexpected(driver_entry_point_error(name, status, result));
  }
  if (*func == nullptr) {
    return make_unexpected(driver_entry_point_error(name, status, result));
  }
  return {};
}

inline expected<void, Error> load_driver_entry_point(void** func, const char* name,
                                                     std::int32_t abi_version) {
  return load_driver_entry_point_with_query(
      func,
      name,
      abi_version,
      [](const char* symbol,
         void** output,
         std::int32_t version,
         cudaDriverEntryPointQueryResult* status) {
        return cudaGetDriverEntryPointByVersion(symbol, output, version, cudaEnableDefault, status);
      });
}

// Resolve a driver entry point and return it as a typed function pointer, or a specific error.
template <typename Fn>
expected<Fn, Error> resolve_driver_entry_point(const char* name, std::int32_t abi_version) {
  Fn loaded = nullptr;
  if (auto result = load_driver_entry_point(reinterpret_cast<void**>(&loaded), name, abi_version);
      !result) {
    return forward_error(result);
  }
  return loaded;
}

using CtxPushCurrentFn = CUresult (*)(CUcontext);
using CtxPopCurrentFn = CUresult (*)(CUcontext*);

// Resolve cuCtxPushCurrent / cuCtxPopCurrent once and cache the result (pointer or diagnostic) in a
// thread-safe magic static. Used by callers that create CUDA objects inside an application-provided
// CUcontext (stream pool, event pool).
inline const expected<CtxPushCurrentFn, Error>& ctx_push_current_fn() {
  static const expected<CtxPushCurrentFn, Error> fn =
      resolve_driver_entry_point<CtxPushCurrentFn>("cuCtxPushCurrent", kCUDADriverABIVersion12030);
  return fn;
}

inline const expected<CtxPopCurrentFn, Error>& ctx_pop_current_fn() {
  static const expected<CtxPopCurrentFn, Error> fn =
      resolve_driver_entry_point<CtxPopCurrentFn>("cuCtxPopCurrent", kCUDADriverABIVersion12030);
  return fn;
}

// Verify the currently-current CUDA context is on @p expected_device_id. Call this with the context
// of interest already made current (e.g. just after cuCtxPushCurrent): the runtime's current device
// follows the current context, so cudaGetDevice reports the pushed context's device. This uses the
// runtime API deliberately — the driver's cuCtxGetDevice is remapped to a different-signature _v2
// entry point by cudaGetDriverEntryPointByVersion on newer toolkits, so calling it through a fixed
// signature is unsafe.
inline expected<void, Error> verify_current_context_device(int expected_device_id) {
  int current_device = -1;
  const cudaError_t result = cudaGetDevice(&current_device);
  if (result != cudaSuccess) {
    return make_unexpected(Error{ErrorCode::kFailure,
                                 fmt::format("cudaGetDevice failed: {} ({})",
                                             cudaGetErrorString(result),
                                             static_cast<int>(result))});
  }
  if (current_device != expected_device_id) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument,
              fmt::format(
                  "provided CUDA context is on device {} but the pool is configured for device {}",
                  current_device,
                  expected_device_id)});
  }
  return {};
}

#endif  // HOLOSCAN_CUDA_DRIVER_ENTRY_POINTS_AVAILABLE

}  // namespace holoscan::cuda_execution_detail
