// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "holoviz_op/error_translation.hpp"

#include <string>
#include "holoscan/logger/logger.hpp"

#include <fmt/format.h>

namespace holoscan::visualization {

const char* holoviz_error_code_name(holoviz::ErrorCode code) {
  switch (code) {
    case holoviz::ErrorCode::kSuccess:
      return "kSuccess";
    case holoviz::ErrorCode::kFailure:
      return "kFailure";
    case holoviz::ErrorCode::kInvalidArgument:
      return "kInvalidArgument";
    case holoviz::ErrorCode::kNotFound:
      return "kNotFound";
    case holoviz::ErrorCode::kNotImplemented:
      return "kNotImplemented";
    case holoviz::ErrorCode::kBackpressured:
      return "kBackpressured";
    case holoviz::ErrorCode::kInvalidState:
      return "kInvalidState";
    case holoviz::ErrorCode::kVulkanError:
      return "kVulkanError";
    case holoviz::ErrorCode::kCudaError:
      return "kCudaError";
    case holoviz::ErrorCode::kOutOfDate:
      return "kOutOfDate";
    case holoviz::ErrorCode::kImageCountChanged:
      return "kImageCountChanged";
    case holoviz::ErrorCode::kFrameDropped:
      return "kFrameDropped";
    case holoviz::ErrorCode::kNullArg:
      return "kNullArg";
    case holoviz::ErrorCode::kPushConstantOverflow:
      return "kPushConstantOverflow";
  }
  return "kUnknown";
}

holoscan::Error translate_error(const holoviz::Error& e, const char* what) {
  holoscan::ErrorCode code = holoscan::ErrorCode::kFailure;
  switch (e.code) {
    case holoviz::ErrorCode::kSuccess:
      code = holoscan::ErrorCode::kSuccess;
      break;
    case holoviz::ErrorCode::kInvalidArgument:
    case holoviz::ErrorCode::kNullArg:
      code = holoscan::ErrorCode::kInvalidArgument;
      break;
    case holoviz::ErrorCode::kNotFound:
      code = holoscan::ErrorCode::kNotFound;
      break;
    case holoviz::ErrorCode::kNotImplemented:
      code = holoscan::ErrorCode::kNotImplemented;
      break;
    case holoviz::ErrorCode::kBackpressured:
      code = holoscan::ErrorCode::kBackpressured;
      break;
    case holoviz::ErrorCode::kOutOfDate:
    case holoviz::ErrorCode::kImageCountChanged:
    case holoviz::ErrorCode::kFrameDropped:
      code = holoscan::ErrorCode::kResourceExhausted;
      break;
    default:
      // kFailure, kInvalidState, kVulkanError, kCudaError, kPushConstantOverflow: no closer
      // holoscan::ErrorCode fits; the original HoloViz code is retained in the message below.
      code = holoscan::ErrorCode::kFailure;
      break;
  }
  return {code,
          std::string(what) + " failed (HoloViz " + holoviz_error_code_name(e.code) +
              "): " + e.message};
}

holoscan::Error translate_cuda_error(CUresult result, const char* what) {
  const char* name = nullptr;
  const char* description = nullptr;
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &description);
  return {holoscan::ErrorCode::kFailure,
          fmt::format("{} failed (CUDA {}): {}",
                      what,
                      name != nullptr ? name : "unknown",
                      description != nullptr ? description : "no description")};
}

holoscan::expected<CUstream, holoscan::Error> create_cuda_stream(CUcontext context) {
  if (context == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "CUDA context is null"});
  }

  CUcontext current = nullptr;
  if (const CUresult result = cuCtxGetCurrent(&current); result != CUDA_SUCCESS) {
    return holoscan::make_unexpected(translate_cuda_error(result, "cuCtxGetCurrent"));
  }
  const bool push_context = current != context;
  if (push_context) {
    if (const CUresult result = cuCtxPushCurrent(context); result != CUDA_SUCCESS) {
      return holoscan::make_unexpected(translate_cuda_error(result, "cuCtxPushCurrent"));
    }
  }

  CUstream stream = nullptr;
  const CUresult create_result = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
  CUresult pop_result = CUDA_SUCCESS;
  if (push_context) {
    CUcontext popped = nullptr;
    pop_result = cuCtxPopCurrent(&popped);
  }
  if (create_result != CUDA_SUCCESS) {
    return holoscan::make_unexpected(translate_cuda_error(create_result, "cuStreamCreate"));
  }
  if (pop_result != CUDA_SUCCESS) {
    (void)cuStreamDestroy(stream);
    return holoscan::make_unexpected(translate_cuda_error(pop_result, "cuCtxPopCurrent"));
  }
  return stream;
}

holoscan::expected<void, holoscan::Error> destroy_cuda_stream(CUcontext context, CUstream stream) {
  if (stream == nullptr) {
    return {};
  }
  if (context == nullptr) {
    return holoscan::make_unexpected(
        holoscan::Error{holoscan::ErrorCode::kInvalidArgument, "CUDA context is null"});
  }

  CUcontext current = nullptr;
  if (const CUresult result = cuCtxGetCurrent(&current); result != CUDA_SUCCESS) {
    return holoscan::make_unexpected(translate_cuda_error(result, "cuCtxGetCurrent"));
  }
  const bool push_context = current != context;
  if (push_context) {
    if (const CUresult result = cuCtxPushCurrent(context); result != CUDA_SUCCESS) {
      return holoscan::make_unexpected(translate_cuda_error(result, "cuCtxPushCurrent"));
    }
  }

  const CUresult destroy_result = cuStreamDestroy(stream);
  CUresult pop_result = CUDA_SUCCESS;
  if (push_context) {
    CUcontext popped = nullptr;
    pop_result = cuCtxPopCurrent(&popped);
  }
  if (destroy_result != CUDA_SUCCESS) {
    return holoscan::make_unexpected(translate_cuda_error(destroy_result, "cuStreamDestroy"));
  }
  if (pop_result != CUDA_SUCCESS) {
    return holoscan::make_unexpected(translate_cuda_error(pop_result, "cuCtxPopCurrent"));
  }
  return {};
}

[[noreturn]] void throw_holoviz_error(const holoviz::Error& e, const char* what) {
  HOLOSCAN_LOG_ERROR(
      "{} failed (HoloViz {}): {}", what, holoviz_error_code_name(e.code), e.message);
  throw holoscan::RuntimeError{translate_error(e, what)};
}

}  // namespace holoscan::visualization
