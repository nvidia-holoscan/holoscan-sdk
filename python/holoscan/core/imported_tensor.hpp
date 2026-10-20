/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime_api.h>

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>

namespace holomq::cuda {
class StreamLifetime;
}  // namespace holomq::cuda

namespace holoscan {

class PyImportedMemoryReference;

// Capture imported-array use at callback exit while the SDK stream is still alive. Nested
// callbacks restore the outer scope; the scope retains owners even if Python drops its views.
class PyImportedTensorScope {
 public:
  explicit PyImportedTensorScope(std::optional<cudaStream_t> stream = std::nullopt) noexcept;
  ~PyImportedTensorScope();
  PyImportedTensorScope(const PyImportedTensorScope&) = delete;
  PyImportedTensorScope& operator=(const PyImportedTensorScope&) = delete;

  void track(std::shared_ptr<PyImportedMemoryReference> reference, cudaStream_t stream);
  [[nodiscard]] static std::optional<cudaStream_t> stream() noexcept;
  [[nodiscard]] static bool active() noexcept;

 private:
  PyImportedTensorScope* previous_{};
  std::optional<cudaStream_t> stream_;
  std::vector<std::pair<std::shared_ptr<PyImportedMemoryReference>, cudaStream_t>> references_;
};

// The owner deleter must acquire the GIL. CPU owners keep synchronous Python destruction;
// CUDA owners retire on a bounded worker after producer and consumer events complete.
[[nodiscard]] std::shared_ptr<TensorMemoryReference> make_python_tensor_memory_reference(
    std::shared_ptr<void> owner, DLDevice device);
void track_python_tensor(const Tensor& tensor,
                         std::optional<cudaStream_t> consumer_stream = std::nullopt);
// Legacy array-interface and no-sync C exchange exports do not report a consumer stream.
// Retire these exports conservatively after device completion on the exporting/owning devices.
void track_python_tensor_untracked_export(const Tensor& tensor);
// Source-tree test observation retains only a liveness block, never Tensor storage or its stream.
std::shared_ptr<const holomq::cuda::StreamLifetime> python_tensor_stream_lifetime(
    const Tensor& tensor, cudaStream_t stream);
void drain_python_tensor_releases(double timeout = -1);
void shutdown_python_tensor_releases();

}  // namespace holoscan
