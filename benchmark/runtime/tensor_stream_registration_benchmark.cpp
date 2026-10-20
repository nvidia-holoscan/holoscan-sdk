// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>
#include <memory>

#include <benchmark/benchmark.h>

#include "core/domain/tensor_access.hpp"

namespace holoscan {
namespace {

// Match the native pool's two-base RTTI shape without measuring CUDA, locking or pinning.
class NativeReference : public TensorMemoryReference, public detail::TensorStreamLifetimeConsumer {
 public:
  bool register_consumer_stream(
      cudaStream_t, const std::shared_ptr<const holomq::cuda::StreamLifetime>&) noexcept override {
    return true;
  }
};

// Imported Python references additionally implement the callback-completion extension.
class ImportedReference final : public TensorMemoryReference,
                                public detail::TensorConsumerCompletion,
                                public detail::TensorStreamLifetimeConsumer {
 public:
  bool register_consumer_stream(
      cudaStream_t, const std::shared_ptr<const holomq::cuda::StreamLifetime>&) noexcept override {
    return true;
  }
  void record_consumer_completion(cudaStream_t) noexcept override {}
};

template <typename Reference>
void BM_LifetimeCrossCast(benchmark::State& state) {
  Reference object;
  auto* reference = static_cast<TensorMemoryReference*>(&object);
  for (auto _ : state) {
    // Keep the dynamic type opaque to the optimizer on every iteration, as at the DSO call.
    benchmark::DoNotOptimize(reference);
    benchmark::DoNotOptimize(dynamic_cast<detail::TensorStreamLifetimeConsumer*>(reference));
  }
}

template <typename Reference>
void BM_CompletionCrossCast(benchmark::State& state) {
  Reference object;
  auto* reference = static_cast<TensorMemoryReference*>(&object);
  for (auto _ : state) {
    benchmark::DoNotOptimize(reference);
    benchmark::DoNotOptimize(dynamic_cast<detail::TensorConsumerCompletion*>(reference));
  }
}

void BM_CachedLifetimePointer(benchmark::State& state) {
  NativeReference object;
  auto* consumer = static_cast<detail::TensorStreamLifetimeConsumer*>(&object);
  for (auto _ : state) {
    benchmark::DoNotOptimize(consumer);
    benchmark::DoNotOptimize(consumer);
  }
}

BENCHMARK_TEMPLATE(BM_LifetimeCrossCast, NativeReference);
BENCHMARK_TEMPLATE(BM_LifetimeCrossCast, ImportedReference);
BENCHMARK_TEMPLATE(BM_CompletionCrossCast, NativeReference);
BENCHMARK_TEMPLATE(BM_CompletionCrossCast, ImportedReference);
BENCHMARK(BM_CachedLifetimePointer);

// Isolate the enqueue admission paid by a native CUDA handoff from graph scheduling.
void BM_StreamRecordAdmission(benchmark::State& state, bool pinned) {
  cudaStream_t stream{};
  cudaEvent_t event{};
  if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess ||
      cudaEventCreateWithFlags(&event, cudaEventDisableTiming) != cudaSuccess) {
    state.SkipWithMessage("requires CUDA stream/event creation");
    return;
  }
  holomq::cuda::StreamLifetime lifetime{stream, 0};
  for (auto _ : state) {
    auto admission = pinned ? lifetime.try_pin() : holomq::cuda::StreamLifetime::PinResult{};
    if ((pinned && !admission.pin) || cudaEventRecord(event, stream) != cudaSuccess) {
      state.SkipWithError("native event enqueue failed");
      break;
    }
  }
  static_cast<void>(cudaStreamSynchronize(stream));
  lifetime.close();
  static_cast<void>(lifetime.publish_retired());
  static_cast<void>(cudaEventDestroy(event));
  static_cast<void>(cudaStreamDestroy(stream));
}
BENCHMARK_CAPTURE(BM_StreamRecordAdmission, Raw, false);
BENCHMARK_CAPTURE(BM_StreamRecordAdmission, Pinned, true);

}  // namespace
}  // namespace holoscan
