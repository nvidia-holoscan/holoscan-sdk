/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>

#include <benchmark/benchmark.h>

#include <holoscan/core/port.hpp>
#include <holoscan/sensor_io/image_descriptor_scratch.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "common/allocation_counter.hpp"

namespace holoscan {
namespace {

// REQ: OPERATOR-056

// Isolate the public owning accessor with an already-selected sample. This binding models one
// finite read credit, but not activation admission, transport, retirement, or a production graph.
struct SelectedSampleState {
  const void* sample{};
  mutable std::size_t claims{};
  std::uint64_t view_generation{1U};

  static void retain(void* opaque) noexcept { ++static_cast<SelectedSampleState*>(opaque)->claims; }

  static void release(void* opaque) noexcept {
    --static_cast<SelectedSampleState*>(opaque)->claims;
  }

  static expected<detail::RuntimeScalarReadBinding, Error> scalar(const void* opaque, std::uint64_t,
                                                                  bool acquire_claim) noexcept {
    const auto& state = *static_cast<const SelectedSampleState*>(opaque);
    detail::PublishedReadClaimBinding claim;
    if (acquire_claim) {
      if (state.claims != 0U) {
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      }
      ++state.claims;
      claim = {.state = const_cast<SelectedSampleState*>(&state),
               .retain = &retain,
               .release = &release};
    }
    return detail::RuntimeScalarReadBinding{
        .sample = state.sample,
        .claim = claim,
        .view_generation = &state.view_generation,
        .expected_view_generation = state.view_generation,
    };
  }

  static inline const detail::RuntimeInputBindingOps kOps{.scalar = &scalar};
};

class PreparedImageSample {
 public:
  PreparedImageSample() {
    const auto prepared = ops_->prepare_sample_storage(storage_.data());
    if (prepared) {
      sample_ = reinterpret_cast<Sample<schema::ImageT>*>(storage_.data());
    }
  }
  PreparedImageSample(const PreparedImageSample&) = delete;
  PreparedImageSample& operator=(const PreparedImageSample&) = delete;
  ~PreparedImageSample() {
    if (sample_ != nullptr) {
      ops_->destroy_sample(storage_.data());
    }
  }

  [[nodiscard]] Sample<schema::ImageT>* get() noexcept { return sample_; }

 private:
  const detail::RuntimePortTypeOps* ops_{detail::runtime_port_type_ops<schema::ImageT>()};
  alignas(Sample<schema::ImageT>) std::array<std::byte, sizeof(Sample<schema::ImageT>)> storage_{};
  Sample<schema::ImageT>* sample_{};
};

// The backing and DLPack record outlive all samples in each benchmark function. Imports and shape
// changes happen before measurement, never inside the accessor or allocation-counting loops.
struct HostPixels {
  std::array<std::uint8_t, 48> bytes{};
  std::array<std::int64_t, 3> shape{4, 4, 3};
  DLManagedTensor managed{};

  explicit HostPixels(bool planar) {
    if (planar) {
      shape = {3, 4, 4};
    }
    managed.dl_tensor = DLTensor{.data = bytes.data(),
                                 .device = {kDLCPU, 0},
                                 .ndim = 3,
                                 .dtype = {kDLUInt, 8, 1},
                                 .shape = shape.data()};
    managed.deleter = [](DLManagedTensor*) {};
  }
};

template <PublishedPayload T>
class BoundInput {
 public:
  explicit BoundInput(const Sample<T>& sample) : state_{.sample = &sample} {
    detail::RuntimePortBindingAccess::bind_input(input, &binding_);
  }
  BoundInput(const BoundInput&) = delete;
  BoundInput& operator=(const BoundInput&) = delete;
  ~BoundInput() { detail::RuntimePortBindingAccess::unbind_input(input, &binding_); }

  [[nodiscard]] bool claims_released() const noexcept { return state_.claims == 0U; }

  Input<T> input{"input"};

 private:
  SelectedSampleState state_;
  detail::RuntimeInputBinding binding_{
      .state = &state_, .ops = &SelectedSampleState::kOps, .generation = 1U};
};

template <typename Read>
void measure(benchmark::State& state, const Read& read, bool require_zero_allocations = false) {
  // Count a companion pass, not the timed loop: counter overhead must not become accessor time.
  constexpr std::size_t kCountedReads = 100U;
  std::size_t first_allocations{};
  std::size_t repeated_allocations{};
  bool valid = true;
  {
    const test::ScopedAllocationCounter counter;
    valid = read();
    first_allocations = counter.count();
  }
  {
    const test::ScopedAllocationCounter counter;
    for (std::size_t index = 0U; index < kCountedReads; ++index) {
      valid = read() && valid;
    }
    repeated_allocations = counter.count();
  }
  if (!valid) {
    state.SkipWithError("owning read or backing-identity validation failed");
    return;
  }
  if (require_zero_allocations && (first_allocations != 0U || repeated_allocations != 0U)) {
    state.SkipWithError("borrowed descriptor handling allocated");
    return;
  }
  for (auto _ : state) {
    if (!read()) {
      state.SkipWithError("owning read failed in timed loop");
      break;
    }
  }
  state.counters["first_read_allocations"] = static_cast<double>(first_allocations);
  state.counters["allocations_per_read"] =
      static_cast<double>(repeated_allocations) / static_cast<double>(kCountedReads);
  state.counters["counted_reads"] = static_cast<double>(kCountedReads);
  state.SetItemsProcessed(state.iterations());
}

void BM_SelectedTensorReceive(benchmark::State& state) {
  HostPixels pixels{false};
  const Sample<Tensor> sample{Tensor{&pixels.managed}, {}};
  BoundInput<Tensor> binding{sample};
  measure(state, [&] {
    auto received = binding.input.receive();
    if (!received) {
      return false;
    }
    benchmark::DoNotOptimize(received->data);
    return received->data.data() == pixels.bytes.data();
  });
  if (!binding.claims_released()) {
    state.SkipWithError("owning receive leaked its read claim");
  }
}

void BM_SelectedTensorBorrow(benchmark::State& state) {
  HostPixels pixels{false};
  const Sample<Tensor> sample{Tensor{&pixels.managed}, {}};
  BoundInput<Tensor> binding{sample};
  measure(
      state,
      [&] {
        auto view = binding.input.selected_sample();
        if (!view) {
          return false;
        }
        auto* payload = &view->data();
        benchmark::DoNotOptimize(payload);
        return &view->sample() == &sample && view->data().data() == pixels.bytes.data() &&
               binding.claims_released();
      },
      true);
}

enum class ImageRead { kReceive, kReceiveData, kPreparedAssignment, kBorrow, kBorrowEdit };

// Share setup and backing-identity checks across the measured accessor variants so their
// allocation and timing results remain directly comparable.
template <ImageRead kRead>
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void BM_SelectedImageCopy(benchmark::State& state) {
  const bool planar = state.range(0) != 0;
  HostPixels pixels{planar};
  PreparedImageSample source;
  PreparedImageSample destination;
  auto scratch = sensor_io::ImageDescriptorScratch::create();
  if (!scratch) {
    state.SkipWithError("Image scratch preparation failed");
    return;
  }
  if (source.get() == nullptr || destination.get() == nullptr) {
    state.SkipWithError("Image slot preparation failed");
    return;
  }
  auto& image = source.get()->data;
  *image.data = Tensor{&pixels.managed};
  image.width = 4U;
  image.height = 4U;
  image.encoding = planar ? schema::ImageEncoding_R8_G8_B8 : schema::ImageEncoding_RGB8;
  image.header->frame_id.assign(static_cast<std::size_t>(state.range(1)), 'c');
  if (planar) {
    for (std::uint64_t plane = 0U; plane != 3U; ++plane) {
      image.plane_layouts.emplace_back(plane * 16U, 4U, 16U);
    }
  }
  // Receiving variants need a mutable input; the prepared-assignment instantiation does not.
  // NOLINTNEXTLINE(misc-const-correctness)
  BoundInput<schema::ImageT> binding{*source.get()};
  const auto check_copy = [&](schema::ImageT& copy) {
    benchmark::DoNotOptimize(copy);
    return copy.data && copy.data != image.data && copy.data->data() == pixels.bytes.data() &&
           copy.header && copy.header != image.header && copy.header->stamp &&
           copy.header->stamp != image.header->stamp &&
           copy.header->frame_id.size() == image.header->frame_id.size() &&
           copy.plane_layouts.size() == image.plane_layouts.size();
  };
  measure(
      state,
      [&] {
        if constexpr (kRead == ImageRead::kReceive) {
          auto received = binding.input.receive();
          return received && check_copy(received->data);
        } else if constexpr (kRead == ImageRead::kReceiveData) {
          auto received = binding.input.receive_data();
          return received && check_copy(*received);
        } else if constexpr (kRead == ImageRead::kBorrow || kRead == ImageRead::kBorrowEdit) {
          auto view = binding.input.selected_sample();
          if (!view) {
            return false;
          }
          auto* payload = &view->data();
          benchmark::DoNotOptimize(payload);
          if constexpr (kRead == ImageRead::kBorrowEdit) {
            if (!scratch->copy_metadata_from(view->data())) {
              return false;
            }
            scratch->image().roi_offset_y = 5;
            benchmark::DoNotOptimize(scratch->image());
            if (scratch->image().data || image.roi_offset_y != 0 ||
                scratch->image().header->frame_id != image.header->frame_id) {
              return false;
            }
          }
          return &view->sample() == source.get() && &view->data() == &image &&
                 view->data().data->data() == pixels.bytes.data() && binding.claims_released();
        } else {
          destination.get()->data = image;
          return check_copy(destination.get()->data);
        }
      },
      kRead == ImageRead::kBorrow || kRead == ImageRead::kBorrowEdit);
  if (!binding.claims_released()) {
    state.SkipWithError("owning receive leaked its read claim");
  }
}

void image_cases(benchmark::Benchmark* benchmark_case) {
  benchmark_case->ArgNames({"planar", "frame_id_bytes"});
  for (const auto frame_id_bytes : {6, 128}) {
    for (const auto planar : {0, 1}) {
      benchmark_case->Args({planar, frame_id_bytes});
    }
  }
}

BENCHMARK(BM_SelectedTensorReceive);
BENCHMARK(BM_SelectedTensorBorrow);
BENCHMARK_TEMPLATE(BM_SelectedImageCopy, ImageRead::kReceive)->Apply(image_cases);
BENCHMARK_TEMPLATE(BM_SelectedImageCopy, ImageRead::kReceiveData)->Apply(image_cases);
BENCHMARK_TEMPLATE(BM_SelectedImageCopy, ImageRead::kPreparedAssignment)->Apply(image_cases);
BENCHMARK_TEMPLATE(BM_SelectedImageCopy, ImageRead::kBorrow)->Apply(image_cases);
// REQ: SENSOR-IO-011
BENCHMARK_TEMPLATE(BM_SelectedImageCopy, ImageRead::kBorrowEdit)->Apply(image_cases);

}  // namespace
}  // namespace holoscan
