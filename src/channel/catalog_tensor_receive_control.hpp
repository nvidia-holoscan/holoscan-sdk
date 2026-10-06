/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holomq {
class Context;
}  // namespace holomq

namespace holoscan {

class Tensor;
struct TensorBoundaryContractRecord;

namespace channel_internal {

struct HoloMQTensorDescriptor;
class CatalogTensorReceiveMemoryReference;

/// Private borrow-bounded control pool for catalog BufferHandle Tensor receive.
class CatalogTensorReceiveControl {
 public:
  CatalogTensorReceiveControl(const CatalogTensorReceiveControl&) = delete;
  CatalogTensorReceiveControl& operator=(const CatalogTensorReceiveControl&) = delete;
  ~CatalogTensorReceiveControl();

  [[nodiscard]] static expected<std::unique_ptr<CatalogTensorReceiveControl>, Error> create(
      std::shared_ptr<void> runtime_lifetime, holomq::Context& context,
      const TensorBoundaryContractRecord& contract, std::uint32_t borrow_capacity);

  [[nodiscard]] expected<Tensor, Error> materialize(const HoloMQTensorDescriptor& descriptor);

  /// Release the subscriber's owner reference. Active Tensor claims retain the control.
  void release() noexcept;

 private:
  friend class CatalogTensorReceiveMemoryReference;

  struct Slot;

  CatalogTensorReceiveControl(std::shared_ptr<void> runtime_lifetime, holomq::Context& context,
                              std::uint32_t borrow_capacity) noexcept;

  [[nodiscard]] expected<void, Error> initialize(const TensorBoundaryContractRecord& contract);
  [[nodiscard]] expected<std::size_t, Error> acquire_slot() noexcept;
  void return_unmaterialized_slot(std::size_t slot_index) noexcept;
  void retire_slot(std::size_t slot_index) noexcept;
  void retain() noexcept;

  std::shared_ptr<void> runtime_lifetime_;
  holomq::Context* context_{};
  std::unique_ptr<class BufferImportArray> imports_;
  // Runtime catalog capacities require dynamically sized contiguous arrays.
  // NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays)
  std::unique_ptr<Slot[]> slots_;
  std::unique_ptr<std::atomic<std::uint64_t>[]> free_words_;
  // NOLINTEND(cppcoreguidelines-avoid-c-arrays)
  std::size_t slot_count_{};
  std::size_t free_word_count_{};
  std::atomic<std::uint32_t> references_{1U};
};

}  // namespace channel_internal
}  // namespace holoscan
