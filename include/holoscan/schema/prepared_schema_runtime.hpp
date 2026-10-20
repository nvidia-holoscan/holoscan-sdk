/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>

#include <holoscan/core/port.hpp>
#include <holoscan/schema/flatbuffer_root_access.hpp>
#include <holoscan/schema/prepared_schema_storage.hpp>

namespace holoscan::schema::detail {

// Two separately constructed objects share an aligned allocation. Sample<T> is not necessarily
// standard-layout: do not rely on pointer interconvertibility with a containing wrapper object.
// Keeping the Sample at byte zero preserves the existing erased public-sample address contract.
template <typename Native, auto... TensorMembers>
struct PreparedSchemaRuntimeStorage {
  using Traits = PreparedSchemaTraits<Native>;
  using Root = typename Native::TableType;
  struct Inventory {
    PreparedMemoryBudget budget{Traits::max_prepared_bytes};
    typename Traits::Storage fields;
  };
  static constexpr std::size_t alignment = std::max(alignof(Sample<Native>), alignof(Inventory));
  static constexpr std::size_t inventory_offset =
      (sizeof(Sample<Native>) + alignof(Inventory) - 1) / alignof(Inventory) * alignof(Inventory);
  static constexpr std::size_t size = inventory_offset + sizeof(Inventory);

  static Inventory* inventory_address(void* storage) noexcept {
    return reinterpret_cast<Inventory*>(static_cast<std::byte*>(storage) + inventory_offset);
  }
  static Inventory& inventory(void* storage) noexcept {
    return *std::launder(inventory_address(storage));
  }
  static Sample<Native>& sample(void* storage) noexcept {
    return *std::launder(static_cast<Sample<Native>*>(storage));
  }
  static expected<void, Error> prepare(void* storage) noexcept {
    if (storage == nullptr || reinterpret_cast<std::uintptr_t>(storage) % alignment != 0)
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    if (size > Traits::max_prepared_bytes)
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    bool sample_constructed = false;
    bool inventory_constructed = false;
    auto cleanup = [&] {
      if (inventory_constructed)
        std::destroy_at(inventory_address(storage));
      if (sample_constructed)
        std::destroy_at(static_cast<Sample<Native>*>(storage));
    };
    try {
      std::construct_at(static_cast<Sample<Native>*>(storage), Native{}, SampleMetadata{});
      sample_constructed = true;
      std::construct_at(inventory_address(storage));
      inventory_constructed = true;
      auto& state = inventory(storage);
      auto& value = sample(storage).data;
      state.budget.charge(sizeof(Native), &PreparedSchemaMemory::native_bytes);
      state.budget.charge(size - sizeof(Native), &PreparedSchemaMemory::inventory_bytes);
      state.fields.prepare(value, state.budget);
      ((value.*TensorMembers = std::allocate_shared<holoscan::Tensor>(
            PreparedOwnerAllocator<holoscan::Tensor>{state.budget})),
       ...);
      return {};
    } catch (const std::bad_alloc&) {
      cleanup();
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    } catch (...) {
      cleanup();
      return make_unexpected(Error{ErrorCode::kFailure});
    }
  }
  static void destroy(void* storage) noexcept {
    std::destroy_at(inventory_address(storage));
    std::destroy_at(static_cast<Sample<Native>*>(storage));
  }
  static void reset(void* storage) noexcept {
    auto& state = inventory(storage);
    auto& current = sample(storage);
    // Borrowed input must not be mutated or retain nested owners past selection retirement.
    // Do not mutate aliased storage; the next assignment diagnoses the local contract violation.
    if (state.fields.valid(current.data))
      state.fields.reset_unchecked(current.data);
    ((current.data.*TensorMembers ? void(*(current.data.*TensorMembers) = holoscan::Tensor{})
                                  : void()),
     ...);
    current.metadata = {};
  }
  static expected<void, Error> assign_root(void* storage, const Root& root,
                                           SampleMetadata metadata) noexcept {
    auto& state = inventory(storage);
    auto& current = sample(storage);
    if (!state.fields.valid(current.data))
      return make_unexpected(Error{ErrorCode::kPreparedStorageInvalid});
    if (!state.fields.wire_fits(root))
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    state.fields.assign_root_unchecked(current.data, root);
    current.metadata = metadata;
    return {};
  }
  static expected<void, Error> assign_bytes(void* storage, std::span<const std::byte> bytes,
                                            SampleMetadata metadata) noexcept {
    if (storage == nullptr || reinterpret_cast<std::uintptr_t>(storage) % alignment != 0 ||
        bytes.empty() || bytes.size() > FlatBufferSchemaBinding<Root>::max_serialized_size)
      return make_unexpected(Error{ErrorCode::kReceiveError});
    if (!verify_flatbuffer_root_bytes<Root>(bytes))
      return make_unexpected(Error{ErrorCode::kCodecError});
    return assign_root(storage, *parse_flatbuffer_root_bytes<Root>(bytes), metadata);
  }
  static expected<void, Error> assign_tensors(void* storage, const Root& root,
                                              std::span<holoscan::Tensor> tensors,
                                              SampleMetadata metadata) noexcept {
    if (storage == nullptr || reinterpret_cast<std::uintptr_t>(storage) % alignment != 0 ||
        tensors.size() != sizeof...(TensorMembers))
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    auto& value = sample(storage).data;
    if (!((value.*TensorMembers != nullptr) && ...))
      return make_unexpected(Error{ErrorCode::kPreparedStorageInvalid});
    auto assigned = assign_root(storage, root, metadata);
    if (!assigned)
      return assigned;
    std::size_t index = 0;
    ((*(value.*TensorMembers) = std::move(tensors[index++])), ...);
    return {};
  }

  static void configure(holoscan::detail::RuntimePortTypeOps& ops) noexcept {
    ops.sample_size = size;
    ops.sample_alignment = alignment;
    ops.persistent_sample_storage = true;
    ops.prepare_sample_storage = prepare;
    ops.reset_sample_storage = reset;
    ops.destroy_sample_storage = destroy;
    if constexpr (sizeof...(TensorMembers) == 0)
      ops.assign_sample_storage = assign_bytes;
    if constexpr (sizeof...(TensorMembers) != 0) {
      ops.descriptor_metadata_fits = [](const void* source) noexcept {
        return source != nullptr && Traits::source_fits(*static_cast<const Native*>(source));
      };
    }
  }
};

}  // namespace holoscan::schema::detail
