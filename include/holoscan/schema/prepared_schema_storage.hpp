/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>

namespace holoscan::schema {

/// @brief Describe charged prepared storage, excluding allocator bookkeeping and cold peaks.
struct PreparedSchemaMemory {
  std::size_t native_bytes{};     ///< Root native value, including its inline fields.
  std::size_t inventory_bytes{};  ///< Facade, state, and private owner/container inventories.
  std::size_t optional_bytes{};   ///< Optional objects and their actual control-block requests.
  std::size_t container_bytes{};  ///< Reserved container storage, including parked string buffers.

  /// @brief Return the total charged bytes; successful preparation checked every addition.
  [[nodiscard]] std::size_t total_bytes() const noexcept {
    return native_bytes + inventory_bytes + optional_bytes + container_bytes;
  }
};

/// @brief Bind explicitly opted-in native types to generated prepared field operations.
/// @note Specializations are emitted by holoscan_add_flatbuffer_schema(), not inferred from T.
template <typename T>
struct PreparedSchemaTraits;

namespace detail {

class PreparedMemoryBudget {
 public:
  explicit PreparedMemoryBudget(std::size_t limit) : limit_(limit) {}
  void require(std::size_t bytes) const {
    if (bytes > limit_ - memory_.total_bytes())
      throw std::bad_alloc{};
  }
  static std::size_t product(std::size_t count, std::size_t size) {
    if (size != 0 && count > std::numeric_limits<std::size_t>::max() / size)
      throw std::bad_alloc{};
    return count * size;
  }
  void charge(std::size_t bytes, std::size_t PreparedSchemaMemory::* category) {
    require(bytes);
    memory_.*category += bytes;
  }
  const PreparedSchemaMemory& memory() const noexcept { return memory_; }

 private:
  std::size_t limit_;
  PreparedSchemaMemory memory_{};
};

// allocate_shared rebinds this allocator to its implementation-specific control block.
// Only cold allocation dereferences budget; deallocation never needs the budget to remain alive.
template <typename T>
struct PreparedOwnerAllocator {
  using value_type = T;
  PreparedMemoryBudget* budget;
  explicit PreparedOwnerAllocator(PreparedMemoryBudget& value) : budget(&value) {}
  template <typename U>
  PreparedOwnerAllocator(const PreparedOwnerAllocator<U>& other) : budget(other.budget) {}
  T* allocate(std::size_t count) {
    budget->charge(PreparedMemoryBudget::product(count, sizeof(T)),
                   &PreparedSchemaMemory::optional_bytes);
    return std::allocator<T>{}.allocate(count);
  }
  void deallocate(T* pointer, std::size_t count) noexcept {
    std::allocator<T>{}.deallocate(pointer, count);
  }
  template <typename U>
  bool operator==(const PreparedOwnerAllocator<U>& other) const noexcept {
    return budget == other.budget;
  }
};

template <typename T>
class PreparedOptionalOwner {
 public:
  void prepare(PreparedMemoryBudget& budget) {
    owner_ = std::allocate_shared<T>(PreparedOwnerAllocator<T>{budget});
  }
  bool valid(const std::shared_ptr<T>& value) const noexcept {
    if (!owner_)
      return false;
    if (!value)
      return value.use_count() == 0 && owner_.use_count() == 1;
    // Exactly the private inventory and the present native field may own this object at reuse.
    // A callback-local alias is harmless if destroyed before reset/assignment. Do not relax this
    // check for future internal holders without also proving they cannot observe in-place writes.
    return value.get() == owner_.get() && !value.owner_before(owner_) &&
           !owner_.owner_before(value) && owner_.use_count() == 2;
  }
  void select(std::shared_ptr<T>& value, bool present) const noexcept {
    if (present)
      value = owner_;
    else
      value.reset();
  }
  T& object() noexcept { return *owner_; }
  const T& object() const noexcept { return *owner_; }

 private:
  std::shared_ptr<T> owner_;
};

inline std::size_t prepared_string_heap_bytes(const std::string& value) {
  const auto buffer = reinterpret_cast<std::uintptr_t>(value.data());
  const auto object = reinterpret_cast<std::uintptr_t>(&value);
  if (buffer >= object && buffer - object < sizeof(value))
    return 0;
  if (value.capacity() == std::numeric_limits<std::size_t>::max())
    throw std::bad_alloc{};
  return value.capacity() + 1;
}

inline void prepare_string(std::string& value, std::size_t bound, PreparedMemoryBudget& budget) {
  if (bound > value.capacity()) {
    if (bound == std::numeric_limits<std::size_t>::max())
      throw std::bad_alloc{};
    budget.require(bound + 1);
    value.reserve(bound);
  }
  budget.charge(prepared_string_heap_bytes(value), &PreparedSchemaMemory::container_bytes);
}

template <typename T>
void prepare_vector(std::vector<T>& value, std::size_t bound, PreparedMemoryBudget& budget) {
  budget.require(PreparedMemoryBudget::product(bound, sizeof(T)));
  value.reserve(bound);
  budget.charge(PreparedMemoryBudget::product(value.capacity(), sizeof(T)),
                &PreparedSchemaMemory::container_bytes);
}

class PreparedString {
 public:
  void prepare(std::string& value, std::size_t bound, PreparedMemoryBudget& budget) {
    bound_ = bound;
    prepare_string(value, bound, budget);
    capacity_ = value.capacity();
  }
  bool valid(const std::string& value) const noexcept {
    return value.capacity() == capacity_ && value.size() <= bound_;
  }
  bool fits(std::size_t size) const noexcept { return size <= bound_; }
  void assign(std::string& value, const char* data, std::size_t size) const noexcept {
    if (size == 0)
      value.clear();
    else
      value.assign(data, size);
  }

 private:
  std::size_t bound_{};
  std::size_t capacity_{};
};

template <typename T>
class PreparedScalarVector {
  static_assert(std::is_arithmetic_v<T> || std::is_enum_v<T>);

 public:
  void prepare(std::vector<T>& value, std::size_t bound, PreparedMemoryBudget& budget) {
    bound_ = bound;
    prepare_vector(value, bound, budget);
    capacity_ = value.capacity();
  }
  bool valid(const std::vector<T>& value) const noexcept {
    return value.capacity() == capacity_ && value.size() <= bound_;
  }
  bool fits(std::size_t size) const noexcept { return size <= bound_; }
  template <typename Read>
  void assign(std::vector<T>& value, std::size_t size, Read read) const noexcept {
    value.resize(size);
    for (std::size_t i = 0; i < size; ++i)
      value[i] = static_cast<T>(read(i));
  }

 private:
  std::size_t bound_{};
  std::size_t capacity_{};
};

class PreparedStringVector {
 public:
  void prepare(std::vector<std::string>& value, std::size_t count, std::size_t bytes,
               PreparedMemoryBudget& budget) {
    bound_ = count;
    element_bound_ = bytes;
    prepare_vector(value, count, budget);
    capacity_ = value.capacity();
    prepare_vector(parked_, count, budget);
    prepare_vector(element_capacities_, count, budget);
    parked_.resize(count);
    element_capacities_.resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      prepare_string(parked_[i], bytes, budget);
      element_capacities_[i] = parked_[i].capacity();
    }
  }
  bool valid(const std::vector<std::string>& value) const noexcept {
    if (value.capacity() != capacity_ || value.size() > bound_)
      return false;
    // Include parked buffers: validation is O(max_elements), not O(current size). Large declared
    // bounds increase the cost of every copy/reset/assignment even for an empty string vector.
    for (std::size_t i = 0; i < bound_; ++i) {
      const auto& element = i < value.size() ? value[i] : parked_[i];
      if (element.capacity() != element_capacities_[i] || element.size() > element_bound_)
        return false;
    }
    return true;
  }
  bool fits(std::size_t size) const noexcept { return size <= bound_; }
  bool element_fits(std::size_t size) const noexcept { return size <= element_bound_; }
  void resize(std::vector<std::string>& value, std::size_t size) noexcept {
    const auto previous = value.size();
    if (size < previous) {
      for (std::size_t i = size; i < previous; ++i) {
        value[i].clear();
        value[i].swap(parked_[i]);
      }
      value.resize(size);
    } else {
      value.resize(size);
      for (std::size_t i = previous; i < size; ++i)
        value[i].swap(parked_[i]);
    }
  }

 private:
  std::size_t bound_{};
  std::size_t element_bound_{};
  std::size_t capacity_{};
  std::vector<std::string> parked_;
  std::vector<std::size_t> element_capacities_;
};

template <typename T>
concept HasPreparedSchema = requires { typename PreparedSchemaTraits<T>::Storage; };

template <typename T, bool Descriptor>
  requires HasPreparedSchema<T> && (PreparedSchemaTraits<T>::descriptor_tensor == Descriptor)
class NativeSchemaScratch {
  using Traits = PreparedSchemaTraits<T>;
  struct State {
    explicit State(std::size_t bytes) : budget(bytes) {}
    PreparedMemoryBudget budget;
    T value{};
    typename Traits::Storage storage;
  };

 public:
  /// @brief Cold-prepare storage, optionally applying a tighter charged-byte budget.
  /// @param max_bytes Caller limit; cannot increase the generated policy's limit.
  /// @return Scratch, or kResourceExhausted for insufficient memory/budget.
  [[nodiscard]] static expected<NativeSchemaScratch, Error> create(
      std::size_t max_bytes = Traits::max_prepared_bytes) noexcept {
    try {
      const auto limit = std::min(max_bytes, Traits::max_prepared_bytes);
      if (sizeof(State) > limit || sizeof(NativeSchemaScratch) > limit - sizeof(State))
        return make_unexpected(Error{ErrorCode::kResourceExhausted});
      auto state = std::make_unique<State>(limit);
      state->budget.charge(sizeof(T), &PreparedSchemaMemory::native_bytes);
      state->budget.charge(sizeof(State) - sizeof(T) + sizeof(NativeSchemaScratch),
                           &PreparedSchemaMemory::inventory_bytes);
      state->storage.prepare(state->value, state->budget);
      return NativeSchemaScratch{std::move(state)};
    } catch (const std::bad_alloc&) {
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
  }
  /// @brief Reject copying prepared storage.
  NativeSchemaScratch(const NativeSchemaScratch&) = delete;
  /// @brief Reject copy assignment of prepared storage.
  NativeSchemaScratch& operator=(const NativeSchemaScratch&) = delete;
  /// @brief Move prepared storage; the source supports destruction/reassignment only.
  NativeSchemaScratch(NativeSchemaScratch&&) noexcept = default;
  /// @brief Replace prepared storage by move.
  NativeSchemaScratch& operator=(NativeSchemaScratch&&) noexcept = default;
  /// @brief Release all private prepared storage.
  ~NativeSchemaScratch() = default;

  /// @brief Copy all ordinary fields without allocating or retaining source owners.
  /// @return PreparedStorageInvalid for altered/aliased storage, ResourceExhausted for bounds.
  [[nodiscard]] expected<void, Error> copy_from(const T& source) noexcept
    requires(!Descriptor)
  {  // NOLINT(whitespace/braces)
    return copy(source);
  }
  /// @brief Copy non-Tensor fields; every destination Tensor remains null.
  /// @return PreparedStorageInvalid for altered/aliased storage, ResourceExhausted for bounds.
  [[nodiscard]] expected<void, Error> copy_metadata_from(const T& source) noexcept
    requires(Descriptor)
  {  // NOLINT(whitespace/braces)
    return copy(source);
  }
  /// @brief Reset values and optional presence without releasing prepared capacity.
  /// @return PreparedStorageInvalid before mutation if storage-preservation rules were violated.
  [[nodiscard]] expected<void, Error> reset() noexcept {
    if (!valid())
      return make_unexpected(Error{ErrorCode::kPreparedStorageInvalid});
    state_->storage.reset_unchecked(state_->value);
    return {};
  }
  /// @brief Return ordinary storage for capacity-preserving edits.
  T& value() noexcept
    requires(!Descriptor)
  {  // NOLINT(whitespace/braces)
    return state_->value;
  }
  /// @brief Return the prepared ordinary value.
  const T& value() const noexcept
    requires(!Descriptor)
  {  // NOLINT(whitespace/braces)
    return state_->value;
  }
  /// @brief Return metadata-only storage for capacity-preserving edits.
  T& descriptor() noexcept
    requires(Descriptor)
  {  // NOLINT(whitespace/braces)
    return state_->value;
  }
  /// @brief Return the prepared metadata-only descriptor.
  const T& descriptor() const noexcept
    requires(Descriptor)
  {  // NOLINT(whitespace/braces)
    return state_->value;
  }
  /// @brief Return cold charged storage, excluding allocator overhead and cold temporary peaks.
  const PreparedSchemaMemory& memory_usage() const noexcept { return state_->budget.memory(); }

 private:
  explicit NativeSchemaScratch(std::unique_ptr<State> state) : state_(std::move(state)) {}
  bool valid() const noexcept {
    return state_ && state_->storage.valid(state_->value) && Traits::tensors_empty(state_->value);
  }
  expected<void, Error> copy(const T& source) noexcept {
    if (!valid())
      return make_unexpected(Error{ErrorCode::kPreparedStorageInvalid});
    if (!state_->storage.source_fits(source))
      return make_unexpected(Error{ErrorCode::kResourceExhausted});
    if (&source != &state_->value)
      state_->storage.copy_unchecked(state_->value, source);
    return {};
  }
  std::unique_ptr<State> state_;
};

}  // namespace detail

/// @brief Own bounded reusable native metadata for an explicitly prepared ordinary schema.
/// @warning Preserve private storage: do not replace/escape optional owners, assign a whole T,
/// move/swap containers, or destructively resize string vectors. Use copy_from()/reset() for
/// structural changes; scalar, inline-struct, and existing bounded-string edits are supported.
/// A const generated shared_ptr does not make its pointee immutable. Never edit borrowed input.
/// Do not keep copies of scratch's nested shared_ptr owners across copy_from()/reset(); the next
/// reuse fails closed with ErrorCode::kPreparedStorageInvalid until those aliases are released.
/// Copy ordinary values into independent storage when they must outlive scratch reuse.
template <typename T>
using SchemaScratch = detail::NativeSchemaScratch<T, false>;

/// @brief Own bounded metadata without retaining any direct Tensor from the source descriptor.
/// @warning Observe SchemaScratch's storage-preservation rules and keep all Tensor fields null.
/// Read input Tensors through the activation's selected sample; publish with separate output loans.
template <typename T>
using DescriptorMetadataScratch = detail::NativeSchemaScratch<T, true>;

}  // namespace holoscan::schema
