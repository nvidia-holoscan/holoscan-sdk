/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/tensor_field.hpp>
#include <holoscan/schema/schema_package.hpp>
#include <holoscan/testing/tensor_snapshot.hpp>

namespace holoscan::testing {
namespace detail {
template <typename T>
struct SchemaSnapshotAccess;
}  // namespace detail

/// @brief Own metadata-only schema content and independently copied present Tensor fields.
/// @tparam T Exact provider-qualified native descriptor schema.
template <PublishedPayload T>
class SchemaSnapshot {
 public:
  /// @brief Return a fresh deep copy of metadata with every Tensor member unset.
  /// @note This value describes the source and is not a complete payload to republish.
  /// @throws RuntimeError if the provider cannot clone metadata.
  [[nodiscard]] T metadata() const {
    auto cloned = DescriptorTensorSchemaNativeTraits<T>::clone_metadata(*metadata_);
    if (!cloned)
      throw RuntimeError{std::move(cloned).error()};
    return std::move(*cloned);
  }
  /// @brief Inspect original field presence by canonical typed selector.
  /// @param field Selector from this schema's generated field inventory.
  /// @throws std::invalid_argument if the selector does not belong to this schema.
  [[nodiscard]] bool has_tensor(TensorFieldSelector<T> field) const {
    check_field(field);
    return tensors_[field.tensor_index].has_value();
  }
  /// @brief Return one present field's content; reject absent or foreign selectors.
  /// @param field Canonical typed field selector.
  /// @throws std::invalid_argument if the selector does not belong to this schema.
  /// @throws std::logic_error if the selected field was absent.
  [[nodiscard]] const TensorSnapshot& tensor(TensorFieldSelector<T> field) const {
    check_field(field);
    const auto& result = tensors_[field.tensor_index];
    if (!result)
      throw std::logic_error{"operator test snapshot Tensor field was absent"};
    return *result;
  }
  /// @brief Return original semantic sample metadata.
  [[nodiscard]] const SampleMetadata& sample_metadata() const noexcept { return sample_metadata_; }

 private:
  SchemaSnapshot() = default;
  void check_field(TensorFieldSelector<T> field) const {
    const auto& ids = DescriptorTensorSchemaNativeTraits<T>::tensor_field_ids;
    if (field.tensor_index >= ids.size() || ids[field.tensor_index] != field.field_id) {
      throw std::invalid_argument{"operator test snapshot Tensor selector mismatch"};
    }
  }
  std::shared_ptr<const T> metadata_;
  std::vector<std::optional<TensorSnapshot>> tensors_;
  SampleMetadata sample_metadata_{};
  friend struct detail::SchemaSnapshotAccess<T>;
};

namespace detail {
template <typename T>
struct SchemaSnapshotAccess {
  static expected<SchemaSnapshot<T>, Error> copy(const T& value, SampleMetadata metadata,
                                                 std::size_t maximum_bytes,
                                                 ExecutionContext& context) {
    using Traits = DescriptorTensorSchemaNativeTraits<T>;
    static_assert(
        requires { Traits::clone_metadata(value); },
        "schema snapshot requires a provider-qualified clone_metadata operation");
    auto cloned = Traits::clone_metadata(value);
    if (!cloned)
      return make_unexpected(std::move(cloned).error());
    SchemaSnapshot<T> result;
    result.metadata_ = std::make_shared<const T>(std::move(*cloned));
    result.sample_metadata_ = metadata;
    result.tensors_.resize(Traits::tensor_count);
    for (std::size_t index = 0; index < Traits::tensor_count; ++index) {
      const auto* tensor = Traits::tensor_at(value, index);
      if (!tensor)
        continue;
      cudaStream_t stream = nullptr;
      if (tensor->data() && (tensor->device().device_type == kDLCUDA ||
                             tensor->device().device_type == kDLCUDAManaged)) {
        stream = context.cuda_stream().get();
      }
      auto copied = capture_tensor_snapshot(*tensor, metadata, maximum_bytes, stream);
      if (!copied) {
        auto error = std::move(copied).error();
        error.message += "; tensor_field_id=" + std::to_string(Traits::tensor_field_ids[index]);
        return make_unexpected(std::move(error));
      }
      maximum_bytes -= copied->bytes().size();
      result.tensors_[index] = std::move(*copied);
    }
    return result;
  }
};
}  // namespace detail
}  // namespace holoscan::testing
