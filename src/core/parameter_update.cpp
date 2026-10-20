/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <holoscan/core/parameter_update.hpp>

namespace holoscan {
namespace detail {

struct ParamPatchValueStorage {
  using Value =
      std::variant<std::string, std::vector<ParamPatchValue>, std::vector<ParamPatchAssignment>>;

  explicit ParamPatchValueStorage(Value value) : value(std::move(value)) {}

  Value value;
};

}  // namespace detail

ParamPatchValue::ParamPatchValue(ParamValue value) noexcept
    : kind_(value.kind() == ParamScalarKind::kNone ? ParamPatchValueKind::kUnset
                                                   : ParamPatchValueKind::kScalar),
      scalar_(value.scalar()) {}

ParamPatchValue ParamPatchValue::none() noexcept {
  return ParamPatchValue{ParamPatchValueKind::kNone, {}, {}};
}

expected<ParamPatchValue, Error> ParamPatchValue::from_string(std::string_view value) noexcept {
  try {
    auto storage = std::make_shared<detail::ParamPatchValueStorage>(std::string{value});
    return ParamPatchValue{ParamPatchValueKind::kString, {}, std::move(storage)};
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  } catch (const std::length_error&) {
    return make_unexpected(Error{ErrorCode::kInvalidArgument});
  }
}

expected<ParamPatchValue, Error> ParamPatchValue::from_sequence(
    std::vector<ParamPatchValue> values) noexcept {
  try {
    auto storage = std::make_shared<detail::ParamPatchValueStorage>(std::move(values));
    return ParamPatchValue{ParamPatchValueKind::kSequence, {}, std::move(storage)};
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
}

expected<ParamPatchValue, Error> ParamPatchValue::from_object(
    std::vector<ParamPatchAssignment> values) noexcept {
  try {
    auto storage = std::make_shared<detail::ParamPatchValueStorage>(std::move(values));
    return ParamPatchValue{ParamPatchValueKind::kObject, {}, std::move(storage)};
  } catch (const std::bad_alloc&) {
    return make_unexpected(Error{ErrorCode::kResourceExhausted});
  }
}

std::string_view ParamPatchValue::string() const noexcept {
  if (kind_ != ParamPatchValueKind::kString || storage_ == nullptr) {
    return {};
  }
  return std::get<std::string>(storage_->value);
}

std::span<const ParamPatchValue> ParamPatchValue::sequence() const noexcept {
  if (kind_ != ParamPatchValueKind::kSequence || storage_ == nullptr) {
    return {};
  }
  return std::get<std::vector<ParamPatchValue>>(storage_->value);
}

std::span<const ParamPatchAssignment> ParamPatchValue::object() const noexcept {
  if (kind_ != ParamPatchValueKind::kObject || storage_ == nullptr) {
    return {};
  }
  return std::get<std::vector<ParamPatchAssignment>>(storage_->value);
}

}  // namespace holoscan
