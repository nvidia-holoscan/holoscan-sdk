/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <memory>
#include <utility>

#include <holoscan/core/parameter_override_set.hpp>

#include "parameter_override_set_internal.hpp"

namespace holoscan {

ParamOverrideSet::ParamOverrideSet(
    std::shared_ptr<const detail::ParamOverrideSetState> state) noexcept
    : state_(std::move(state)) {}

ParamOverrideSet::ParamOverrideSet(const ParamOverrideSet& other) noexcept = default;
ParamOverrideSet::ParamOverrideSet(ParamOverrideSet&& other) noexcept = default;
ParamOverrideSet& ParamOverrideSet::operator=(const ParamOverrideSet& other) noexcept = default;
ParamOverrideSet& ParamOverrideSet::operator=(ParamOverrideSet&& other) noexcept = default;
ParamOverrideSet::~ParamOverrideSet() = default;

const ParamRebindProvenance& ParamOverrideSet::provenance() const noexcept {
  return state_->provenance;
}

}  // namespace holoscan
