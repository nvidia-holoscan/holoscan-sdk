/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <string_view>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace holoscan::authoring::examples {

/// @brief Carry the named constructor configuration mapped by the hand-authored rows.
struct ManualRowSourceParameters {
  std::uint32_t limit{4U};  ///< Maximum sequential value emitted by the source.
};

HOLOSCAN_PARAM_SCHEMA(
    ManualRowSourceParameters,
    param_schema("holoscan.test.manual_row.ManualRowSourceParameters", 1),
    field<&ManualRowSourceParameters::limit>("limit")
        .range(1U, 1'000'000U)
        .description("Maximum number of sequential values emitted by the source."));

/// @brief Demonstrate registration of a deliberately constructor-configured Operator.
///
/// This test/example type is the canonical pointable implementation for a third-party Operator
/// that cannot use `HOLOSCAN_REGISTER_OPERATOR`. Its rows explicitly map the named `limit`
/// assignment to this constructor; no positional or language-runtime coercion is involved.
/// Representative test: `AuthoringManualRow.MatchesTypedAuthoringAndExecutionForBothRows`.
class ManualRowSource final : public Operator<> {
 public:
  /// @brief Construct a bounded counter source from one required positional C++ value.
  /// @param limit Maximum sequential value to emit.
  explicit ManualRowSource(std::uint32_t limit) noexcept : limit_(limit) {}

  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override {
    if (next_ >= limit_) {
      return {};
    }
    const std::uint32_t value = next_ + 1U;
    auto emitted = output.emit(value);
    if (emitted) {
      next_ = value;
    }
    return emitted;
  }

  Output<std::uint32_t> output;

 private:
  std::uint32_t limit_{};
  std::uint32_t next_{};
};

inline constexpr std::string_view kManualRowWithDescriptorName{
    "holoscan.test.manual_row.ManualRowSourceWithDescriptor"};
inline constexpr std::string_view kManualRowWithoutDescriptorName{
    "holoscan.test.manual_row.ManualRowSourceWithoutDescriptor"};

}  // namespace holoscan::authoring::examples
