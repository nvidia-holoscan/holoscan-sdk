/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

#include <holoscan/core/operator.hpp>

#if defined(_WIN32)
#if defined(parameter_dso_test_plugin_EXPORTS)
#define HOLOSCAN_PARAMETER_DSO_TEST_EXPORT __declspec(dllexport)
#else
#define HOLOSCAN_PARAMETER_DSO_TEST_EXPORT __declspec(dllimport)
#endif
#define HOLOSCAN_PARAMETER_DSO_TEST_HIDDEN
#else
#define HOLOSCAN_PARAMETER_DSO_TEST_EXPORT __attribute__((visibility("default")))
#define HOLOSCAN_PARAMETER_DSO_TEST_HIDDEN __attribute__((visibility("hidden")))
#endif

namespace holoscan::test {

enum class HOLOSCAN_PARAMETER_DSO_TEST_HIDDEN DSOFieldValue : std::uint32_t {
  kDefault = 42,
};

}  // namespace holoscan::test

namespace holoscan::detail {

template <>
HOLOSCAN_PARAMETER_DSO_TEST_HIDDEN inline constexpr std::byte
    kParameterTypeToken<test::DSOFieldValue>{};

}  // namespace holoscan::detail

namespace holoscan::test {

struct DSOParams {
  DSOFieldValue value{DSOFieldValue::kDefault};
};

HOLOSCAN_PARAM_SCHEMA(DSOParams, param_schema("holoscan.test.DSOParams", 1),
                      field<&DSOParams::value>("value").runtime_mutable());

class DSOParameterOperator final : public Operator<DSOParams> {
 public:
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;
  [[nodiscard]] const DSOParams* observed_params() const noexcept { return observed_params_; }

 private:
  const DSOParams* observed_params_{};
};

extern "C" {
[[nodiscard]] HOLOSCAN_PARAMETER_DSO_TEST_EXPORT DSOParameterOperator*
create_dso_parameter_operator();
HOLOSCAN_PARAMETER_DSO_TEST_EXPORT void destroy_dso_parameter_operator(
    DSOParameterOperator* op) noexcept;
[[nodiscard]] HOLOSCAN_PARAMETER_DSO_TEST_EXPORT const void* dso_aggregate_type_token() noexcept;
[[nodiscard]] HOLOSCAN_PARAMETER_DSO_TEST_EXPORT const void* dso_field_type_token() noexcept;
}

}  // namespace holoscan::test

#undef HOLOSCAN_PARAMETER_DSO_TEST_EXPORT
#undef HOLOSCAN_PARAMETER_DSO_TEST_HIDDEN
