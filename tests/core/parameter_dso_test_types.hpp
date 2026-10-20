/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>
#include <optional>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/graph.hpp>

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
  kConfigured = 73,
  kPrepared = 81,
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

struct DSOOpaqueParams {
  std::uint32_t threshold{4U};
};

class DSOOpaqueParameterOperator final : public Operator<DSOOpaqueParams> {
 public:
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;
};

struct DSOContractParams {
  DSOFieldValue value{DSOFieldValue::kDefault};
};

#if defined(parameter_dso_test_plugin_EXPORTS)
HOLOSCAN_PARAM_SCHEMA(DSOContractParams, param_schema("holoscan.test.DSOContractParams", 1),
                      field<&DSOContractParams::value>("value").runtime_mutable());
#else
HOLOSCAN_PARAM_SCHEMA(DSOContractParams, param_schema("holoscan.test.DSOContractParams", 1),
                      field<&DSOContractParams::value>("value"));
#endif

class DSOContractParameterOperator final : public Operator<DSOContractParams> {
 public:
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;
};

class DSOParameterOperator final : public Operator<DSOParams> {
 public:
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;
  [[nodiscard]] const DSOParams* observed_params() const noexcept { return observed_params_; }

 private:
  const DSOParams* observed_params_{};
};

struct DSOProjectorParams {
  std::uint32_t bytes{64U};
};

HOLOSCAN_PARAM_SCHEMA(DSOProjectorParams, param_schema("holoscan.test.DSOProjectorParams", 1),
                      field<&DSOProjectorParams::bytes>("bytes").recompile_required());

[[nodiscard]] HOLOSCAN_PARAMETER_DSO_TEST_EXPORT expected<std::uint64_t, Error>
dso_shared_byte_span_projector(const DSOProjectorParams& params) noexcept;

// Authors a projected Tensor output from the plugin side of a hidden-visibility boundary.
// ProjectorIdentityAcceptsOneExportedFunctionAcrossModules compares it with a host-side author.
class HOLOSCAN_PARAMETER_DSO_TEST_EXPORT DSOProjectedOperator final
    : public Operator<DSOProjectorParams> {
 public:
  void setup(OperatorSpec& spec) override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override;

  Output<Tensor> output;
};

class HOLOSCAN_PARAMETER_DSO_TEST_HIDDEN DSOParameterGraph final : public Graph {
 public:
  DSOParameterGraph();
  [[nodiscard]] OperatorHandle<DSOParameterOperator> parameter_operator() const;
  [[nodiscard]] OperatorHandle<DSOOpaqueParameterOperator> opaque_parameter_operator() const;
  [[nodiscard]] OperatorHandle<DSOContractParameterOperator> contract_parameter_operator() const;

 protected:
  void compose() override;

 private:
  std::optional<OperatorHandle<DSOParameterOperator>> parameter_operator_;
  std::optional<OperatorHandle<DSOOpaqueParameterOperator>> opaque_parameter_operator_;
  std::optional<OperatorHandle<DSOContractParameterOperator>> contract_parameter_operator_;
};

extern "C" {
[[nodiscard]] HOLOSCAN_PARAMETER_DSO_TEST_EXPORT DSOParameterOperator*
create_dso_parameter_operator();
HOLOSCAN_PARAMETER_DSO_TEST_EXPORT void destroy_dso_parameter_operator(
    DSOParameterOperator* op) noexcept;
[[nodiscard]] HOLOSCAN_PARAMETER_DSO_TEST_EXPORT const void* dso_aggregate_type_token() noexcept;
[[nodiscard]] HOLOSCAN_PARAMETER_DSO_TEST_EXPORT const void* dso_field_type_token() noexcept;
[[nodiscard]] HOLOSCAN_PARAMETER_DSO_TEST_EXPORT const void*
dso_opaque_aggregate_type_token() noexcept;
[[nodiscard]] HOLOSCAN_PARAMETER_DSO_TEST_EXPORT DSOParameterGraph* create_dso_parameter_graph();
HOLOSCAN_PARAMETER_DSO_TEST_EXPORT void destroy_dso_parameter_graph(
    DSOParameterGraph* graph) noexcept;
}

[[nodiscard]] HOLOSCAN_PARAMETER_DSO_TEST_EXPORT OperatorHandle<DSOParameterOperator>
dso_parameter_operator_handle(DSOParameterGraph* graph);
[[nodiscard]] HOLOSCAN_PARAMETER_DSO_TEST_EXPORT OperatorHandle<DSOOpaqueParameterOperator>
dso_opaque_parameter_operator_handle(DSOParameterGraph* graph);
[[nodiscard]] HOLOSCAN_PARAMETER_DSO_TEST_EXPORT OperatorHandle<DSOContractParameterOperator>
dso_contract_parameter_operator_handle(DSOParameterGraph* graph);

}  // namespace holoscan::test

#undef HOLOSCAN_PARAMETER_DSO_TEST_EXPORT
#undef HOLOSCAN_PARAMETER_DSO_TEST_HIDDEN
