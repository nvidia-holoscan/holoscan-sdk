/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/foreign_operator.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_contract.hpp>

#include "native_operator.hpp"
#include "payload.hpp"

namespace holoscan {

namespace nb = nanobind;

// Immutable class-level declaration used to create one run-specific native port binding.
struct NativePortDefinition {
  std::string attribute_name;
  std::string port_name;
  PortDirection direction{PortDirection::kInput};
  const authoring::RegisteredPayloadType* payload{};
  GilSafeObject view;
  GilSafeObject schema_type;
};

class PyNativeOperatorBindings;
class PyLifecycleHookBindings;
class PySample;
class PySelectedSample;
class PySelectedBatch;
class PyTensorLoan;
class PyTensorSchemaLoan;
class PyTensorInputSpec;
class PyTensorOutputSpec;
struct NativePortBinding;

[[nodiscard]] nb::object python_sample(authoring::ForeignScalarSample sample);

[[nodiscard]] nb::object python_sample(const authoring::ForeignObjectSample& sample);

[[nodiscard]] nb::object python_sample(const authoring::ForeignSchemaSample& sample,
                                       nb::handle schema_type);

[[nodiscard]] nb::object python_sample(authoring::ForeignTensorSample sample);

using ParameterByteSpanResolver = std::function<TensorByteSpanExpression(nb::handle, bool)>;
using ParameterMemoryKindResolver =
    std::function<TensorMemoryKindExpression(nb::handle, MemoryKind, MemoryKind)>;

// Python view of a declared port that resolves its run-specific Core reference on each operation.
class PyNativePort {
 public:
  PyNativePort(std::shared_ptr<PyNativeOperatorBindings> bindings, std::string name);

  [[nodiscard]] const std::string& name() const noexcept;
  [[nodiscard]] PortDirection direction() const;
  [[nodiscard]] PyPayloadType payload_type() const;
  [[nodiscard]] InputSelection selection() const;
  [[nodiscard]] std::shared_ptr<PySelectedSample> selected_sample() const;
  [[nodiscard]] std::shared_ptr<PySelectedBatch> selected_batch() const;
  [[nodiscard]] PySample receive() const;
  [[nodiscard]] nb::object receive_data() const;
  void emit(nb::handle data, nb::handle options) const;
  void forward(const PySample& sample, nb::handle data, nb::handle options) const;
  void forward_selected(const PySelectedSample& sample, nb::handle options) const;
  [[nodiscard]] std::shared_ptr<PyTensorLoan> allocate_tensor(
      const std::vector<std::int64_t>& shape, nb::handle dtype, nb::handle strides,
      nb::handle options) const;
  [[nodiscard]] std::shared_ptr<PyTensorSchemaLoan> allocate_tensors(nb::handle descriptor,
                                                                     const nb::dict& requests,
                                                                     nb::handle options) const;
  PyNativePort& produces_tensor(const PyTensorOutputSpec& spec,
                                std::optional<TensorByteSpanExpression> byte_span,
                                std::optional<TensorMemoryKindExpression> memory_kind,
                                std::optional<std::string> field);
  PyNativePort& expects_tensor(const PyTensorInputSpec& spec);
  PyNativePort& accepts_projected_tensor();
  PyNativePort& expected_rate(std::uint64_t numerator, std::uint64_t denominator);
  PyNativePort& nominal_rate(std::uint64_t numerator, std::uint64_t denominator);
  PyNativePort& follows(const PyNativePort& input, std::uint32_t numerator,
                        std::uint32_t denominator);
  PyNativePort& follows(const PyNativePort& input, RateRatio ratio);
  [[nodiscard]] TensorByteSpanExpression input_max_byte_span() const;

  [[nodiscard]] PortRef contract_input_ref() const;
  Contract& contract_input(Contract& contract, Exact selector) const;
  Contract& contract_input(Contract& contract, LatestWithin selector) const;
  Contract& contract_input(Contract& contract, DrainAll selector) const;
  Contract& contract_input(Contract& contract, BracketedWindow selector) const;

 private:
  [[nodiscard]] NativePortBinding& binding() const;
  [[nodiscard]] NativePortBinding& bound(PortDirection direction) const;
  void emit_value(nb::handle data, EmitOptions options) const;

  std::shared_ptr<PyNativeOperatorBindings> bindings_;
  std::string name_;
  // Entry in bindings_, looked up by name on first use.
  mutable NativePortBinding* binding_{};
};

[[nodiscard]] std::optional<NativePortDefinition> native_port_definition(
    nb::handle value, std::string attribute_name);

[[nodiscard]] std::shared_ptr<PyNativeOperatorBindings> make_native_operator_bindings(
    const std::vector<NativePortDefinition>& definitions, HostedParameterSnapshot parameters,
    ParameterByteSpanResolver parameter_byte_span_resolver,
    ParameterMemoryKindResolver parameter_memory_kind_resolver);

[[nodiscard]] TensorByteSpanExpression active_parameter_byte_span(nb::handle parameter, bool shape);

[[nodiscard]] TensorMemoryKindExpression active_parameter_memory_kind(nb::handle parameter,
                                                                      MemoryKind when_false,
                                                                      MemoryKind when_true);

void attach_native_operator_bindings(nb::handle instance,
                                     const std::shared_ptr<PyNativeOperatorBindings>& bindings);

void setup_native_operator(nb::handle callback, ForeignOperatorSpec& spec,
                           const std::shared_ptr<PyNativeOperatorBindings>& bindings,
                           PyLifecycleHookBindings& lifecycle);

void compute_native_operator(nb::handle callback, ExecutionContext& context,
                             const std::shared_ptr<PyNativeOperatorBindings>& bindings);

[[nodiscard]] nb::object python_authoritative_emit_options_from_native(const EmitOptions& options);

void init_native_io(nb::module_& module);

}  // namespace holoscan
