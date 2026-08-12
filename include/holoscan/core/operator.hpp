/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <concepts>
#include <cstring>
#include <type_traits>
#include <utility>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema.hpp>
#include <holoscan/core/temporal_contract.hpp>

// Native MSVC preserves its stable ABI by ignoring the standard attribute unless the
// vendor-qualified spelling is used. Keep the workaround local to this header so the public
// macro namespace does not become part of the API.
#if defined(_MSC_VER)
#define HOLOSCAN_DETAIL_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define HOLOSCAN_DETAIL_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

namespace holoscan {

class ExecutionContext;
class OperatorSpec;

namespace detail {
class GraphOperatorAccess;
class OperatorParameterBinding;
class OperatorRuntimeAccess;
class SyntheticOperatorBinding;
}  // namespace detail

/// @brief Define the non-templated lifecycle and compute extension points for an Operator.
///
/// An Operator is a stable noncopyable, nonmovable authoring object. `setup()` records cold
/// metadata only. When all required runtime bindings are admitted, Runtime invokes an overridden
/// `start()` whenever its `kStart` checkpoint is eligible during initial startup or a restart,
/// invokes `compute()` for frozen activations, and invokes an overridden `stop()` whenever its
/// `kStop` checkpoint is eligible during startup rollback, restart processing, or final teardown.
/// Missing bindings fail before callbacks rather than selecting weaker lifecycle or scheduling
/// semantics.
class HOLOSCAN_CORE_EXPORT OperatorBase {
 public:
  /// @brief Construct an unbound Operator authoring object.
  OperatorBase();

  /// @brief Destroy an Operator after its owning graph and run bindings have retired.
  virtual ~OperatorBase();

  /// @brief Reject copying a stable Operator authoring object.
  /// @param other Operator that would otherwise be copied.
  OperatorBase(const OperatorBase& other) = delete;

  /// @brief Reject copy assignment of a stable Operator authoring object.
  /// @param other Operator that would otherwise replace this one.
  OperatorBase& operator=(const OperatorBase& other) = delete;

  /// @brief Reject moving a stable Operator authoring object.
  /// @param other Operator that would otherwise be moved.
  OperatorBase(OperatorBase&& other) = delete;

  /// @brief Reject move assignment of a stable Operator authoring object.
  /// @param other Operator that would otherwise replace this one.
  OperatorBase& operator=(OperatorBase&& other) = delete;

  /// @brief Declare stable ports and other cold metadata by reference.
  /// @param spec Fresh declaration pass owned by the compiler or private conformance fixture.
  /// @note Repeated calls receive fresh specs and must describe the same deterministic contract.
  virtual void setup(OperatorSpec& spec) { (void)spec; }

  /// @brief Return this Operator type's complete cold temporal contract.
  /// @return Fresh deep-owning descriptor; an empty value requests the inputful
  /// compile-time \ref OnAnyInput default and is invalid for a zero-input Operator.
  /// @note Runtime does not retain this authoring object's address. Compilation
  /// calls this operation while replaying an owning Operator factory and freezes
  /// the validated descriptor into pointer-free plan records.
  [[nodiscard]] virtual Contract contract() const { return {}; }

  /// @brief Enter steady-state execution after admitted startup barriers.
  /// @note An override must remain public and keep the exact `void start()` signature. Graph
  /// authoring lowers it into the single required \ref LifecycleStage::kStart checkpoint; an
  /// Operator cannot also register an advanced callback for that stage. Cold preparation belongs
  /// to explicit lifecycle stages added by their owning component. For a typed session-owned
  /// Operator, `params()` exposes the exact parameter generation selected for this phase.
  virtual void start() {}

  /// @brief Leave steady-state execution before resource release.
  /// @note An override must remain public and keep the exact `void stop()` signature. Graph
  /// authoring lowers it into the single required \ref LifecycleStage::kStop checkpoint; an
  /// Operator cannot also register an advanced callback for that stage. For a typed session-owned
  /// Operator, `params()` exposes the exact parameter generation selected for this phase,
  /// including the restored old-published view during rollback.
  virtual void stop() {}

  /// @brief Process one frozen activation candidate.
  /// @param context Borrowed callback-lifetime runtime services and activation evidence.
  /// @return Success after Operator work completes, or a typed failure for Runtime policy.
  /// @note The callback is invoked at most once for one activation tag. It must not wait for or
  /// recursively trigger another activation.
  [[nodiscard]] virtual expected<void, Error> compute(ExecutionContext& context) = 0;

 private:
  [[nodiscard]] expected<void, Error> invoke_compute(ExecutionContext& context);
  [[nodiscard]] const ParamTypeDescriptor& defining_param_type() const noexcept;
  [[nodiscard]] const ParamSchemaDescriptor* defining_param_schema() const noexcept;
  [[nodiscard]] virtual const ParamTypeDescriptor& defining_param_type_impl() const noexcept = 0;
  [[nodiscard]] virtual const ParamSchemaDescriptor* defining_param_schema_impl()
      const noexcept = 0;
  [[nodiscard]] virtual const void* parameter_address_impl() const noexcept = 0;
  virtual void replace_runtime_parameter_value_impl(const void* value) noexcept = 0;
  [[nodiscard]] virtual bool has_no_params_impl() const noexcept = 0;

  friend class detail::OperatorParameterBinding;
  friend class detail::OperatorRuntimeAccess;
  friend class detail::SyntheticOperatorBinding;
};

/// @brief Define a typed Operator with one immutable parameter aggregate.
/// @tparam ParamsT Complete unqualified parameter type, or `NoParams` for no configuration.
///
/// The concrete Operator remains noncopyable and nonmovable through `OperatorBase`. During normal
/// execution, `ExecutionContext::params<ParamsT>()` provides the callback-pinned generation. This
/// object's `params()` accessor exposes its authoring default before session binding. On a
/// session-owned instance, Runtime replaces the stored aggregate while callbacks are quiesced so
/// legacy `start()` and `stop()` observe the exact phase-selected generation. Advanced lifecycle
/// callbacks instead consume their pinned view through `LifecycleContext`; compute callbacks use
/// `ExecutionContext`. A parameter aggregate need not be nothrow-swappable to provide typed
/// callback storage. The private EA1 generation-zero whole-aggregate replacement seam may reject a
/// non-default candidate when it cannot publish that candidate with a non-throwing swap; later
/// compiler-selected materialization strategies own broader replacement support.
template <typename ParamsT = NoParams>
  requires ParameterAggregate<ParamsT>
class Operator : public OperatorBase {
 public:
  /// @brief Name the exact immutable parameter aggregate type.
  using Params = ParamsT;

  /// @brief Construct an Operator with its default authoring parameter aggregate.
  Operator() = default;

  /// @brief Destroy the typed Operator after inherited runtime bindings have retired.
  ~Operator() override = default;

  /// @brief Return this Operator instance's currently stored parameter aggregate.
  /// @return Process-static shared `NoParams` value, or a stable reference to this instance's
  /// aggregate.
  /// @note Before session binding, the aggregate is the authoring default. During a legacy
  /// `start()` or `stop()` callback on a session-owned instance, it is the exact generation
  /// selected for that lifecycle phase. Advanced lifecycle callbacks must use `LifecycleContext`
  /// and compute callbacks must use `ExecutionContext::params<ParamsT>()` for their pinned views.
  /// Runtime may replace the stored aggregate between quiesced lifecycle callbacks, so callers
  /// must not retain the returned reference across callback boundaries. This operation does not
  /// allocate or synchronize.
  [[nodiscard]] const Params& params() const noexcept {
    if constexpr (std::same_as<Params, NoParams>) {
      return detail::kNoParams;
    } else {
      return params_;
    }
  }

 private:
  [[nodiscard]] const ParamTypeDescriptor& defining_param_type_impl() const noexcept override {
    return param_type_descriptor<Params>();
  }

  [[nodiscard]] const ParamSchemaDescriptor* defining_param_schema_impl() const noexcept override {
    return param_schema_descriptor<Params>();
  }

  [[nodiscard]] const void* parameter_address_impl() const noexcept override {
    if constexpr (std::same_as<Params, NoParams>) {
      return &detail::kNoParams;
    } else {
      return &params_;
    }
  }

  void replace_runtime_parameter_value_impl(const void* value) noexcept override {
    if constexpr (!std::same_as<Params, NoParams>) {
      // EA1 runtime generations admit only trivially copyable complete aggregates. Updating the
      // session-owned Operator copy while callbacks are quiesced lets legacy start()/stop()
      // callbacks observe their exact phase view without exposing a second public lifecycle
      // context.
      if constexpr (std::is_trivially_copyable_v<Params>) {
        if (value != &params_) {
          std::memcpy(&params_, value, sizeof(Params));
        }
      }
    }
  }

  [[nodiscard]] bool has_no_params_impl() const noexcept override {
    return std::same_as<Params, NoParams>;
  }

  void bind_initial_params(Params& value) noexcept {
    static_assert(std::is_nothrow_swappable_v<Params>);
    using std::swap;
    swap(params_, value);
  }

  HOLOSCAN_DETAIL_NO_UNIQUE_ADDRESS Params params_{};

  friend class detail::OperatorParameterBinding;
  friend class detail::GraphOperatorAccess;
};

}  // namespace holoscan

#undef HOLOSCAN_DETAIL_NO_UNIQUE_ADDRESS
