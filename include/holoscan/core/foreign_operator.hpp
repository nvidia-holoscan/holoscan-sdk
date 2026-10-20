/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/parameter.hpp>

namespace holoscan {

class Contract;
class ExecutionContext;
class LifecycleContext;
class OperatorSpec;

/// @brief Bound free-form callback message bytes copied by ForeignOperatorCore.
inline constexpr std::size_t kForeignOperatorMessageCapacity = kDiagnosticMessageCapacity;

/// @brief Return one allocation-free result across a foreign Operator callback boundary.
///
/// The callback retains ownership of `message`. For a failure result, its storage must remain valid
/// until the invoking ForeignOperatorCore method returns. Core copies at most
/// `kForeignOperatorMessageCapacity` bytes after the callback returns and never reads the pointer
/// for a successful result. `message_size` must not exceed the readable storage beginning at
/// `message`; the capacity clamp is not pointer validation.
struct ForeignOperatorResult {
  ErrorCode code{ErrorCode::kSuccess};  ///< Callback outcome.
  const char* message{};                ///< Optional callback-owned message bytes.
  std::size_t message_size{};           ///< Available bytes beginning at `message`.

  /// @brief Construct a successful callback result.
  /// @return The allocation-free success record.
  [[nodiscard]] static constexpr ForeignOperatorResult success() noexcept { return {}; }

  /// @brief Construct one borrowed failure record.
  /// @param error_code Non-success Core error category.
  /// @param detail Callback-owned bytes valid until the invoking ForeignOperatorCore method
  /// returns.
  /// @param detail_size Number of readable bytes beginning at \p detail.
  /// @return Failure record whose message is clamped when Core consumes it.
  [[nodiscard]] static constexpr ForeignOperatorResult failure(ErrorCode error_code,
                                                               const char* detail,
                                                               std::size_t detail_size) noexcept {
    return ForeignOperatorResult{error_code, detail, detail_size};
  }

  /// @brief Construct one borrowed failure record from static literal storage.
  // An array reference retains the string literal extent without copying its storage.
  template <std::size_t Size>
  [[nodiscard]] static constexpr ForeignOperatorResult failure(
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays)
      ErrorCode error_code, const char (&detail)[Size]) noexcept {
    static_assert(Size != 0U);
    return ForeignOperatorResult{error_code, detail, Size - 1U};
  }
};

/// @brief Declare stable typed ports for one foreign-hosted Operator setup pass.
///
/// A language adapter resolves its payload declarations to process-local registry rows before
/// calling these methods. Each success retains the allocated typed port for the owning
/// ForeignOperatorCore instance. The callback must not retain this setup-only object or call it
/// after the setup callback returns.
class HOLOSCAN_CORE_EXPORT ForeignOperatorSpec {
 public:
  /// @brief Allocate and bind one input using a registered payload row.
  /// @param payload Exact process-local payload registration.
  /// @param name Stable local port name copied into the declaration.
  /// @param queue_depth Optional finite input history capacity.
  /// @param optional Whether this input may remain unwired under an eligible temporal contract.
  /// @return Stable erased port reference or the typed public OperatorSpec failure as data.
  [[nodiscard]] expected<authoring::OpaquePortRef, Error> input(
      const authoring::RegisteredPayloadType& payload, std::string_view name,
      std::optional<std::uint32_t> queue_depth = std::nullopt, bool optional = false) noexcept;

  /// @brief Allocate and bind one output using a registered payload row.
  /// @param payload Exact process-local payload registration.
  /// @param name Stable local port name copied into the declaration.
  /// @return Stable erased port reference or the typed public OperatorSpec failure as data.
  [[nodiscard]] expected<authoring::OpaquePortRef, Error> output(
      const authoring::RegisteredPayloadType& payload, std::string_view name,
      std::optional<std::uint32_t> max_emits_per_compute = std::nullopt) noexcept;

 private:
  ForeignOperatorSpec(OperatorSpec& spec, std::vector<authoring::OpaquePortRef>& ports) noexcept
      : spec_(&spec), ports_(&ports) {}

  OperatorSpec* spec_{};
  std::vector<authoring::OpaquePortRef>* ports_{};

  friend class ForeignOperatorCore;
};

/// @brief Provide the raw C-shaped callback table for one foreign Operator instance.
///
/// Every callback must be non-throwing. A null callback requests the native no-op/default behavior,
/// except `compute_cb`, which is required. The three advanced lifecycle callbacks are optional as
/// a group, but supplying any one requires supplying all three. The context and function pointers
/// must remain valid until `destroy` is called exactly once after the final copied capture is
/// released.
struct ForeignOperatorCallbackTable {
  void* context{};                            ///< Opaque adapter-owned instance state.
  void (*destroy)(void*) noexcept = nullptr;  ///< Optional final context release hook.
  /// Declare ports during each cold setup replay.
  ForeignOperatorResult (*setup_cb)(void*, ForeignOperatorSpec*) noexcept = nullptr;
  /// Return the temporal contract during compile and session-start replay.
  ForeignOperatorResult (*contract_cb)(void*, Contract*) noexcept = nullptr;
  /// Perform one runtime activation.
  ForeignOperatorResult (*compute_cb)(void*, ExecutionContext*) noexcept = nullptr;
  /// Perform optional lifecycle start work.
  ForeignOperatorResult (*start_cb)(void*) noexcept = nullptr;
  /// Perform optional lifecycle stop work.
  ForeignOperatorResult (*stop_cb)(void*) noexcept = nullptr;
  /// Report whether one advanced lifecycle-stage callback was declared during setup.
  bool (*has_lifecycle_cb)(void*, LifecycleStage) noexcept = nullptr;
  /// Report whether one declared advanced lifecycle-stage callback is required.
  bool (*lifecycle_required_cb)(void*, LifecycleStage) noexcept = nullptr;
  /// Invoke one declared advanced lifecycle-stage callback.
  ForeignOperatorResult (*lifecycle_cb)(void*, LifecycleStage, LifecycleContext*,
                                        LifecycleStatus*) noexcept = nullptr;
};

static_assert(std::is_standard_layout_v<ForeignOperatorResult>);
static_assert(std::is_trivially_copyable_v<ForeignOperatorResult>);
static_assert(std::is_standard_layout_v<ForeignOperatorCallbackTable>);
static_assert(std::is_trivially_copyable_v<ForeignOperatorCallbackTable>);

/// @brief Own a copyable foreign callback capture with exactly-once context destruction.
class HOLOSCAN_CORE_EXPORT ForeignOperatorCallbacks {
 public:
  /// @brief Adopt one raw callback table.
  /// @param table Complete table whose context is released through its destroy hook.
  /// @throws RuntimeError when the table has no compute callback or an incomplete advanced
  /// lifecycle callback group.
  explicit ForeignOperatorCallbacks(ForeignOperatorCallbackTable table);

  ForeignOperatorCallbacks(const ForeignOperatorCallbacks&) noexcept = default;
  ForeignOperatorCallbacks& operator=(const ForeignOperatorCallbacks&) noexcept = default;
  ForeignOperatorCallbacks(ForeignOperatorCallbacks&&) noexcept = default;
  ForeignOperatorCallbacks& operator=(ForeignOperatorCallbacks&&) noexcept = default;
  ~ForeignOperatorCallbacks() = default;

 private:
  struct State;
  std::shared_ptr<State> state_;

  [[nodiscard]] const ForeignOperatorCallbackTable& table() const noexcept;

  friend class ForeignOperatorCore;
};

/// @brief Configure plan-visible hosting and serialization behavior for a foreign Operator.
struct ForeignOperatorOptions {
  /// Shared callback serialization domain, or no value to opt out of compiler co-partitioning.
  ///
  /// Equal values ask automatic partitioning to place compatible hosts under one partition owner;
  /// this is a compile-time placement request, not a runtime mutex or graph dependency. A manual or
  /// topology-required split is permitted and produces an advisory. A present value contains 1
  /// through 128 ASCII alphanumeric, period, underscore, or hyphen characters. Invalid values are
  /// rejected by ForeignOperatorCore construction.
  std::optional<std::string> serialization_domain;
};

/// @brief Host one language-neutral Operator through a non-throwing callback table.
class HOLOSCAN_CORE_EXPORT ForeignOperatorCore : public Operator<NoParams> {
 public:
  /// @brief Construct one hosted Operator from a copyable callback capture.
  /// @param callbacks Owned callback capture copied by the ordinary Graph factory.
  /// @param options Plan-visible hosting and serialization declarations.
  explicit ForeignOperatorCore(ForeignOperatorCallbacks callbacks,
                               ForeignOperatorOptions options = {});

  ~ForeignOperatorCore() override = default;

  void setup(OperatorSpec& spec) override;
  [[nodiscard]] Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] expected<void, Error> compute(ExecutionContext& context) override;
  [[nodiscard]] OperatorHostMetadata host_metadata() const noexcept override;

 protected:
  /// @brief Return whether the foreign adapter declared one advanced lifecycle callback.
  /// @param stage Exact stage queried after the setup callback completed.
  /// @return `true` only when the adapter declared \p stage and supplied a dispatcher.
  [[nodiscard]] bool has_foreign_lifecycle_callback(LifecycleStage stage) const noexcept;

  /// @brief Return whether one declared foreign lifecycle callback is required.
  /// @param stage Exact declared advanced lifecycle stage.
  /// @return `true` for a required callback and `false` for an optional or absent callback.
  [[nodiscard]] bool foreign_lifecycle_callback_required(LifecycleStage stage) const noexcept;

  /// @brief Invoke one foreign advanced lifecycle callback without crossing the ABI with an
  /// exception.
  /// @param stage Exact stage registered by the concrete hosted Operator.
  /// @param context Borrowed callback-lifetime lifecycle services.
  /// @return The adapter-provided status, or fatal failure for an invalid callback result.
  [[nodiscard]] LifecycleStatus invoke_foreign_lifecycle_callback(
      LifecycleStage stage, LifecycleContext& context) noexcept;

 private:
  [[nodiscard]] Error callback_error(ForeignOperatorResult result) const;

  ForeignOperatorCallbacks callbacks_;
  ForeignOperatorOptions options_;
  std::vector<authoring::OpaquePortRef> ports_;
};

}  // namespace holoscan
