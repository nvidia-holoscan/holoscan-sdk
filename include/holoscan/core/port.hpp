/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <array>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iterator>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include <holoscan/holoscan_core_export.h>
#include <holoscan/core/activation.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/core/tensor_schema_output_loan.hpp>
#include <holoscan/schema/schema_package.hpp>

namespace holoscan {

template <PublishedPayload T>
class Input;

template <PublishedPayload T>
class Output;

template <PublishedPayload T>
class PredicateInputView;

namespace detail {

template <PublishedPayload T>
inline constexpr bool kRuntimeInlinePodPayload =
    std::is_standard_layout_v<T> && std::is_trivially_copyable_v<T> && !std::same_as<T, Tensor> &&
    (kPublishedScalar<T> || HasChannelPodContract<T>);

template <PublishedPayload T>
inline constexpr bool kRuntimeCustomPodPayload =
    kRuntimeInlinePodPayload<T> && HasChannelPodContract<T>;

template <PublishedPayload T>
inline constexpr bool kRuntimeGeneratedSchemaPayload =
    !std::same_as<T, Tensor> && HasSchemaIdentity<T> && HasGeneratedSchemaNativeTraits<T> &&
    HasSchemaPackage<T> && std::is_nothrow_move_constructible_v<T>;

template <PublishedPayload T>
inline constexpr bool kRuntimeDescriptorTensorSchemaPayload = [] {
  static_assert(
      !(HasDescriptorTensorSchemaNativeTraits<T> && HasGeneratedSchemaNativeTraits<T>),
      "DescriptorTensorSchemaNativeTraits<T> and GeneratedSchemaNativeTraits<T> are mutually "
      "exclusive");
  return !std::same_as<T, Tensor> && HasSchemaIdentity<T> &&
         HasDescriptorTensorSchemaNativeTraits<T>;
}();

template <PublishedPayload T>
inline constexpr bool kRuntimeSingularDescriptorTensorSchemaPayload = [] {
  if constexpr (kRuntimeDescriptorTensorSchemaPayload<T>) {
    return descriptor_tensor_schema_tensor_count_v<T> == 1U;
  }
  return false;
}();

template <PublishedPayload T>
inline constexpr bool kRuntimeMultiDescriptorTensorSchemaPayload = [] {
  if constexpr (kRuntimeDescriptorTensorSchemaPayload<T>) {
    return descriptor_tensor_schema_tensor_count_v<T> > 1U;
  }
  return false;
}();

// Generic output loans keep a live, already-constructed value in plan-owned storage. Use one
// constraint for both public overload participation and Runtime pool construction so neither layer
// advertises an authority the other cannot realize. Assignment remains a baseline eligibility rule
// even though Runtime resets a retired slot by destruction and no-throw default construction.
template <PublishedPayload T>
inline constexpr bool kGenericOutputLoanPayload =
    (kRuntimeInlinePodPayload<T> || kRuntimeGeneratedSchemaPayload<T>) &&
    std::default_initializable<T> && std::assignable_from<T&, T> &&
    std::is_nothrow_default_constructible_v<T> && std::is_nothrow_destructible_v<T>;

}  // namespace detail

/// @brief Identify whether a logical port receives or publishes samples.
enum class PortDirection : std::uint8_t {
  kInput,   ///< Runtime selects samples for the Operator to inspect.
  kOutput,  ///< The Operator publishes samples to Runtime.
};

/// @brief Identify one logical port within a compiled execution plan.
///
/// The nonzero numeric representation is plan-local and carries no ordering, address, schema, or
/// transport meaning. Zero is invalid. Compilation assigns the final value after canonical port
/// ordering; an authoring port member does not manufacture one.
class PortId {
 public:
  /// @brief Construct an invalid port identity.
  constexpr PortId() noexcept = default;

  /// @brief Construct a plan-local port identity from its frozen representation.
  /// @param value Nonzero compiler-assigned value, or zero for an invalid value.
  explicit constexpr PortId(std::uint64_t value) noexcept : value_(value) {}

  /// @brief Return the fixed-width plan-local representation.
  /// @return Numeric representation, with zero denoting invalid.
  [[nodiscard]] constexpr std::uint64_t value() const noexcept { return value_; }

  /// @brief Return whether this identity is nonzero.
  /// @return \c true for a compiler-assigned value; otherwise \c false.
  [[nodiscard]] explicit constexpr operator bool() const noexcept { return value_ != 0; }

  /// @brief Compare complete plan-local identities.
  /// @param lhs Left identity.
  /// @param rhs Right identity.
  /// @return \c true when both numeric representations match.
  friend constexpr bool operator==(PortId lhs, PortId rhs) noexcept = default;

 private:
  std::uint64_t value_{};
};

static_assert(sizeof(InputSelection) == 2);
static_assert(std::is_standard_layout_v<InputSelection>);
static_assert(std::is_trivially_copyable_v<InputSelection>);

namespace detail {
class SyntheticOperatorBinding;
class TensorOutputPortBindingAccess;
class GraphPortOwnershipAccess;
class RuntimePortBindingAccess;
struct TensorOutputBinding;

template <PublishedPayload T>
struct PredicateInputBinding;

template <PublishedPayload T>
[[nodiscard]] expected<PredicateInputView<T>, Error> make_runtime_predicate_input_view(
    PredicateInputBinding<T> binding) noexcept;

/// @brief Provide one process-local exact payload-type discriminator shared by cold contracts.
/// @tparam T Published payload type represented by the discriminator.
///
/// The address is valid only for in-process declaration validation. It is never a schema,
/// serialization, plan, trace, or transport identity.
template <PublishedPayload T>
inline constexpr std::byte kPublishedPayloadLocalTypeToken{};

template <PublishedPayload T>
struct SelectedBatchBinding {
  const Sample<T>* const* samples{};
  const void* indirect_state{};
  const void* (*indirect_sample_at)(const void*, std::size_t) noexcept {};
  std::size_t size{};
  const std::uint64_t* active_generation{};
  std::uint64_t expected_generation{};
};

template <PublishedPayload T>
struct ScalarReadBinding {
  const Sample<T>* sample{};
  PublishedReadClaimBinding claim{};
};

template <PublishedPayload T>
struct InputBindingOps {
  InputSelection (*selection)(const void*, std::uint64_t) noexcept;
  expected<ScalarReadBinding<T>, Error> (*scalar)(const void*, std::uint64_t, bool) noexcept;
  expected<SelectedBatchBinding<T>, Error> (*batch)(const void*, std::uint64_t) noexcept;
};

template <PublishedPayload T>
struct InputBinding {
  void* state{};
  const InputBindingOps<T>* ops{};
  std::uint64_t generation{};
};

template <PublishedPayload T>
struct LoanBindingOps {
  void (*cancel)(void*) noexcept;
};

template <PublishedPayload T>
struct LoanAllocation {
  T* value{};
  void* state{};
  const LoanBindingOps<T>* ops{};
};

template <PublishedPayload T>
struct OutputBindingOps {
  expected<void, Error> (*emit_sample)(void*, Sample<T>) noexcept;
  expected<LoanAllocation<T>, Error> (*allocate)(void*) noexcept;
  expected<void, Error> (*emit_loan)(void*, void*, EmitOptions) noexcept;
};

template <PublishedPayload T>
struct OutputBinding {
  void* state{};
  const OutputBindingOps<T>* ops{};
};

/// @brief Describe one type-erased scalar result supplied by a private Runtime binding.
struct RuntimeScalarReadBinding {
  const void* sample{};
  PublishedReadClaimBinding claim{};
};

/// @brief Describe one type-erased activation-scoped batch supplied by Runtime.
struct RuntimeBatchReadBinding {
  const void* state{};
  std::size_t size{};
  const std::uint64_t* active_generation{};
  std::uint64_t expected_generation{};
  const void* (*sample_at)(const void*, std::size_t) noexcept {};
};

/// @brief Dispatch type-erased input access without exposing a transport handle.
struct RuntimeInputBindingOps {
  InputSelection (*selection)(const void*, std::uint64_t) noexcept;
  expected<RuntimeScalarReadBinding, Error> (*scalar)(const void*, std::uint64_t, bool) noexcept;
  expected<RuntimeBatchReadBinding, Error> (*batch)(const void*, std::uint64_t) noexcept;
};

/// @brief Bind one stable replay declaration to session-owned input state.
struct RuntimeInputBinding {
  void* state{};
  const RuntimeInputBindingOps* ops{};
  std::uint64_t generation{};
};

/// @brief Retain type-erased cancellation state for one generic output loan.
struct RuntimeLoanBinding {
  void* state{};
  void (*cancel)(void*) noexcept {};
};

/// @brief Return one already-constructed type-erased output value.
struct RuntimeLoanAllocation {
  void* value{};
  RuntimeLoanBinding* loan{};
};

/// @brief Dispatch publication through one private session-owned output binding.
struct RuntimeOutputBindingOps {
  expected<void, Error> (*emit_sample)(void*, const void*) noexcept;
  expected<void, Error> (*emit_descriptor_tensor)(void*, const void*, const Tensor*,
                                                  EmitOptions) noexcept;
  expected<void, Error> (*emit_descriptor_tensors)(void*, const void*, std::span<const Tensor>,
                                                   EmitOptions) noexcept;
  expected<RuntimeLoanAllocation, Error> (*allocate)(void*) noexcept;
  expected<void, Error> (*emit_loan)(void*, RuntimeLoanBinding*, EmitOptions) noexcept;
};

/// @brief Bind one stable replay declaration to session-owned publication state.
struct RuntimeOutputBinding {
  void* state{};
  const RuntimeOutputBindingOps* ops{};
};

/// @brief Classify the process-local codec for one exact published C++ payload type.
enum class RuntimePayloadCodecFamily : std::uint8_t {
  kUnsupported = 0,             ///< No baseline Runtime Channel materialization path is available.
  kInlinePod = 1,               ///< Fixed-width built-in scalar bytes.
  kGeneratedSchema = 2,         ///< Bounded generated-package serialization and materialization.
  kCustomPod = 3,               ///< Explicit fieldwise custom-POD encoding and materialization.
  kDescriptorTensorSchema = 4,  ///< Bounded schema metadata plus sidecar-retained Tensor storage.
};

/// @brief Metadata decoded verbatim from one descriptor-Tensor wire envelope.
struct DescriptorTensorWireMetadata {
  SampleMetadata metadata{};
};

/// @brief Plan-normalized metadata assigned to one public descriptor-Tensor sample.
struct DescriptorTensorSampleMetadata {
  SampleMetadata metadata{};
};

/// @brief Provide process-local operations for one exact published C++ payload type.
///
/// This table is captured only in executable factory inventory and replay declarations. Its
/// address, function pointers, alignment, and C++ object layout never enter an ExecutionPlan,
/// digest, trace, schema, or transport record. Runtime uses it only after the immutable payload
/// identity has independently matched the replay declaration.
struct RuntimePortTypeOps {
  /// Native C++ payload size in bytes.
  std::size_t payload_size{};
  /// Native C++ payload alignment in bytes.
  std::size_t payload_alignment{};
  /// Maximum canonical encoded payload size in bytes.
  std::size_t wire_capacity{};
  /// Native `Sample<T>` object size in bytes.
  std::size_t sample_size{};
  /// Native `Sample<T>` object alignment in bytes.
  std::size_t sample_alignment{};
  /// Native `PredicateInputView<T>` object size in bytes.
  std::size_t predicate_view_size{};
  /// Native `PredicateInputView<T>` object alignment in bytes.
  std::size_t predicate_view_alignment{};
  /// Process-local codec family selected for the payload.
  RuntimePayloadCodecFamily codec_family{RuntimePayloadCodecFamily::kUnsupported};
  /// Core semantic payload representation.
  PayloadKind payload_kind{PayloadKind::kInlineTyped};
  /// Whether the payload belongs to the admitted fixed-size POD family.
  bool inline_pod{};
  /// Whether the payload supports the generic finite-pool output-loan path.
  bool generic_loan_supported{};
  /// Whether one cold-constructed sample object is reused for each typed input slot.
  bool persistent_sample_storage{};
  /// Byte capacity of one generic output-loan payload slot, or zero when unsupported.
  std::size_t generic_loan_slot_size_bytes{};
  /// Byte offset of the payload descriptor in a descriptor-Tensor schema envelope.
  std::size_t descriptor_offset{};
  /// Static number of consecutive payload descriptors in the schema envelope.
  std::uint32_t descriptor_tensor_static_count{};
  /// Encode one native payload into caller-provided canonical storage, or null when unnecessary.
  expected<std::size_t, Error> (*encode_payload)(const void*, std::span<std::byte>) noexcept {};
  /// Construct one `Sample<T>` in caller-provided storage from canonical payload bytes.
  expected<void, Error> (*construct_sample)(void*, std::span<const std::byte>,
                                            SampleMetadata) noexcept {};
  /// Destroy one `Sample<T>` previously constructed by \ref construct_sample.
  void (*destroy_sample)(void*) noexcept {};
  /// Cold-construct one persistent sample slot before Runtime enters steady state.
  expected<void, Error> (*prepare_sample_storage)(void*) noexcept {};
  /// Clear one persistent sample while retaining its cold allocation capacity.
  void (*reset_sample_storage)(void*) noexcept {};
  /// Return the payload address borrowed from one constructed `Sample<T>`.
  const void* (*sample_payload)(const void*) noexcept {};
  /// Return the metadata address borrowed from one constructed `Sample<T>`.
  const SampleMetadata* (*sample_metadata)(const void*) noexcept {};
  /// Return the embedded Tensor carried by one descriptor-backed schema value.
  const Tensor* (*descriptor_tensor)(const void*) noexcept {};
  /// Return the static number of canonically indexed Tensor fields carried by a schema value.
  std::uint32_t (*descriptor_tensor_count)(const void*) noexcept {};
  /// Return one canonically indexed embedded Tensor carried by a descriptor-backed schema value.
  const Tensor* (*descriptor_tensor_at)(const void*, std::uint32_t) noexcept {};
  /// Encode one descriptor-backed schema value and normalized metadata into bounded wire storage.
  expected<std::size_t, Error> (*encode_descriptor_tensor_sample)(const void*, SampleMetadata,
                                                                  std::uint64_t endpoint_generation,
                                                                  std::span<std::byte>) noexcept {};
  /// Encode descriptor-schema metadata with a separately allocated authoritative Tensor.
  expected<std::size_t, Error> (*encode_descriptor_tensor_loan)(const void*, const Tensor&,
                                                                SampleMetadata,
                                                                std::uint64_t endpoint_generation,
                                                                std::span<std::byte>) noexcept {};
  /// Encode descriptor-schema metadata with authoritative canonically indexed Tensor fields.
  expected<std::size_t, Error> (*encode_descriptor_tensor_loans)(const void*,
                                                                 std::span<const Tensor>,
                                                                 SampleMetadata,
                                                                 std::uint64_t endpoint_generation,
                                                                 std::span<std::byte>) noexcept {};
  /// Validate one descriptor-backed schema envelope and return its public sample metadata.
  /// The descriptor itself remains mandatory for transport receives and is absent only for a
  /// compiler-authorized direct-retention handoff.
  /// `expected_generation` is the receiver's current endpoint attachment generation, so a late
  /// frame stamped by a prior sender attachment is rejected before typed exposure.
  /// A successful result authorizes the immediate assignment hook to consume only the same
  /// immutable byte span while its caller retains exclusive access.
  expected<SampleMetadata, Error> (*descriptor_tensor_sample_metadata)(
      std::span<const std::byte>, std::uint64_t descriptor_carriage_mask,
      std::uint64_t expected_generation) noexcept {};
  /// Validate one descriptor-backed schema envelope and return its embedded Tensor view.
  expected<TensorViewMetadata, Error> (*descriptor_tensor_view_metadata)(
      std::span<const std::byte>, std::uint64_t expected_generation) noexcept {};
  /// Validate one descriptor-backed schema envelope and return one indexed Tensor view.
  expected<TensorViewMetadata, Error> (*descriptor_tensor_view_metadata_at)(
      std::span<const std::byte>, std::uint32_t, std::uint64_t expected_generation) noexcept {};
  /// Populate one cold-constructed sample slot from validated schema bytes and one resolved
  /// Tensor. `wire_metadata` must match the encoded envelope; `sample_metadata` is the
  /// plan-normalized metadata assigned to the public sample.
  expected<void, Error> (*assign_descriptor_tensor_sample)(
      void*, std::span<const std::byte>, Tensor, DescriptorTensorWireMetadata wire_metadata,
      DescriptorTensorSampleMetadata sample_metadata) noexcept {};
  /// Transactionally populate one sample from validated schema bytes and indexed Tensors.
  /// `wire_metadata` must match the encoded envelope; `sample_metadata` is the plan-normalized
  /// metadata assigned to the public sample.
  expected<void, Error> (*assign_descriptor_tensors_sample)(
      void*, std::span<const std::byte>, std::span<Tensor>,
      DescriptorTensorWireMetadata wire_metadata,
      DescriptorTensorSampleMetadata sample_metadata) noexcept {};
  /// Construct one `PredicateInputView<T>` in caller-provided storage.
  expected<void, Error> (*construct_predicate_view)(void*, const void*, InputSelection,
                                                    const std::uint64_t*,
                                                    std::uint64_t) noexcept {};
  /// Destroy one predicate view previously constructed by \ref construct_predicate_view.
  void (*destroy_predicate_view)(void*) noexcept {};
  /// Default-construct one payload value in caller-provided storage when supported.
  void (*construct_value)(void*) noexcept {};
  /// Destroy one payload value previously constructed by \ref construct_value.
  void (*destroy_value)(void*) noexcept {};
  /// Bind one typed input declaration to session-owned runtime state.
  void (*bind_input)(void*, RuntimeInputBinding*) noexcept {};
  /// Remove the runtime binding from one typed input declaration.
  void (*unbind_input)(void*, RuntimeInputBinding*) noexcept {};
  /// Bind one typed output declaration to session-owned runtime state.
  void (*bind_output)(void*, RuntimeOutputBinding*) noexcept {};
  /// Remove the runtime binding from one typed output declaration.
  void (*unbind_output)(void*, RuntimeOutputBinding*) noexcept {};
  /// Publish one provider-backed Tensor through the exact output declaration.
  expected<void, Error> (*publish_tensor_loan)(void*, const void*, Tensor, EmitOptions) noexcept {};
  /// Publish one canonical provider-backed Tensor field set through the exact output declaration.
  expected<void, Error> (*publish_tensor_loans)(void*, const void*, std::span<const Tensor>,
                                                EmitOptions) noexcept {};
  /// Bind one shaped Tensor allocation capability to a Tensor-backed output declaration.
  void (*bind_tensor_output)(void*, TensorOutputBinding) noexcept {};
  /// Remove the shaped Tensor allocation capability from an output declaration.
  void (*unbind_tensor_output)(void*) noexcept {};
};

template <PublishedPayload T>
[[nodiscard]] const RuntimePortTypeOps* runtime_port_type_ops() noexcept;

/// Specialize Tensor port operations in a translation unit where PredicateInputView<Tensor> is
/// complete. Keeping this declaration here lets compiler/runtime code compare one stable ops
/// identity without forcing the circular predicate-view header through the public port header.
template <>
[[nodiscard]] HOLOSCAN_CORE_EXPORT const RuntimePortTypeOps*
runtime_port_type_ops<Tensor>() noexcept;

struct TensorOutputBindingOps {
  expected<TensorOutputLoan, Error> (*allocate)(void*, TensorLoanRequest) noexcept;
  expected<void, Error> (*emit_loan)(void*, TensorOutputLoan, const void*, EmitOptions) noexcept;
  expected<TensorSchemaLoanBundleStorage, Error> (*allocate_tensors)(
      void*, const TensorSchemaLoanRequestList&) noexcept;
  expected<void, Error> (*emit_tensors)(void*, TensorSchemaLoanBundleStorage, const void*,
                                        EmitOptions) noexcept;
};

struct TensorOutputBinding {
  void* state{};
  const TensorOutputBindingOps* ops{};
  std::optional<DLDataType> tensor_dtype;
  std::span<const std::optional<DLDataType>> tensor_field_dtypes;
};

}  // namespace detail

/// @brief Provide the non-templated authoring identity shared by typed inputs.
///
/// A port member is stable, noncopyable, and nonmovable. Its name is cold authoring metadata and
/// may allocate during construction. It does not own a queue, backend endpoint, or execution
/// session, and it exposes no testing load operation.
class InputBase {
 public:
  /// @brief Reject copying a stable input declaration.
  /// @param other Declaration that would otherwise be copied.
  InputBase(const InputBase& other) = delete;

  /// @brief Reject copy assignment of a stable input declaration.
  /// @param other Declaration that would otherwise replace this one.
  InputBase& operator=(const InputBase& other) = delete;

  /// @brief Reject moving a stable input declaration.
  /// @param other Declaration that would otherwise be moved.
  InputBase(InputBase&& other) = delete;

  /// @brief Reject move assignment of a stable input declaration.
  /// @param other Declaration that would otherwise replace this one.
  InputBase& operator=(InputBase&& other) = delete;

  /// @brief Destroy an unbound authoring input declaration.
  virtual ~InputBase() = default;

  /// @brief Return the cold declaration name.
  /// @return View valid until this port is destroyed.
  [[nodiscard]] std::string_view name() const noexcept { return name_; }

 protected:
  /// @brief Construct an unnamed input declaration.
  InputBase() = default;

  /// @brief Construct an input declaration that owns its name.
  /// @param name Declaration name copied into cold authoring storage.
  explicit InputBase(std::string_view name) : name_(name) {}

 private:
  std::string name_;
  mutable const void* graph_owner_token_{};

  friend class PortRef;
  friend class OperatorSpec;
  friend class detail::SyntheticOperatorBinding;
  friend class detail::GraphPortOwnershipAccess;
};

/// @brief Provide the non-templated authoring identity shared by typed outputs.
///
/// An output declaration is a stable member slot rather than an endpoint owner. It carries no
/// queue, scheduler, or testing retrieval API. Its cold name may allocate during construction.
class OutputBase {
 public:
  /// @brief Reject copying a stable output declaration.
  /// @param other Declaration that would otherwise be copied.
  OutputBase(const OutputBase& other) = delete;

  /// @brief Reject copy assignment of a stable output declaration.
  /// @param other Declaration that would otherwise replace this one.
  OutputBase& operator=(const OutputBase& other) = delete;

  /// @brief Reject moving a stable output declaration.
  /// @param other Declaration that would otherwise be moved.
  OutputBase(OutputBase&& other) = delete;

  /// @brief Reject move assignment of a stable output declaration.
  /// @param other Declaration that would otherwise replace this one.
  OutputBase& operator=(OutputBase&& other) = delete;

  /// @brief Destroy an unbound authoring output declaration.
  virtual ~OutputBase() = default;

  /// @brief Return the cold declaration name.
  /// @return View valid until this port is destroyed.
  [[nodiscard]] std::string_view name() const noexcept { return name_; }

 protected:
  /// @brief Construct an unnamed output declaration.
  OutputBase() = default;

  /// @brief Construct an output declaration that owns its name.
  /// @param name Declaration name copied into cold authoring storage.
  explicit OutputBase(std::string_view name) : name_(name) {}

  /// @brief Return the private shaped-allocation binding for a Tensor-backed output.
  /// @return Mutable binding used only by exact typed output specializations.
  [[nodiscard]] detail::TensorOutputBinding& tensor_output_binding() noexcept {
    return tensor_binding_;
  }

 private:
  std::string name_;
  mutable const void* graph_owner_token_{};
  detail::TensorOutputBinding tensor_binding_{};

  friend class PortRef;
  friend class OperatorSpec;
  friend class detail::SyntheticOperatorBinding;
  friend class detail::TensorOutputPortBindingAccess;
  friend class detail::GraphPortOwnershipAccess;
};

/// @brief Borrow a typed authoring port declaration for cold contract construction.
///
/// The reference does not own or bind a runtime endpoint. It is valid only while the nonmovable
/// declaration outlives it. Authoring addresses never enter execution plans, schema identities,
/// serialized records, or transport metadata.
class PortRef {
 public:
  /// @brief Borrow an input declaration.
  /// @param port Stable input member that outlives this reference.
  PortRef(const InputBase& port) noexcept  // NOLINT(runtime/explicit)
      : port_(&port), direction_(PortDirection::kInput) {}

  /// @brief Borrow an output declaration.
  /// @param port Stable output member that outlives this reference.
  PortRef(const OutputBase& port) noexcept  // NOLINT(runtime/explicit)
      : port_(&port), direction_(PortDirection::kOutput) {}

  /// @brief Return the direction of the borrowed declaration.
  /// @return Input or output direction.
  [[nodiscard]] PortDirection direction() const noexcept { return direction_; }

  /// @brief Return the borrowed declaration name.
  /// @return Name view, or an empty view for an invalid internal state.
  [[nodiscard]] std::string_view name() const noexcept {
    if (port_ == nullptr) {
      return {};
    }
    return direction_ == PortDirection::kInput ? static_cast<const InputBase*>(port_)->name()
                                               : static_cast<const OutputBase*>(port_)->name();
  }

  /// @brief Return whether this reference borrows a declaration.
  /// @return \c true when bound to an input or output member.
  [[nodiscard]] explicit operator bool() const noexcept { return port_ != nullptr; }

 private:
  const void* port_{};
  PortDirection direction_{PortDirection::kInput};
};

/// @brief Borrow the samples selected for one batch-shaped input activation.
///
/// The view is immutable, stable-multipass, random access, and deliberately non-contiguous. It
/// owns no samples or read claims and expires when the activation generation changes. Indexed or
/// iterator dereference after expiry is a contract violation that terminates deterministically.
template <PublishedPayload T>
class SelectedBatchView {
 public:
  /// @brief Name the owning sample type exposed by this view.
  using value_type = Sample<T>;

  /// @brief Name an immutable activation-scoped sample reference.
  using reference = const value_type&;

  /// @brief Name the unsigned selection-size and index type.
  using size_type = std::size_t;

  /// @brief Name the signed iterator distance type.
  using difference_type = std::ptrdiff_t;

  /// @brief Copy the same activation-scoped immutable selection view.
  /// @param other View whose borrowed binding is copied without retaining samples.
  SelectedBatchView(const SelectedBatchView& other) noexcept = default;

  /// @brief Replace this view with another borrowed selection binding.
  /// @param other View whose borrowed binding is copied without retaining samples.
  /// @return This view after replacement.
  SelectedBatchView& operator=(const SelectedBatchView& other) noexcept = default;

  /// @brief Move the borrowed selection binding from another view.
  /// @param other View whose trivially borrowed binding is copied by the move.
  SelectedBatchView(SelectedBatchView&& other) noexcept = default;

  /// @brief Replace this view by moving another borrowed selection binding.
  /// @param other View whose trivially borrowed binding is copied by the move.
  /// @return This view after replacement.
  SelectedBatchView& operator=(SelectedBatchView&& other) noexcept = default;

  /// @brief Destroy this view without retaining or releasing selected samples.
  ~SelectedBatchView() noexcept = default;

  /// @brief Iterate indirectly over immutable selected samples.
  class const_iterator {
   public:
    /// @brief Expose the C++20 random-access iterator concept tag.
    using iterator_concept = std::random_access_iterator_tag;

    /// @brief Expose the legacy random-access iterator category tag.
    using iterator_category = std::random_access_iterator_tag;

    /// @brief Name the selected sample value type.
    using value_type = Sample<T>;

    /// @brief Name the signed iterator distance type.
    using difference_type = std::ptrdiff_t;

    /// @brief Name an immutable activation-scoped sample reference.
    using reference = const value_type&;

    /// @brief Name a pointer to an immutable selected sample.
    using pointer = const value_type*;

    /// @brief Construct a singular iterator that cannot be dereferenced.
    const_iterator() noexcept = default;

    /// @brief Return the selected sample at this iterator position.
    /// @return Immutable sample reference from the bound view.
    /// @pre The iterator is nonsingular, is not the end iterator, and its activation is live.
    [[nodiscard]] reference operator*() const noexcept { return view_->at(index_); }

    /// @brief Return pointer access to the selected sample at this position.
    /// @return Pointer valid for the same lifetime as the bound activation-scoped view.
    /// @pre The iterator is nonsingular, is not the end iterator, and its activation is live.
    [[nodiscard]] pointer operator->() const noexcept { return &view_->at(index_); }

    /// @brief Return a selected sample at an offset from this iterator.
    /// @param offset Signed displacement that remains within the bound view.
    /// @return Immutable sample reference at the displaced position.
    reference operator[](difference_type offset) const noexcept { return *(*this + offset); }

    /// @brief Advance this iterator by one selected sample.
    /// @return This advanced iterator.
    const_iterator& operator++() noexcept {
      ++index_;
      return *this;
    }
    /// @brief Advance this iterator while returning its prior value.
    /// @return Iterator value before the increment.
    const_iterator operator++(int) noexcept {
      const_iterator copy = *this;
      ++(*this);
      return copy;
    }
    /// @brief Move this iterator backward by one selected sample.
    /// @return This decremented iterator.
    const_iterator& operator--() noexcept {
      --index_;
      return *this;
    }
    /// @brief Move this iterator backward while returning its prior value.
    /// @return Iterator value before the decrement.
    const_iterator operator--(int) noexcept {
      const_iterator copy = *this;
      --(*this);
      return copy;
    }
    /// @brief Advance this iterator by a signed displacement.
    /// @param offset Displacement that keeps the resulting position within the bound range.
    /// @return This displaced iterator.
    const_iterator& operator+=(difference_type offset) noexcept {
      index_ = static_cast<size_type>(static_cast<difference_type>(index_) + offset);
      return *this;
    }
    /// @brief Move this iterator backward by a signed displacement.
    /// @param offset Displacement that keeps the resulting position within the bound range.
    /// @return This displaced iterator.
    const_iterator& operator-=(difference_type offset) noexcept { return *this += -offset; }

    /// @brief Return an iterator advanced by a signed displacement.
    /// @param iterator Iterator to displace.
    /// @param offset Signed displacement within the same bound view.
    /// @return Displaced iterator.
    friend const_iterator operator+(const_iterator iterator, difference_type offset) noexcept {
      iterator += offset;
      return iterator;
    }
    /// @brief Return an iterator advanced by a signed displacement.
    /// @param offset Signed displacement within the same bound view.
    /// @param iterator Iterator to displace.
    /// @return Displaced iterator.
    friend const_iterator operator+(difference_type offset, const_iterator iterator) noexcept {
      iterator += offset;
      return iterator;
    }
    /// @brief Return an iterator moved backward by a signed displacement.
    /// @param iterator Iterator to displace.
    /// @param offset Signed displacement within the same bound view.
    /// @return Displaced iterator.
    friend const_iterator operator-(const_iterator iterator, difference_type offset) noexcept {
      iterator -= offset;
      return iterator;
    }
    /// @brief Return the signed distance between two iterators into the same view.
    /// @param lhs Left iterator.
    /// @param rhs Right iterator.
    /// @return Number of selected positions from \p rhs to \p lhs.
    friend difference_type operator-(const_iterator lhs, const_iterator rhs) noexcept {
      assert(lhs.view_ == rhs.view_);
      return static_cast<difference_type>(lhs.index_) - static_cast<difference_type>(rhs.index_);
    }
    /// @brief Compare iterator positions within the same view.
    /// @param lhs Left iterator.
    /// @param rhs Right iterator.
    /// @return \c true when both view and position representations match.
    friend bool operator==(const_iterator lhs, const_iterator rhs) noexcept = default;

    /// @brief Order two iterator positions within the same bound view.
    /// @param lhs Left iterator.
    /// @param rhs Right iterator.
    /// @return \c true when \p lhs precedes \p rhs.
    friend bool operator<(const_iterator lhs, const_iterator rhs) noexcept {
      assert(lhs.view_ == rhs.view_);
      return lhs.index_ < rhs.index_;
    }
    /// @brief Order two iterator positions within the same bound view.
    /// @param lhs Left iterator.
    /// @param rhs Right iterator.
    /// @return \c true when \p lhs follows \p rhs.
    friend bool operator>(const_iterator lhs, const_iterator rhs) noexcept { return rhs < lhs; }

    /// @brief Order two iterator positions within the same bound view.
    /// @param lhs Left iterator.
    /// @param rhs Right iterator.
    /// @return \c true when \p lhs does not follow \p rhs.
    friend bool operator<=(const_iterator lhs, const_iterator rhs) noexcept { return !(rhs < lhs); }

    /// @brief Order two iterator positions within the same bound view.
    /// @param lhs Left iterator.
    /// @param rhs Right iterator.
    /// @return \c true when \p lhs does not precede \p rhs.
    friend bool operator>=(const_iterator lhs, const_iterator rhs) noexcept { return !(lhs < rhs); }

   private:
    const_iterator(const SelectedBatchView* view, size_type index) noexcept
        : view_(view), index_(index) {}

    const SelectedBatchView* view_{};
    size_type index_{};

    friend class SelectedBatchView;
  };

  /// @brief Return the number of selected samples.
  /// @return Frozen sample count; zero is a valid completed batch.
  [[nodiscard]] size_type size() const noexcept { return binding_.size; }

  /// @brief Return whether the completed selection is empty.
  /// @return \c true when \ref size is zero.
  [[nodiscard]] bool empty() const noexcept { return size() == 0; }

  /// @brief Return one selected sample in frozen logical order.
  /// @param index Zero-based index smaller than \ref size.
  /// @return Immutable activation-scoped sample reference.
  [[nodiscard]] reference operator[](size_type index) const noexcept { return at(index); }

  /// @brief Return the first selected sample.
  /// @return Immutable first sample; the view must not be empty.
  [[nodiscard]] reference front() const noexcept { return at(0); }

  /// @brief Return the last selected sample.
  /// @return Immutable last sample; the view must not be empty.
  [[nodiscard]] reference back() const noexcept { return at(size() - 1U); }

  /// @brief Return an iterator to the first selected sample.
  /// @return Random-access indirect iterator.
  [[nodiscard]] const_iterator begin() const noexcept { return const_iterator{this, 0}; }

  /// @brief Return an iterator one past the final selected sample.
  /// @return Random-access indirect sentinel iterator.
  [[nodiscard]] const_iterator end() const noexcept { return const_iterator{this, size()}; }

 private:
  explicit SelectedBatchView(detail::SelectedBatchBinding<T> binding) noexcept
      : binding_(binding) {}

  [[nodiscard]] reference at(size_type index) const noexcept {
    if (index >= binding_.size || binding_.expected_generation == 0 ||
        binding_.active_generation == nullptr ||
        *binding_.active_generation != binding_.expected_generation) {
      std::terminate();
    }
    if (binding_.indirect_sample_at != nullptr) {
      if (binding_.indirect_state == nullptr) {
        std::terminate();
      }
      const void* sample = binding_.indirect_sample_at(binding_.indirect_state, index);
      if (sample == nullptr) {
        std::terminate();
      }
      return *static_cast<const Sample<T>*>(sample);
    }
    if (binding_.samples == nullptr || binding_.samples[index] == nullptr) {
      std::terminate();
    }
    return *binding_.samples[index];
  }

  detail::SelectedBatchBinding<T> binding_{};

  friend class Input<T>;
};

/// @brief Hold sole unpublished authority for one generic output slot.
///
/// A loan exposes an already-constructed value. Move construction transfers authority; copying
/// and both assignment forms are forbidden so a pool slot has one named owner. Destruction or
/// explicit cancellation retires unpublished authority without throwing. Constructor-style
/// `emplace` is deliberately deferred beyond baseline.
template <PublishedPayload T>
class LoanedSample {
 public:
  /// @brief Transfer unpublished authority from another loan.
  /// @param other Loan whose authority becomes moved-from.
  LoanedSample(LoanedSample&& other) noexcept
      : value_(std::exchange(other.value_, nullptr)),
        state_(std::exchange(other.state_, nullptr)),
        ops_(std::exchange(other.ops_, nullptr)) {}

  /// @brief Reject copying sole unpublished authority.
  /// @param other Loan that would otherwise be copied.
  LoanedSample(const LoanedSample& other) = delete;

  /// @brief Reject copy assignment of sole unpublished authority.
  /// @param other Loan that would otherwise replace this one.
  LoanedSample& operator=(const LoanedSample& other) = delete;

  /// @brief Reject move assignment so a named live loan never silently retires another slot.
  /// @param other Loan that would otherwise replace this one.
  LoanedSample& operator=(LoanedSample&& other) = delete;

  /// @brief Retire unpublished authority without throwing.
  ~LoanedSample() noexcept { cancel(); }

  /// @brief Return mutable access to the already-constructed payload.
  /// @return Reference valid while this loan remains live.
  [[nodiscard]] T& value() & noexcept {
    assert(value_ != nullptr);
    return *value_;
  }

  /// @brief Return pointer access to the already-constructed payload.
  /// @return Non-null pointer while this loan remains live.
  [[nodiscard]] T* operator->() noexcept {
    assert(value_ != nullptr);
    return value_;
  }

  /// @brief Return mutable access to the already-constructed payload.
  /// @return Reference valid while this loan remains live.
  [[nodiscard]] T& operator*() & noexcept { return value(); }

  /// @brief Cancel unpublished authority and begin bounded retirement.
  /// @note Calling this operation more than once is harmless. It performs no publication.
  void cancel() noexcept {
    if (state_ != nullptr && ops_ != nullptr && ops_->cancel != nullptr) {
      ops_->cancel(state_);
    }
    value_ = nullptr;
    state_ = nullptr;
    ops_ = nullptr;
  }

 private:
  explicit LoanedSample(detail::LoanAllocation<T> allocation) noexcept
      : value_(allocation.value), state_(allocation.state), ops_(allocation.ops) {}

  void disarm() noexcept {
    value_ = nullptr;
    state_ = nullptr;
    ops_ = nullptr;
  }

  T* value_{};
  void* state_{};
  const detail::LoanBindingOps<T>* ops_{};

  friend class Output<T>;
};

/// @brief Declare one typed sample input as a stable Operator member.
///
/// Runtime late-binds the declaration for an activation; the authoring object owns no queue or
/// backend endpoint. All accessors use the same deterministic error precedence: unbound, stale,
/// closed, not selected, cardinality mismatch, empty, then backend failure.
template <PublishedPayload T>
class Input : public InputBase {
 public:
  /// @brief Construct an unnamed unbound input declaration.
  Input() = default;

  /// @brief Construct an unbound input declaration with an owned cold name.
  /// @param name Declaration name copied into authoring storage.
  explicit Input(std::string_view name) : InputBase(name) {}

  /// @brief Destroy this stable input declaration after all runtime bindings have retired.
  ~Input() override = default;

  /// @brief Return the frozen selection status for the current activation.
  /// @return Status recorded by the selector, or absent/never-seen when unbound.
  /// @pre The declaration is inspected only during its Operator callback.
  [[nodiscard]] InputSelection selection() const noexcept {
    if (binding_.state == nullptr || binding_.ops == nullptr ||
        binding_.ops->selection == nullptr) {
      return {InputState::kAbsent, SelectionReason::kNeverSeen};
    }
    return binding_.ops->selection(binding_.state, binding_.generation);
  }

  /// @brief Return whether selection produced no usable value.
  /// @return \c true for absent, safe-default, or not-selected states.
  [[nodiscard]] bool is_missing() const noexcept {
    const InputState state = selection().state;
    return state == InputState::kAbsent || state == InputState::kSafeDefault ||
           state == InputState::kNotSelected;
  }

  /// @brief Return whether selection reused or rejected stale history.
  /// @return \c true for a reused value or a `kTooStale` reason.
  [[nodiscard]] bool is_stale() const noexcept {
    const InputSelection selected = selection();
    return selected.state == InputState::kReused || selected.reason == SelectionReason::kTooStale;
  }

  /// @brief Copy the selected scalar sample and its metadata.
  /// @return An owning sample, or the deterministic accessor error.
  /// @retval ErrorCode::kFailure Runtime-internal owning-sample construction failed after the
  /// frozen selection was classified; selection and queue state remain unchanged.
  /// @retval ErrorCode::kResourceExhausted The finite scalar read-claim capacity is already held by
  /// owning samples returned from earlier calls.
  /// @note Repeated calls return independent copies of the same frozen selection; they do not pop
  /// a queue or advance selection state. Each success nevertheless owns an independent finite read
  /// claim until that sample and all of its copies are destroyed.
  [[nodiscard]] expected<Sample<T>, Error> receive() {
    if (binding_.state == nullptr || binding_.ops == nullptr || binding_.ops->scalar == nullptr) {
      return make_unexpected(Error{ErrorCode::kInputUndeclared});
    }
    auto result = binding_.ops->scalar(binding_.state, binding_.generation, true);
    if (!result) {
      return make_unexpected(std::move(result).error());
    }
    const bool complete_claim = result->claim.state != nullptr && result->claim.retain != nullptr &&
                                result->claim.release != nullptr;
    if (!complete_claim) {
      // A backend that returned ownership together with a release callback must not leak that
      // reservation merely because the rest of its private binding record is malformed.
      if (result->claim.state != nullptr && result->claim.release != nullptr) {
        result->claim.release(result->claim.state);
      }
      return make_unexpected(Error{ErrorCode::kReceiveError});
    }
    detail::PublishedReadClaim claim{result->claim};
    if (result->sample == nullptr) {
      return make_unexpected(Error{ErrorCode::kReceiveError});
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
      Sample<T> sample{*result->sample};
      sample.read_claim_ = std::move(claim);
      return sample;
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#else
    Sample<T> sample{*result->sample};
    sample.read_claim_ = std::move(claim);
    return sample;
#endif
  }

  /// @brief Copy only the selected scalar payload.
  /// @return Owning payload value, or the same classified error as \ref receive.
  /// @throws Any exception raised by the admitted semantic copy in an exception-enabled build.
  /// @note The public semantic copy is ordinary caller-observable C++ behavior. Runtime-internal
  /// materialization failures are classified by the bound scalar operation before this copy.
  /// This payload-only accessor acquires no finite sample read claim.
  [[nodiscard]] expected<T, Error> receive_data() {
    if (binding_.state == nullptr || binding_.ops == nullptr || binding_.ops->scalar == nullptr) {
      return make_unexpected(Error{ErrorCode::kInputUndeclared});
    }
    auto result = binding_.ops->scalar(binding_.state, binding_.generation, false);
    if (!result) {
      return make_unexpected(std::move(result).error());
    }
    const bool empty_claim = result->claim.state == nullptr && result->claim.retain == nullptr &&
                             result->claim.release == nullptr;
    if (!empty_claim) {
      const bool complete_claim = result->claim.state != nullptr &&
                                  result->claim.retain != nullptr &&
                                  result->claim.release != nullptr;
      if (complete_claim) {
        // Payload-only access must not consume finite read capacity. Adopt an unexpected complete
        // claim solely so this error path deterministically releases the backend reservation.
        const detail::PublishedReadClaim cleanup{result->claim};
      } else if (result->claim.state != nullptr && result->claim.release != nullptr) {
        result->claim.release(result->claim.state);
      }
      return make_unexpected(Error{ErrorCode::kReceiveError});
    }
    if (result->sample == nullptr) {
      return make_unexpected(Error{ErrorCode::kReceiveError});
    }
    return result->sample->data;
  }

  /// @brief Borrow the completed batch-shaped selection.
  /// @return Immutable activation-scoped view, including a successful empty view, or a classified
  /// accessor error.
  [[nodiscard]] expected<SelectedBatchView<T>, Error> selected_batch() const noexcept {
    if (binding_.state == nullptr || binding_.ops == nullptr || binding_.ops->batch == nullptr) {
      return make_unexpected(Error{ErrorCode::kInputUndeclared});
    }
    auto result = binding_.ops->batch(binding_.state, binding_.generation);
    if (!result) {
      return make_unexpected(std::move(result).error());
    }
    const bool direct = result->samples != nullptr;
    const bool indirect =
        result->indirect_state != nullptr && result->indirect_sample_at != nullptr;
    if (result->size != 0 && direct == indirect) {
      return make_unexpected(Error{ErrorCode::kReceiveError});
    }
    if (result->expected_generation == 0 || result->active_generation == nullptr ||
        *result->active_generation != result->expected_generation) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    return SelectedBatchView<T>{*result};
  }

  /// @brief Invoke a fallible callable for each selected batch sample.
  /// @tparam Fn Callable compatible with `expected<void, Error>(const Sample<T>&)`.
  /// @param fn Callable invoked in frozen selection order.
  /// @return Success after every call, or the first accessor/callable error.
  /// @note The callable is bound once as an lvalue before iteration; an rvalue callable is never
  /// repeatedly forwarded or moved between selected samples.
  template <typename Fn>
    requires requires(std::remove_reference_t<Fn>& callable, const Sample<T>& sample) {
      { callable(sample) } -> std::same_as<expected<void, Error>>;
    }
  [[nodiscard]] expected<void, Error> for_each_selected(Fn&& fn) const {
    auto batch = selected_batch();
    if (!batch) {
      return make_unexpected(std::move(batch).error());
    }
    auto&& callable = std::forward<Fn>(fn);
    for (const Sample<T>& sample : *batch) {
      auto result = callable(sample);
      if (!result) {
        return make_unexpected(std::move(result).error());
      }
    }
    return {};
  }

 private:
  detail::InputBinding<T> binding_{};

  friend class detail::SyntheticOperatorBinding;
  friend class detail::RuntimePortBindingAccess;
};

/// @brief Declare one typed sample output as a stable Operator member.
///
/// Runtime late-binds publication authority for an activation. The declaration owns no queue,
/// scheduler, or backend endpoint. Ordinary values and forwarded samples are passed by value;
/// direct generic allocation uses a distinct move-only loan.
template <PublishedPayload T>
class Output : public OutputBase {
 public:
  /// @brief Construct an unnamed unbound output declaration.
  Output() = default;

  /// @brief Construct an unbound output declaration with an owned cold name.
  /// @param name Declaration name copied into authoring storage.
  explicit Output(std::string_view name) : OutputBase(name) {}

  /// @brief Destroy this stable output declaration after all runtime bindings have retired.
  ~Output() override = default;

  /// @brief Publish one owning logical value with optional semantic metadata.
  /// @param value Value moved into Runtime ownership on every outcome. A retryable publication
  /// error preserves logical publication authority but does not restore this by-value argument;
  /// retry with a newly supplied equivalent value.
  /// @param options Optional capture time, frame identity, trace, and semantic flags.
  /// @return Success after accepted publication, or a typed error.
  /// @retval ErrorCode::kFailure An internal payload move or sample construction threw before
  /// Runtime accepted publication. Construction of the by-value argument itself remains the
  /// caller's expression and can throw before this function is entered.
  [[nodiscard]] expected<void, Error> emit(T value, EmitOptions options = {}) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      SampleMetadata metadata{};
      if (options.capture_time.has_value()) {
        metadata.capture_timestamp_ns = options.capture_time->timestamp_ns;
        metadata.source_clock_id = options.capture_time->clock_id;
      }
      if (options.frame_id.has_value()) {
        metadata.frame_id = *options.frame_id;
      }
      if (options.trace.has_value()) {
        metadata.trace_id = options.trace->trace_id;
      }
      metadata.flags = options.flags;
      return emit(Sample<T>{std::move(value), metadata});
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  /// @brief Forward one owning sample while preserving semantic metadata.
  /// @param sample Sample copied or moved into the by-value argument before entry, then moved into
  /// Runtime ownership on every outcome. A retryable publication error preserves logical
  /// publication authority but does not restore this argument; retry with a newly supplied
  /// equivalent sample.
  /// @return Success after accepted publication, or a typed error.
  /// @retval ErrorCode::kNotReady This output has no active Runtime binding.
  /// @retval ErrorCode::kFailure Moving the entered sample into the bound Runtime operation threw
  /// before that operation began.
  /// @note Runtime replaces framework-owned publication sequence/time/size/classification fields;
  /// capture, source clock, trace, frame, and semantic flags remain forwarding evidence.
  /// @note This overload deliberately has no \ref EmitOptions parameter because \p sample already
  /// supplies the semantic metadata to forward.
  [[nodiscard]] expected<void, Error> emit(Sample<T> sample) {
    if (binding_.state == nullptr || binding_.ops == nullptr ||
        binding_.ops->emit_sample == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      return binding_.ops->emit_sample(binding_.state, std::move(sample));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  /// @brief Acquire one already-constructed value from a finite output pool.
  /// @return Sole unpublished authority, or an immediate bounded allocation/state error.
  /// @retval ErrorCode::kNotReady This output has no active Runtime binding.
  /// @retval ErrorCode::kNotSupported The active execution plan provides no generic-loan pool for
  /// this otherwise eligible payload type.
  /// @retval ErrorCode::kStaleEndpoint The output is not active in the current activation.
  /// @retval ErrorCode::kResourceExhausted No plan-owned slot is available.
  /// @note This operation never blocks, grows a pool, or falls back to heap allocation.
  /// @note This overload participates only for supported inline POD or generated-schema payloads
  /// that are default-initializable, assignable from \c T, and no-throw default-constructible and
  /// destructible. Other admitted payloads remain available through the ordinary `emit(T)` path.
  [[nodiscard]] expected<LoanedSample<T>, Error> allocate() noexcept
    requires detail::kGenericOutputLoanPayload<T>
  {  // NOLINT(whitespace/braces)
    if (binding_.state == nullptr || binding_.ops == nullptr || binding_.ops->allocate == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    auto allocation = binding_.ops->allocate(binding_.state);
    if (!allocation) {
      return make_unexpected(std::move(allocation).error());
    }
    return LoanedSample<T>{*allocation};
  }

  /// @brief Submit one generic output loan for publication.
  /// @param loan Sole authority bound to this output and activation.
  /// @param options Optional semantic metadata.
  /// @return Publication result. A proven pre-submit rejection leaves \p loan live; acceptance or
  /// a terminal partial or indeterminate publication outcome consumes it.
  [[nodiscard]] expected<void, Error> emit(LoanedSample<T>&& loan,
                                           EmitOptions options = {}) noexcept {
    if (loan.state_ == nullptr || binding_.state == nullptr || binding_.ops == nullptr ||
        binding_.ops->emit_loan == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    auto result = binding_.ops->emit_loan(binding_.state, loan.state_, options);
    if (result || publication_error_consumes_authority(result.error().code)) {
      loan.disarm();
    }
    return result;
  }

  /// @brief Request one shaped allocation for a descriptor-backed Tensor schema output.
  /// @param request Logical shape, element type, and optional byte strides.
  /// @return Sole provider-backed Tensor loan, or a typed bounds/capacity/state error.
  /// @note The overload participates only for descriptor-backed Tensor schema payloads. The
  /// returned Tensor remains separate from schema metadata until \ref emit_tensor.
  [[nodiscard]] expected<TensorOutputLoan, Error> allocate_tensor(TensorLoanRequest request)
    requires detail::kRuntimeSingularDescriptorTensorSchemaPayload<T>
  {  // NOLINT(whitespace/braces)
    auto& tensor_binding = tensor_output_binding();
    if (tensor_binding.state == nullptr || tensor_binding.ops == nullptr ||
        tensor_binding.ops->allocate == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    return tensor_binding.ops->allocate(tensor_binding.state, request);
  }

  /// @brief Request one shaped allocation using the output spec's exact dtype.
  /// @param shape Logical extents copied before return.
  /// @param byte_strides Optional explicit byte strides copied before return.
  /// @return Sole provider-backed Tensor loan, or a typed contract/capacity/state error.
  /// @note The overload participates only for singular descriptor-backed Tensor schemas. The
  /// output spec must declare one exact dtype.
  [[nodiscard]] expected<TensorOutputLoan, Error> allocate_tensor(
      TensorShapeView shape, TensorByteStridesView byte_strides = std::nullopt)
    requires detail::kRuntimeSingularDescriptorTensorSchemaPayload<T>
  {  // NOLINT(whitespace/braces)
    auto& tensor_binding = tensor_output_binding();
    if (tensor_binding.state == nullptr || tensor_binding.ops == nullptr ||
        tensor_binding.ops->allocate == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    if (!tensor_binding.tensor_dtype) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "TENSOR_OUTPUT_DTYPE_UNRESOLVED"});
    }
    return tensor_binding.ops->allocate(
        tensor_binding.state,
        TensorLoanRequest{
            .shape = shape, .dtype = *tensor_binding.tensor_dtype, .byte_strides = byte_strides});
  }

  /// @brief Publish provider-backed Tensor storage with descriptor-schema metadata.
  /// @param loan Sole shaped Tensor authority; consumed for success and every error.
  /// @param descriptor Schema value whose non-Tensor fields are encoded synchronously. Its
  /// embedded Tensor member is ignored; the provider-owned Tensor from \p loan is authoritative.
  /// @param options Optional semantic sample metadata.
  /// @return Publication result.
  /// @note The overload participates only for descriptor-backed Tensor schema payloads. Neither
  /// the descriptor object nor its strings may be mutated concurrently with this call.
  [[nodiscard]] expected<void, Error> emit_tensor(TensorOutputLoan loan, const T& descriptor,
                                                  EmitOptions options = {})
    requires detail::kRuntimeSingularDescriptorTensorSchemaPayload<T>
  {  // NOLINT(whitespace/braces)
    auto& tensor_binding = tensor_output_binding();
    if (tensor_binding.state == nullptr || tensor_binding.ops == nullptr ||
        tensor_binding.ops->emit_loan == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    return tensor_binding.ops->emit_loan(
        tensor_binding.state, std::move(loan), &descriptor, options);
  }

  /// @brief Transactionally request provider-backed storage for every Tensor schema field.
  /// @param requests Exactly one generated field selector and shaped request per schema field.
  /// @return One sole multi-field publication bundle, or a typed validation, capacity, or state
  /// error. No partial bundle escapes on failure.
  /// @note The overload participates only for descriptor-backed Tensor schema payloads. Requests
  /// may be supplied in any order; Runtime canonicalizes them by generated Tensor index.
  [[nodiscard]] expected<TensorSchemaOutputLoan<T>, Error> allocate_tensors(
      std::span<const TensorFieldLoanRequest<T>> requests)
    requires detail::kRuntimeMultiDescriptorTensorSchemaPayload<T>
  {  // NOLINT(whitespace/braces)
    auto& tensor_binding = tensor_output_binding();
    if (tensor_binding.state == nullptr || tensor_binding.ops == nullptr ||
        tensor_binding.ops->allocate_tensors == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    if (requests.empty() || requests.size() > kMaximumTensorSchemaLoanFields) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    detail::TensorSchemaLoanRequestList erased;
    erased.field_count = static_cast<std::uint32_t>(requests.size());
    for (std::size_t index = 0U; index < requests.size(); ++index) {
      erased.fields[index] = detail::TensorSchemaLoanRequest{
          .field_id = requests[index].field.field_id,
          .tensor_index = requests[index].field.tensor_index,
          .tensor = requests[index].tensor,
      };
    }
    auto allocated = tensor_binding.ops->allocate_tensors(tensor_binding.state, erased);
    if (!allocated) {
      return make_unexpected(std::move(allocated).error());
    }
    return detail::TensorSchemaOutputLoanAccess::make<T>(std::move(*allocated));
  }

  /// @brief Transactionally request every Tensor field using its admitted exact dtype.
  /// @param requests Exactly one generated selector, shape, and optional strides per schema field.
  /// @return One sole multi-field publication bundle, or a typed contract, capacity, or state
  /// error. No partial bundle escapes on failure.
  /// @note Each selected output field must declare one exact dtype. Request views are copied into
  /// bounded Runtime storage before this call returns.
  [[nodiscard]] expected<TensorSchemaOutputLoan<T>, Error> allocate_tensors(
      std::span<const TensorFieldShapeRequest<T>> requests)
    requires detail::kRuntimeMultiDescriptorTensorSchemaPayload<T>
  {  // NOLINT(whitespace/braces)
    auto& tensor_binding = tensor_output_binding();
    if (tensor_binding.state == nullptr || tensor_binding.ops == nullptr ||
        tensor_binding.ops->allocate_tensors == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    if (requests.empty() || requests.size() > kMaximumTensorSchemaLoanFields) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    detail::TensorSchemaLoanRequestList erased;
    erased.field_count = static_cast<std::uint32_t>(requests.size());
    for (std::size_t index = 0U; index < requests.size(); ++index) {
      const auto tensor_index = requests[index].field.tensor_index;
      if (tensor_index >= tensor_binding.tensor_field_dtypes.size() ||
          !tensor_binding.tensor_field_dtypes[tensor_index]) {
        return make_unexpected(Error{
            ErrorCode::kInvalidArgument,
            "TENSOR_SCHEMA_BUNDLE_FIELD_DTYPE_UNRESOLVED (field_id=" +
                std::to_string(requests[index].field.field_id) +
                ", tensor_index=" + std::to_string(tensor_index) + ")",
        });
      }
      erased.fields[index] = detail::TensorSchemaLoanRequest{
          .field_id = requests[index].field.field_id,
          .tensor_index = tensor_index,
          .tensor =
              TensorLoanRequest{
                  .shape = requests[index].shape,
                  .dtype = *tensor_binding.tensor_field_dtypes[tensor_index],
                  .byte_strides = requests[index].byte_strides,
              },
      };
    }
    auto allocated = tensor_binding.ops->allocate_tensors(tensor_binding.state, erased);
    if (!allocated) {
      return make_unexpected(std::move(allocated).error());
    }
    return detail::TensorSchemaOutputLoanAccess::make<T>(std::move(*allocated));
  }

  /// @brief Publish one provider-backed Tensor field bundle with descriptor-schema metadata.
  /// @param loan Sole multi-field authority; consumed for success and every error.
  /// @param descriptor Schema value whose non-Tensor fields are encoded synchronously. Embedded
  /// Tensor members are ignored; the bundle fields are authoritative.
  /// @param options Optional semantic sample metadata.
  /// @return Publication result.
  [[nodiscard]] expected<void, Error> emit_tensors(TensorSchemaOutputLoan<T> loan,
                                                   const T& descriptor, EmitOptions options = {})
    requires detail::kRuntimeMultiDescriptorTensorSchemaPayload<T>
  {  // NOLINT(whitespace/braces)
    auto& tensor_binding = tensor_output_binding();
    if (tensor_binding.state == nullptr || tensor_binding.ops == nullptr ||
        tensor_binding.ops->emit_tensors == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    return tensor_binding.ops->emit_tensors(
        tensor_binding.state,
        detail::TensorSchemaOutputLoanAccess::release(std::move(loan)),
        &descriptor,
        options);
  }

 private:
  detail::OutputBinding<T> binding_{};

  friend class detail::SyntheticOperatorBinding;
  friend class detail::RuntimePortBindingAccess;
};

/// @brief Specialize output publication for shaped Tensor allocation.
///
/// Tensor output deliberately omits generic `allocate()`: direct allocation requires shape, type,
/// and stride metadata and returns one `TensorOutputLoan`. Existing owning Tensor values and
/// forwarded Tensor samples remain separate pass-through paths.
template <>
class Output<Tensor> : public OutputBase {
 public:
  /// @brief Construct an unnamed unbound tensor output declaration.
  Output() = default;

  /// @brief Construct an unbound tensor output declaration with an owned cold name.
  /// @param name Declaration name copied into authoring storage.
  explicit Output(std::string_view name) : OutputBase(name) {}

  /// @brief Destroy this stable Tensor output after all runtime bindings have retired.
  ~Output() override = default;

  /// @brief Request one shaped allocation from the compiled tensor pool.
  /// @param request Logical shape, element type, and optional byte strides.
  /// @return Sole tensor loan, or a typed bounds/capacity/state error.
  [[nodiscard]] expected<TensorOutputLoan, Error> allocate_tensor(TensorLoanRequest request) {
    auto& tensor_binding = tensor_output_binding();
    if (tensor_binding.state == nullptr || tensor_binding.ops == nullptr ||
        tensor_binding.ops->allocate == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    return tensor_binding.ops->allocate(tensor_binding.state, request);
  }

  /// @brief Request one shaped allocation using the output spec's exact dtype.
  /// @param shape Logical extents copied before return.
  /// @param byte_strides Optional explicit byte strides copied before return.
  /// @return Sole Tensor loan, or a typed contract/capacity/state error.
  /// @note The output spec must declare one exact dtype. Use the full request overload when
  /// dtype is intentionally dynamic.
  [[nodiscard]] expected<TensorOutputLoan, Error> allocate_tensor(
      TensorShapeView shape, TensorByteStridesView byte_strides = std::nullopt) {
    auto& tensor_binding = tensor_output_binding();
    if (tensor_binding.state == nullptr || tensor_binding.ops == nullptr ||
        tensor_binding.ops->allocate == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    if (!tensor_binding.tensor_dtype) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "TENSOR_OUTPUT_DTYPE_UNRESOLVED"});
    }
    return tensor_binding.ops->allocate(
        tensor_binding.state,
        TensorLoanRequest{
            .shape = shape, .dtype = *tensor_binding.tensor_dtype, .byte_strides = byte_strides});
  }

  /// @brief Publish one terminal tensor loan attempt.
  /// @param loan Sole shaped tensor authority; consumed for success and every error.
  /// @param options Optional semantic metadata.
  /// @return Publication result.
  [[nodiscard]] expected<void, Error> emit(TensorOutputLoan loan, EmitOptions options = {}) {
    auto& tensor_binding = tensor_output_binding();
    if (tensor_binding.state == nullptr || tensor_binding.ops == nullptr ||
        tensor_binding.ops->emit_loan == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    return tensor_binding.ops->emit_loan(tensor_binding.state, std::move(loan), nullptr, options);
  }

  /// @brief Publish an existing owning Tensor value.
  /// @param value Tensor moved into Runtime ownership on every outcome. A retryable publication
  /// error preserves logical publication authority but does not restore this by-value argument;
  /// retry with a newly supplied equivalent Tensor.
  /// @param options Optional semantic metadata.
  /// @return Publication result.
  [[nodiscard]] expected<void, Error> emit(Tensor value, EmitOptions options = {}) {
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      SampleMetadata metadata{};
      if (options.capture_time.has_value()) {
        metadata.capture_timestamp_ns = options.capture_time->timestamp_ns;
        metadata.source_clock_id = options.capture_time->clock_id;
      }
      if (options.frame_id.has_value()) {
        metadata.frame_id = *options.frame_id;
      }
      if (options.trace.has_value()) {
        metadata.trace_id = options.trace->trace_id;
      }
      metadata.flags = options.flags;
      return emit(Sample<Tensor>{std::move(value), metadata});
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

  /// @brief Forward one owning Tensor sample while preserving semantic metadata.
  /// @param sample Sample copied or moved into the by-value argument before entry, then moved into
  /// Runtime ownership on every outcome. A retryable publication error preserves logical
  /// publication authority but does not restore this argument; retry with a newly supplied
  /// equivalent sample.
  /// @return Success after accepted publication, or a typed error.
  /// @retval ErrorCode::kNotReady This output has no active Runtime binding.
  /// @retval ErrorCode::kFailure Moving the entered sample into the bound Runtime operation threw
  /// before that operation began.
  /// @note Runtime replaces framework-owned publication sequence/time/size/classification fields;
  /// capture, source clock, trace, frame, and semantic flags remain forwarding evidence.
  /// @note This overload deliberately has no \ref EmitOptions parameter because \p sample already
  /// supplies the semantic metadata to forward. Runtime admission still determines whether direct
  /// Tensor pass-through is available for the compiled output.
  [[nodiscard]] expected<void, Error> emit(Sample<Tensor> sample) {
    if (binding_.state == nullptr || binding_.ops == nullptr ||
        binding_.ops->emit_sample == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    try {
#endif
      return binding_.ops->emit_sample(binding_.state, std::move(sample));
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
    } catch (...) {
      return make_unexpected(Error{ErrorCode::kFailure});
    }
#endif
  }

 private:
  detail::OutputBinding<Tensor> binding_{};

  friend class detail::SyntheticOperatorBinding;
  friend class detail::RuntimePortBindingAccess;
};

namespace detail {

/// @brief Bind shaped allocation authority without exposing its Runtime owner.
class TensorOutputPortBindingAccess {
 public:
  static void bind(OutputBase& output, TensorOutputBinding binding) noexcept {
    output.tensor_binding_ = binding;
  }

  static void unbind(OutputBase& output) noexcept { output.tensor_binding_ = {}; }
};

/// @brief Adapt private type-erased Runtime bindings to the exact public port specialization.
class RuntimePortBindingAccess {
 public:
  template <PublishedPayload T>
  static void bind_input(Input<T>& input, RuntimeInputBinding* binding) noexcept {
    input.binding_ = InputBinding<T>{
        .state = binding,
        .ops = &input_ops<T>(),
        .generation = binding != nullptr ? binding->generation : 0U,
    };
  }

  template <PublishedPayload T>
  static void unbind_input(Input<T>& input, RuntimeInputBinding* expected_binding) noexcept {
    if (input.binding_.state == expected_binding && input.binding_.ops == &input_ops<T>()) {
      input.binding_ = {};
    }
  }

  template <PublishedPayload T>
  static void bind_output(Output<T>& output, RuntimeOutputBinding* binding) noexcept {
    output.binding_ = OutputBinding<T>{.state = binding, .ops = &output_ops<T>()};
  }

  template <PublishedPayload T>
  static void unbind_output(Output<T>& output, RuntimeOutputBinding* expected_binding) noexcept {
    if (output.binding_.state == expected_binding && output.binding_.ops == &output_ops<T>()) {
      output.binding_ = {};
    }
  }

  template <PublishedPayload T>
  [[nodiscard]] static expected<void, Error> emit_descriptor_tensor(Output<T>& output,
                                                                    const void* descriptor,
                                                                    const Tensor& tensor,
                                                                    EmitOptions options) noexcept {
    auto* binding = static_cast<RuntimeOutputBinding*>(output.binding_.state);
    if (descriptor == nullptr || binding == nullptr || binding->state == nullptr ||
        binding->ops == nullptr || binding->ops->emit_descriptor_tensor == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    return binding->ops->emit_descriptor_tensor(binding->state, descriptor, &tensor, options);
  }

  template <PublishedPayload T>
  [[nodiscard]] static expected<void, Error> emit_descriptor_tensors(
      Output<T>& output, const void* descriptor, const std::span<const Tensor> tensors,
      EmitOptions options) noexcept {
    if (tensors.empty()) {
      return make_unexpected(Error{ErrorCode::kInvalidArgument});
    }
    auto* binding = static_cast<RuntimeOutputBinding*>(output.binding_.state);
    if (descriptor == nullptr || binding == nullptr || binding->state == nullptr ||
        binding->ops == nullptr || binding->ops->emit_descriptor_tensors == nullptr) {
      return make_unexpected(Error{ErrorCode::kNotReady});
    }
    return binding->ops->emit_descriptor_tensors(binding->state, descriptor, tensors, options);
  }

 private:
  template <PublishedPayload T>
  [[nodiscard]] static const InputBindingOps<T>& input_ops() noexcept {
    static const InputBindingOps<T> ops{
        .selection =
            [](const void* state, std::uint64_t generation) noexcept {
              const auto& binding = *static_cast<const RuntimeInputBinding*>(state);
              if (binding.ops == nullptr || binding.ops->selection == nullptr) {
                return InputSelection{InputState::kAbsent, SelectionReason::kNeverSeen};
              }
              return binding.ops->selection(binding.state, generation);
            },
        .scalar = [](const void* state,
                     std::uint64_t generation,
                     bool claim) noexcept -> expected<ScalarReadBinding<T>, Error> {
          const auto& binding = *static_cast<const RuntimeInputBinding*>(state);
          if (binding.ops == nullptr || binding.ops->scalar == nullptr) {
            return make_unexpected(Error{ErrorCode::kReceiveError});
          }
          auto selected = binding.ops->scalar(binding.state, generation, claim);
          if (!selected) {
            return make_unexpected(std::move(selected).error());
          }
          return ScalarReadBinding<T>{
              .sample = static_cast<const Sample<T>*>(selected->sample),
              .claim = selected->claim,
          };
        },
        .batch = [](const void* state,
                    std::uint64_t generation) noexcept -> expected<SelectedBatchBinding<T>, Error> {
          const auto& binding = *static_cast<const RuntimeInputBinding*>(state);
          if (binding.ops == nullptr || binding.ops->batch == nullptr) {
            return make_unexpected(Error{ErrorCode::kReceiveError});
          }
          auto selected = binding.ops->batch(binding.state, generation);
          if (!selected) {
            return make_unexpected(std::move(selected).error());
          }
          return SelectedBatchBinding<T>{
              .indirect_state = selected->state,
              .indirect_sample_at = selected->sample_at,
              .size = selected->size,
              .active_generation = selected->active_generation,
              .expected_generation = selected->expected_generation,
          };
        },
    };
    return ops;
  }

  template <PublishedPayload T>
  [[nodiscard]] static const LoanBindingOps<T>& loan_ops() noexcept {
    static const LoanBindingOps<T> ops{
        .cancel =
            [](void* state) noexcept {
              auto& loan = *static_cast<RuntimeLoanBinding*>(state);
              if (loan.cancel != nullptr) {
                loan.cancel(loan.state);
              }
            },
    };
    return ops;
  }

  template <PublishedPayload T>
  [[nodiscard]] static const OutputBindingOps<T>& output_ops() noexcept {
    static const OutputBindingOps<T> ops{
        .emit_sample =
            [](void* state, Sample<T> sample) noexcept {
              auto& binding = *static_cast<RuntimeOutputBinding*>(state);
              if (binding.ops == nullptr || binding.ops->emit_sample == nullptr) {
                return expected<void, Error>{make_unexpected(Error{ErrorCode::kFailure})};
              }
              return binding.ops->emit_sample(binding.state, &sample);
            },
        .allocate = [](void* state) noexcept -> expected<LoanAllocation<T>, Error> {
          auto& binding = *static_cast<RuntimeOutputBinding*>(state);
          if (binding.ops == nullptr || binding.ops->allocate == nullptr) {
            return make_unexpected(Error{ErrorCode::kFailure});
          }
          auto allocation = binding.ops->allocate(binding.state);
          if (!allocation) {
            return make_unexpected(std::move(allocation).error());
          }
          if (allocation->value == nullptr || allocation->loan == nullptr ||
              allocation->loan->state == nullptr || allocation->loan->cancel == nullptr) {
            if (allocation->loan != nullptr && allocation->loan->state != nullptr &&
                allocation->loan->cancel != nullptr) {
              allocation->loan->cancel(allocation->loan->state);
            }
            return make_unexpected(Error{ErrorCode::kFailure});
          }
          return LoanAllocation<T>{
              .value = static_cast<T*>(allocation->value),
              .state = allocation->loan,
              .ops = &RuntimePortBindingAccess::loan_ops<T>(),
          };
        },
        .emit_loan =
            [](void* state, void* loan_state, EmitOptions options) noexcept {
              auto& binding = *static_cast<RuntimeOutputBinding*>(state);
              if (binding.ops == nullptr || binding.ops->emit_loan == nullptr ||
                  loan_state == nullptr) {
                return expected<void, Error>{make_unexpected(Error{ErrorCode::kFailure})};
              }
              return binding.ops->emit_loan(
                  binding.state, static_cast<RuntimeLoanBinding*>(loan_state), options);
            },
    };
    return ops;
  }
};

template <PublishedPayload T>
[[nodiscard]] constexpr std::uint64_t serialized_payload_capacity_bytes() noexcept {
  if constexpr (kRuntimeGeneratedSchemaPayload<T>) {
    return static_cast<std::uint64_t>(generated_schema_max_serialized_size_v<T>);
  }
  return 0U;
}

template <PublishedPayload T>
[[nodiscard]] std::uint64_t external_topic_payload_capacity_bytes() noexcept {
  if constexpr (kRuntimeDescriptorTensorSchemaPayload<T>) {
    return static_cast<std::uint64_t>(runtime_port_type_ops<T>()->wire_capacity);
  }
  return serialized_payload_capacity_bytes<T>();
}

template <PublishedPayload T>
// This compile-time factory deliberately keeps the complete erased operation table together so
// every payload-family branch can be audited against the same lifetime and capability contract.
// Splitting the fixed table merely to satisfy a path-count metric would scatter those invariants.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
[[nodiscard]] const RuntimePortTypeOps* runtime_port_type_ops() noexcept {
  static_assert(
      !kRuntimeDescriptorTensorSchemaPayload<T>,
      "a descriptor-backed Tensor schema must provide a runtime_port_type_ops<T>() specialization");
  // NOLINTNEXTLINE(readability-function-cognitive-complexity)
  static const RuntimePortTypeOps ops = [] {
    constexpr std::size_t kWireCapacity = [] {
      if constexpr (kRuntimeGeneratedSchemaPayload<T>) {
        return generated_schema_max_serialized_size_v<T>;
      } else if constexpr (kRuntimeInlinePodPayload<T>) {
        return sizeof(T);
      }
      return std::size_t{0U};
    }();
    RuntimePortTypeOps result{
        .payload_size = sizeof(T),
        .payload_alignment = alignof(T),
        .wire_capacity = kWireCapacity,
        .sample_size = sizeof(Sample<T>),
        .sample_alignment = alignof(Sample<T>),
        .predicate_view_size = sizeof(PredicateInputView<T>),
        .predicate_view_alignment = alignof(PredicateInputView<T>),
        .codec_family =
            kRuntimeCustomPodPayload<T>
                ? RuntimePayloadCodecFamily::kCustomPod
                : (kRuntimeInlinePodPayload<T> ? RuntimePayloadCodecFamily::kInlinePod
                                               : (kRuntimeGeneratedSchemaPayload<T>
                                                      ? RuntimePayloadCodecFamily::kGeneratedSchema
                                                      : RuntimePayloadCodecFamily::kUnsupported)),
        .payload_kind = kRuntimeGeneratedSchemaPayload<T> ? PayloadKind::kSerializedBytes
                                                          : PayloadKind::kInlineTyped,
        .inline_pod = kRuntimeInlinePodPayload<T>,
        .generic_loan_supported = kGenericOutputLoanPayload<T>,
        .generic_loan_slot_size_bytes = kGenericOutputLoanPayload<T> ? sizeof(T) : std::size_t{0U},
    };
    if constexpr (kRuntimeInlinePodPayload<T> || kRuntimeGeneratedSchemaPayload<T>) {
      if constexpr (kRuntimeGeneratedSchemaPayload<T>) {
        result.encode_payload = [](const void* payload, std::span<std::byte> destination) noexcept
            -> expected<std::size_t, Error> {
          if (payload == nullptr || destination.size() < kWireCapacity) {
            return make_unexpected(Error{ErrorCode::kResourceExhausted});
          }
          const auto& package = schema_package<T>();
          if (package.max_serialized_size() != kWireCapacity) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          return package.encode(*static_cast<const T*>(payload), destination);
        };
      }
      result.construct_sample =
          [](void* storage, std::span<const std::byte> payload, SampleMetadata metadata) noexcept
          -> expected<void, Error> {
        if (storage == nullptr || payload.empty() || payload.size() > kWireCapacity ||
            reinterpret_cast<std::uintptr_t>(storage) % alignof(Sample<T>) != 0U) {
          return make_unexpected(Error{ErrorCode::kReceiveError});
        }
        if constexpr (kRuntimeGeneratedSchemaPayload<T>) {
          const auto& package = schema_package<T>();
          if (package.max_serialized_size() != kWireCapacity) {
            return make_unexpected(Error{ErrorCode::kProtocolError});
          }
          auto value = package.materialize(payload);
          if (!value) {
            return make_unexpected(std::move(value).error());
          }
          std::construct_at(static_cast<Sample<T>*>(storage), std::move(*value), metadata);
        } else {
          if (payload.size() != sizeof(T)) {
            return make_unexpected(Error{ErrorCode::kReceiveError});
          }
          std::array<std::byte, sizeof(T)> native{};
          std::memcpy(native.data(), payload.data(), native.size());
          T value = std::bit_cast<T>(native);
          std::construct_at(static_cast<Sample<T>*>(storage), std::move(value), metadata);
        }
        return {};
      };
      result.destroy_sample = [](void* storage) noexcept {
        std::destroy_at(static_cast<Sample<T>*>(storage));
      };
      result.sample_payload = [](const void* sample) noexcept -> const void* {
        return &static_cast<const Sample<T>*>(sample)->data;
      };
      result.sample_metadata = [](const void* sample) noexcept -> const SampleMetadata* {
        return &static_cast<const Sample<T>*>(sample)->metadata;
      };
      result.construct_predicate_view =
          [](void* storage,
             const void* sample,
             InputSelection selection,
             const std::uint64_t* active_generation,
             std::uint64_t expected_generation) noexcept -> expected<void, Error> {
        if (storage == nullptr || sample == nullptr || active_generation == nullptr ||
            reinterpret_cast<std::uintptr_t>(storage) % alignof(PredicateInputView<T>) != 0U) {
          return make_unexpected(Error{ErrorCode::kInvalidArgument});
        }
        auto view = make_runtime_predicate_input_view<T>(PredicateInputBinding<T>{
            .sample = static_cast<const Sample<T>*>(sample),
            .selection = selection,
            .active_generation = active_generation,
            .expected_generation = expected_generation,
        });
        if (!view) {
          return make_unexpected(std::move(view).error());
        }
        std::construct_at(static_cast<PredicateInputView<T>*>(storage), std::move(*view));
        return {};
      };
      result.destroy_predicate_view = [](void* storage) noexcept {
        std::destroy_at(static_cast<PredicateInputView<T>*>(storage));
      };
      if constexpr (kGenericOutputLoanPayload<T>) {
        result.construct_value = [](void* storage) noexcept {
          std::construct_at(static_cast<T*>(storage));
        };
        result.destroy_value = [](void* storage) noexcept {
          std::destroy_at(static_cast<T*>(storage));
        };
      }
      result.bind_input = [](void* declaration, RuntimeInputBinding* binding) noexcept {
        RuntimePortBindingAccess::bind_input(*static_cast<Input<T>*>(declaration), binding);
      };
      result.unbind_input = [](void* declaration, RuntimeInputBinding* binding) noexcept {
        RuntimePortBindingAccess::unbind_input(*static_cast<Input<T>*>(declaration), binding);
      };
      result.bind_output = [](void* declaration, RuntimeOutputBinding* binding) noexcept {
        RuntimePortBindingAccess::bind_output(*static_cast<Output<T>*>(declaration), binding);
      };
      result.unbind_output = [](void* declaration, RuntimeOutputBinding* binding) noexcept {
        RuntimePortBindingAccess::unbind_output(*static_cast<Output<T>*>(declaration), binding);
      };
      result.bind_tensor_output = [](void* declaration, TensorOutputBinding binding) noexcept {
        TensorOutputPortBindingAccess::bind(*static_cast<Output<T>*>(declaration), binding);
      };
      result.unbind_tensor_output = [](void* declaration) noexcept {
        TensorOutputPortBindingAccess::unbind(*static_cast<Output<T>*>(declaration));
      };
    }
    return result;
  }();
  return &ops;
}

}  // namespace detail

}  // namespace holoscan
