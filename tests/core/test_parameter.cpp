/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <forward_list>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <queue>
#include <ratio>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include <holoscan/core/parameter.hpp>
#include <holoscan/core/parameter_schema.hpp>

#include "../../src/core/operator_parameter_binding.hpp"
#include "../../src/core/parameter_schema_test_access.hpp"
#include "operator_test_fixture.hpp"

namespace holoscan {
namespace {

// REQ: PARAM-001
static_assert(sizeof(Operator<NoParams>) == sizeof(OperatorBase),
              "NoParams must add no storage to a typed Operator");

// OpaqueParams: aggregate without a HOLOSCAN_PARAM_SCHEMA declaration; exercises the
// kOpaqueLocal schema availability path and verifies that validate_aggregate still works.
struct OpaqueParams {
  std::uint32_t threshold{4};
};

// PointerParams: aggregate holding a borrowed raw pointer. The non-owning pointer makes
// fixed_object_extent and supports_bounded_materialization false, with zero inline_bytes.
struct PointerParams {
  int* borrowed{};
};

// RestrictedField: value type with noexcept(false) move and throwing swap. Wrapping it in
// RestrictedParams yields a !nothrow_swappable aggregate that disables staged replacement.
struct RestrictedField {
  int value{};

  RestrictedField() = default;
  explicit RestrictedField(int input) : value(input) {}
  RestrictedField(const RestrictedField&) = default;
  RestrictedField& operator=(const RestrictedField&) = default;
  RestrictedField(RestrictedField&& other) noexcept(false) : value(other.value) {}
  RestrictedField& operator=(RestrictedField&& other) noexcept(false) {
    value = other.value;
    return *this;
  }

  friend bool operator==(const RestrictedField&, const RestrictedField&) noexcept = default;
  friend void swap(RestrictedField& left, RestrictedField& right) noexcept(false) {
    std::swap(left.value, right.value);
  }
};

// RestrictedParams: wraps RestrictedField; confirms ParameterAggregate but not
// nothrow_swappable, so replace_member_from_staged is null in local field operations.
struct RestrictedParams {
  RestrictedField value{};
};

static_assert(ParameterAggregate<RestrictedParams>);
static_assert(!std::is_nothrow_swappable_v<RestrictedParams>);

// ThrowingDestructorParams: noexcept(false) destructor is sufficient to disqualify
// ParameterAggregate — the static_assert below confirms !ParameterAggregate<T>.
struct ThrowingDestructorParams {
  int value{};

  ~ThrowingDestructorParams() noexcept(false) {}
};

static_assert(!ParameterAggregate<ThrowingDestructorParams>);

HOLOSCAN_PARAM_SCHEMA(RestrictedParams, holoscan::param_schema("holoscan.test.RestrictedParams", 1),
                      holoscan::field<&RestrictedParams::value>("value"));

// ThrowingBoolProxy: converting to bool is noexcept(false); returned by ThrowingEqualityField's
// operator== so NothrowSemanticEquality<ThrowingEqualityField> is false.
struct ThrowingBoolProxy {
  operator bool() const noexcept(false) { return true; }
};

// ThrowingEqualityField: equality returns a ThrowingBoolProxy; the noexcept(false) bool
// conversion clears semantic_equal_to_staged in the local field operations.
struct ThrowingEqualityField {
  int value{};

  friend ThrowingBoolProxy operator==(const ThrowingEqualityField&,
                                      const ThrowingEqualityField&) noexcept {
    return {};
  }
};

// ThrowingEqualityParams: wraps ThrowingEqualityField; used in
// LocalFieldOperationsSupportStrongQualifiedStaging to confirm the null equality path.
struct ThrowingEqualityParams {
  ThrowingEqualityField value{};
};

HOLOSCAN_PARAM_SCHEMA(ThrowingEqualityParams,
                      holoscan::param_schema("holoscan.test.ThrowingEqualityParams", 1),
                      holoscan::field<&ThrowingEqualityParams::value>("value"));

// NonComparableField: copyable field element without equality. Standard pair and tuple equality
// declarations must not cause a semantic-equality thunk to be instantiated for this type.
struct NonComparableField {
  std::string value;
};

struct NonComparableLess {
  bool operator()(const NonComparableField& left, const NonComparableField& right) const noexcept {
    return left.value < right.value;
  }
};

struct NonComparableHash {
  std::size_t operator()(const NonComparableField& value) const noexcept {
    return std::hash<std::string>{}(value.value);
  }
};

struct NonComparableKeyEqual {
  bool operator()(const NonComparableField& left, const NonComparableField& right) const noexcept {
    return left.value == right.value;
  }
};

using ConstNonComparableVector = const std::vector<NonComparableField>;

std::vector<NonComparableField> non_comparable_defaults;

// ComposedEqualityParams: exercises direct and nested pair/tuple fields, reference-qualified
// elements, and standard containers whose equality declarations do not constrain their bodies.
struct ComposedEqualityParams {
  std::pair<std::int32_t, NonComparableField> pair;
  std::tuple<std::int32_t, NonComparableField> tuple;
  std::vector<std::pair<std::int32_t, NonComparableField>> pairs;
  std::tuple<ConstNonComparableVector> tuple_const_vector;
  std::pair<ConstNonComparableVector, std::int32_t> pair_const_vector;
  std::tuple<const std::vector<NonComparableField>&> tuple_const_vector_reference{
      non_comparable_defaults};
  std::pair<const std::vector<NonComparableField>&, std::int32_t> pair_const_vector_reference{
      non_comparable_defaults, 0};
  std::deque<NonComparableField> deque;
  std::list<NonComparableField> list;
  std::unordered_map<std::int32_t, NonComparableField> unordered_map;
  std::forward_list<NonComparableField> forward_list;
  std::multimap<std::int32_t, NonComparableField> multimap;
  std::variant<std::int32_t, NonComparableField> variant;
  std::unordered_multimap<std::int32_t, NonComparableField> unordered_multimap;
  std::set<NonComparableField, NonComparableLess> set;
  std::multiset<NonComparableField, NonComparableLess> multiset;
  std::unordered_set<NonComparableField, NonComparableHash, NonComparableKeyEqual> unordered_set;
  std::unordered_multiset<NonComparableField, NonComparableHash, NonComparableKeyEqual>
      unordered_multiset;
  std::queue<NonComparableField> queue;
  std::stack<NonComparableField> stack;
};

HOLOSCAN_PARAM_SCHEMA(
    ComposedEqualityParams, holoscan::param_schema("holoscan.test.ComposedEqualityParams", 1),
    holoscan::field<&ComposedEqualityParams::pair>("pair"),
    holoscan::field<&ComposedEqualityParams::tuple>("tuple"),
    holoscan::field<&ComposedEqualityParams::pairs>("pairs"),
    holoscan::field<&ComposedEqualityParams::tuple_const_vector>("tuple_const_vector"),
    holoscan::field<&ComposedEqualityParams::pair_const_vector>("pair_const_vector"),
    holoscan::field<&ComposedEqualityParams::tuple_const_vector_reference>(
        "tuple_const_vector_reference"),
    holoscan::field<&ComposedEqualityParams::pair_const_vector_reference>(
        "pair_const_vector_reference"),
    holoscan::field<&ComposedEqualityParams::deque>("deque"),
    holoscan::field<&ComposedEqualityParams::list>("list"),
    holoscan::field<&ComposedEqualityParams::unordered_map>("unordered_map"),
    holoscan::field<&ComposedEqualityParams::forward_list>("forward_list"),
    holoscan::field<&ComposedEqualityParams::multimap>("multimap"),
    holoscan::field<&ComposedEqualityParams::variant>("variant"),
    holoscan::field<&ComposedEqualityParams::unordered_multimap>("unordered_multimap"),
    holoscan::field<&ComposedEqualityParams::set>("set"),
    holoscan::field<&ComposedEqualityParams::multiset>("multiset"),
    holoscan::field<&ComposedEqualityParams::unordered_set>("unordered_set"),
    holoscan::field<&ComposedEqualityParams::unordered_multiset>("unordered_multiset"),
    holoscan::field<&ComposedEqualityParams::queue>("queue"),
    holoscan::field<&ComposedEqualityParams::stack>("stack"));

static_assert(!detail::SemanticEquality<std::pair<std::int32_t, NonComparableField>>);
static_assert(!detail::SemanticEquality<std::tuple<std::int32_t, NonComparableField>>);
static_assert(!detail::SemanticEquality<std::vector<std::pair<std::int32_t, NonComparableField>>>);
static_assert(!detail::SemanticEquality<std::tuple<ConstNonComparableVector>>);
static_assert(!detail::SemanticEquality<std::pair<ConstNonComparableVector, std::int32_t>>);
static_assert(!detail::SemanticEquality<std::tuple<const std::vector<NonComparableField>&>>);
static_assert(
    !detail::SemanticEquality<std::pair<const std::vector<NonComparableField>&, std::int32_t>>);
static_assert(!detail::SemanticEquality<std::deque<NonComparableField>>);
static_assert(!detail::SemanticEquality<std::list<NonComparableField>>);
static_assert(!detail::SemanticEquality<std::unordered_map<std::int32_t, NonComparableField>>);
static_assert(!detail::SemanticEquality<std::forward_list<NonComparableField>>);
static_assert(!detail::SemanticEquality<std::multimap<std::int32_t, NonComparableField>>);
static_assert(!detail::SemanticEquality<std::variant<std::int32_t, NonComparableField>>);
static_assert(!detail::SemanticEquality<std::unordered_multimap<std::int32_t, NonComparableField>>);
static_assert(!detail::SemanticEquality<std::set<NonComparableField, NonComparableLess>>);
static_assert(!detail::SemanticEquality<std::multiset<NonComparableField, NonComparableLess>>);
static_assert(!detail::SemanticEquality<
              std::unordered_set<NonComparableField, NonComparableHash, NonComparableKeyEqual>>);
static_assert(
    !detail::SemanticEquality<
        std::unordered_multiset<NonComparableField, NonComparableHash, NonComparableKeyEqual>>);
static_assert(!detail::SemanticEquality<std::queue<NonComparableField>>);
static_assert(!detail::SemanticEquality<std::stack<NonComparableField>>);
static_assert(detail::SemanticEquality<std::pair<std::int32_t, std::string>>);
static_assert(detail::SemanticEquality<std::tuple<std::int32_t, std::string>>);
static_assert(detail::SemanticEquality<std::vector<std::string>>);
static_assert(detail::SemanticEquality<std::optional<std::string>>);
static_assert(detail::SemanticEquality<std::forward_list<std::string>>);
static_assert(detail::SemanticEquality<std::multimap<std::int32_t, std::string>>);
static_assert(detail::SemanticEquality<std::variant<std::int32_t, std::string>>);
static_assert(detail::SemanticEquality<std::unordered_multimap<std::int32_t, std::string>>);
static_assert(detail::SemanticEquality<std::set<std::string>>);
static_assert(detail::SemanticEquality<std::multiset<std::string>>);
static_assert(detail::SemanticEquality<std::unordered_set<std::string>>);
static_assert(detail::SemanticEquality<std::unordered_multiset<std::string>>);
static_assert(detail::SemanticEquality<std::queue<std::string>>);
static_assert(detail::SemanticEquality<std::stack<std::string>>);

// SignedMode / UnsignedMode: enum types with signed vs unsigned underlying types; used in
// EnumLogicalTypesFollowUnderlyingSignedness to verify kSignedInteger vs kUnsignedInteger.
enum class SignedMode : std::int8_t {
  kNegative = -1,
};

enum class UnsignedMode : std::uint8_t {
  kPositive = 1,
};

// EnumParams: holds one signed-enum and one unsigned-enum field; the schema verifies that
// logical_type follows the enum's underlying integer signedness (REQ: PARAM-003).
struct EnumParams {
  SignedMode signed_mode{SignedMode::kNegative};
  UnsignedMode unsigned_mode{UnsignedMode::kPositive};
};

HOLOSCAN_PARAM_SCHEMA(EnumParams, holoscan::param_schema("holoscan.test.EnumParams", 1),
                      holoscan::field<&EnumParams::signed_mode>("signed_mode")
                          .enum_values({param_enum_value("negative", SignedMode::kNegative)}),
                      holoscan::field<&EnumParams::unsigned_mode>("unsigned_mode")
                          .enum_values({param_enum_value("positive", UnsignedMode::kPositive)}));

// NamedMode / NamedEnumParams: direct, optional, and sequence enum fields share one explicit
// authored name/value domain without relying on compiler-derived C++ identifiers.
enum class NamedMode : std::uint8_t {
  kDisabled = 0,
  kAutomatic = 2,
  kManual = 7,
};

struct NamedEnumParams {
  NamedMode mode{NamedMode::kAutomatic};
  std::optional<NamedMode> fallback{};
  std::vector<NamedMode> modes{NamedMode::kDisabled, NamedMode::kManual};
};

HOLOSCAN_PARAM_SCHEMA(NamedEnumParams, holoscan::param_schema("holoscan.test.NamedEnumParams", 1),
                      holoscan::field<&NamedEnumParams::mode>("mode").enum_values(
                          {param_enum_value("disabled", NamedMode::kDisabled),
                           param_enum_value("automatic", NamedMode::kAutomatic),
                           param_enum_value("manual", NamedMode::kManual)}),
                      holoscan::field<&NamedEnumParams::fallback>("fallback")
                          .enum_values({param_enum_value("disabled", NamedMode::kDisabled),
                                        param_enum_value("automatic", NamedMode::kAutomatic),
                                        param_enum_value("manual", NamedMode::kManual)}),
                      holoscan::field<&NamedEnumParams::modes>("modes")
                          .enum_values({param_enum_value("disabled", NamedMode::kDisabled),
                                        param_enum_value("automatic", NamedMode::kAutomatic),
                                        param_enum_value("manual", NamedMode::kManual)})
                          .max_elements(4));

struct NamedEnumShapeA {
  NamedMode mode{NamedMode::kAutomatic};
};

HOLOSCAN_PARAM_SCHEMA(NamedEnumShapeA, holoscan::param_schema("holoscan.test.NamedEnumShape", 1),
                      holoscan::field<&NamedEnumShapeA::mode>("mode").enum_values(
                          {param_enum_value("disabled", NamedMode::kDisabled),
                           param_enum_value("automatic", NamedMode::kAutomatic),
                           param_enum_value("manual", NamedMode::kManual)}));

struct NamedEnumShapeB {
  NamedMode mode{NamedMode::kAutomatic};
};

HOLOSCAN_PARAM_SCHEMA(NamedEnumShapeB, holoscan::param_schema("holoscan.test.NamedEnumShape", 1),
                      holoscan::field<&NamedEnumShapeB::mode>("mode").enum_values(
                          {param_enum_value("disabled", NamedMode::kDisabled),
                           param_enum_value("auto", NamedMode::kAutomatic),
                           param_enum_value("manual", NamedMode::kManual)}));

struct UnnamedEnumParams {
  NamedMode mode{NamedMode::kAutomatic};
};

HOLOSCAN_PARAM_SCHEMA(UnnamedEnumParams,
                      holoscan::param_schema("holoscan.test.UnnamedEnumParams", 1),
                      holoscan::field<&UnnamedEnumParams::mode>("mode").allowed_values(
                          {NamedMode::kDisabled, NamedMode::kAutomatic}));

constexpr auto kUnnamedEnumDomain =
    holoscan::field<&NamedEnumShapeA::mode>("mode").allowed_values({NamedMode::kAutomatic});
static_assert(!kUnnamedEnumDomain.invalid());
constexpr auto kEmptyNamedEnumDomain =
    holoscan::field<&NamedEnumShapeA::mode>("mode").enum_values<NamedMode>({});
static_assert(kEmptyNamedEnumDomain.invalid());
constexpr auto kDuplicateEnumLabel = holoscan::field<&NamedEnumShapeA::mode>("mode").enum_values(
    {param_enum_value("mode", NamedMode::kDisabled),
     param_enum_value("mode", NamedMode::kAutomatic)});
static_assert(kDuplicateEnumLabel.invalid());

template <typename T, bool DefaultConstructible = false>
class StatefulAllocator {
 public:
  using value_type = T;

  template <typename U>
  struct rebind {
    using other = StatefulAllocator<U, DefaultConstructible>;
  };

  StatefulAllocator()
    requires(DefaultConstructible)
  = default;
  StatefulAllocator()
    requires(!DefaultConstructible)
  = delete;
  explicit StatefulAllocator(std::uint32_t identity) noexcept : identity_(identity) {}

  template <typename U>
  StatefulAllocator(const StatefulAllocator<U, DefaultConstructible>& other) noexcept
      : identity_(other.identity()) {}

  [[nodiscard]] T* allocate(std::size_t count) { return std::allocator<T>{}.allocate(count); }

  void deallocate(T* pointer, std::size_t count) noexcept {
    std::allocator<T>{}.deallocate(pointer, count);
  }

  [[nodiscard]] std::uint32_t identity() const noexcept { return identity_; }

  template <typename U>
  [[nodiscard]] bool operator==(
      const StatefulAllocator<U, DefaultConstructible>& other) const noexcept {
    return identity_ == other.identity();
  }

 private:
  std::uint32_t identity_{};
};

using StatefulScalarSequence = std::vector<std::int32_t, StatefulAllocator<std::int32_t>>;
using StatefulStringSequence = std::vector<std::string, StatefulAllocator<std::string>>;
using DefaultableStatefulScalarSequence =
    std::vector<std::int32_t, StatefulAllocator<std::int32_t, true>>;

struct StatefulAllocatorParams {
  StatefulScalarSequence scalars{StatefulAllocator<std::int32_t>{17U}};
  StatefulStringSequence strings{StatefulAllocator<std::string>{23U}};
  std::optional<StatefulScalarSequence> optional_scalars;
  std::optional<StatefulStringSequence> optional_strings;
  std::optional<DefaultableStatefulScalarSequence> defaultable_optional_scalars{
      std::in_place, StatefulAllocator<std::int32_t, true>{29U}};
};

static_assert(ParameterAggregate<StatefulAllocatorParams>);

HOLOSCAN_PARAM_SCHEMA(
    StatefulAllocatorParams, holoscan::param_schema("holoscan.test.StatefulAllocatorParams", 1),
    holoscan::field<&StatefulAllocatorParams::scalars>("scalars"),
    holoscan::field<&StatefulAllocatorParams::strings>("strings"),
    holoscan::field<&StatefulAllocatorParams::optional_scalars>("optional_scalars"),
    holoscan::field<&StatefulAllocatorParams::optional_strings>("optional_strings"),
    holoscan::field<&StatefulAllocatorParams::defaultable_optional_scalars>(
        "defaultable_optional_scalars"));

// PartialParams: schema with kDeclaredFieldsOnly coverage (.partial()); the single declared
// field carries runtime_mutable, external_tuning, persisted, and a numeric range constraint.
struct PartialParams {
  std::uint32_t threshold{4};
  bool unmanaged_debug{};
};

HOLOSCAN_PARAM_SCHEMA(PartialParams,
                      holoscan::param_schema("holoscan.test.PartialParams", 1).partial(),
                      holoscan::field<&PartialParams::threshold>("threshold")
                          .runtime_mutable()
                          .range(1U, 8U)
                          .persisted()
                          .external_tuning());

// CompleteParams: full-coverage schema with label (compile-time), gain (runtime-mutable),
// and workspace_mb (lifecycle-mutable at kAllocate); exercises all three mutability classes.
struct CompleteParams {
  std::string label{"camera"};
  float gain{1.0F};
  std::uint32_t workspace_mb{512};
};

HOLOSCAN_PARAM_SCHEMA(
    CompleteParams, holoscan::param_schema("holoscan.test.CompleteParams", 3),
    holoscan::field<&CompleteParams::label>("label")
        .compile_time()
        .encoding(ParamStringEncoding::kUtf8)
        .max_utf8_bytes(16)
        .description("Stable input label."),
    holoscan::field<&CompleteParams::gain>("gain").runtime_mutable().range(0.0F, 4.0F).persisted(),
    holoscan::field<&CompleteParams::workspace_mb>("workspace_mb")
        .lifecycle_mutable(LifecycleStage::kAllocate)
        .range(256U, 2048U)
        .internal());

// PolicyParams: four fields, each exercising a distinct access/persistence/sensitivity
// policy combination; confirms both descriptor and document views propagate all policies.
struct PolicyParams {
  std::uint32_t required{};
  std::uint32_t readable{};
  std::uint32_t secret{};
  std::uint32_t scratch{};
};

HOLOSCAN_PARAM_SCHEMA(PolicyParams, holoscan::param_schema("holoscan.test.PolicyParams", 1),
                      holoscan::field<&PolicyParams::required>("required")
                          .required_on_load()
                          .alias("legacy_required"),
                      holoscan::field<&PolicyParams::readable>("readable").external_read_only(),
                      holoscan::field<&PolicyParams::secret>("secret").sensitive(),
                      holoscan::field<&PolicyParams::scratch>("scratch").transient());

// InferredPolicyEvidenceParams: schema where all policy fields use inferred defaults,
// leaving policy_evidence.complete() == false. Paired with ExplicitPolicyEvidenceParams
// to verify contract_hash is unchanged while metadata_hash differs.
struct InferredPolicyEvidenceParams {
  std::uint32_t value{};
};

HOLOSCAN_PARAM_SCHEMA(InferredPolicyEvidenceParams,
                      holoscan::param_schema("holoscan.test.PolicyEvidenceParams", 1),
                      holoscan::field<&InferredPolicyEvidenceParams::value>("value"));

// ExplicitPolicyEvidenceParams: same stable_id as InferredPolicyEvidenceParams but all three
// policy axes set explicitly, making policy_evidence.complete() == true.
struct ExplicitPolicyEvidenceParams {
  std::uint32_t value{};
};

HOLOSCAN_PARAM_SCHEMA(ExplicitPolicyEvidenceParams,
                      holoscan::param_schema("holoscan.test.PolicyEvidenceParams", 1),
                      holoscan::field<&ExplicitPolicyEvidenceParams::value>("value")
                          .persistence(ParamPersistence::kPersisted)
                          .external_access(ParamExternalAccess::kNone)
                          .sensitivity(ParamSensitivity::kNormal));

constexpr auto kInvalidExplicitPolicy =
    holoscan::field<&ExplicitPolicyEvidenceParams::value>("value").external_access(
        static_cast<ParamExternalAccess>(0xFFU));
static_assert(kInvalidExplicitPolicy.invalid());
constexpr auto kConflictingExplicitPolicy =
    holoscan::field<&ExplicitPolicyEvidenceParams::value>("value")
        .external_access(ParamExternalAccess::kNone)
        .external_tuning();
static_assert(kConflictingExplicitPolicy.invalid());

// MutabilityTransitionParams: each field sets a lifecycle stage then overrides it with a
// terminal mutability marker; the last marker must win and clear intermediate lifecycle state.
// Paired with CanonicalMutabilityParams to confirm identical contract and metadata hashes.
struct MutabilityTransitionParams {
  std::uint32_t compile_time{};
  std::uint32_t runtime{};
  std::uint32_t runtime_after_invalid{};
  std::uint32_t recompile{};
};

HOLOSCAN_PARAM_SCHEMA(
    MutabilityTransitionParams,
    holoscan::param_schema("holoscan.test.MutabilityTransitionParams", 1),
    holoscan::field<&MutabilityTransitionParams::compile_time>("compile_time")
        .lifecycle_mutable(LifecycleStage::kAllocate)
        .compile_time(),
    holoscan::field<&MutabilityTransitionParams::runtime>("runtime")
        .lifecycle_mutable(LifecycleStage::kAllocate)
        .runtime_mutable(),
    holoscan::field<&MutabilityTransitionParams::runtime_after_invalid>("runtime_after_invalid")
        .lifecycle_mutable(LifecycleStage::kRelease)
        .runtime_mutable(),
    holoscan::field<&MutabilityTransitionParams::recompile>("recompile")
        .lifecycle_mutable(LifecycleStage::kStart)
        .recompile_required());

// CanonicalMutabilityParams: equivalent to MutabilityTransitionParams using only terminal
// markers (no intermediate lifecycle stage); confirms last-marker-wins (REQ: PARAM-015).
struct CanonicalMutabilityParams {
  std::uint32_t compile_time{};
  std::uint32_t runtime{};
  std::uint32_t runtime_after_invalid{};
  std::uint32_t recompile{};
};

HOLOSCAN_PARAM_SCHEMA(
    CanonicalMutabilityParams,
    holoscan::param_schema("holoscan.test.MutabilityTransitionParams", 1),
    holoscan::field<&CanonicalMutabilityParams::compile_time>("compile_time").compile_time(),
    holoscan::field<&CanonicalMutabilityParams::runtime>("runtime").runtime_mutable(),
    holoscan::field<&CanonicalMutabilityParams::runtime_after_invalid>("runtime_after_invalid")
        .runtime_mutable(),
    holoscan::field<&CanonicalMutabilityParams::recompile>("recompile").recompile_required());

// AllowedValueParams: a single integer field restricted to {1, 2, 4}; exercises the
// allowed_values constraint path and its precedence over cardinality/encoding checks.
struct AllowedValueParams {
  std::int32_t mode{2};
};

HOLOSCAN_PARAM_SCHEMA(AllowedValueParams,
                      holoscan::param_schema("holoscan.test.AllowedValueParams", 1),
                      holoscan::field<&AllowedValueParams::mode>("mode").allowed_values({1, 2, 4}));

// OptionalParams: absent fields remain valid, while present values use the same scalar,
// cardinality, and encoding constraints as their underlying types (REQ: PARAM-054).
struct OptionalParams {
  std::optional<std::int32_t> count{};
  std::optional<std::string> label{};
  std::optional<std::vector<std::string>> tags{};
};

HOLOSCAN_PARAM_SCHEMA(
    OptionalParams, holoscan::param_schema("holoscan.test.OptionalParams", 1),
    holoscan::field<&OptionalParams::count>("count").range(1, 8).allowed_values({2, 4, 6}),
    holoscan::field<&OptionalParams::label>("label")
        .encoding(ParamStringEncoding::kUtf8)
        .max_utf8_bytes(8),
    holoscan::field<&OptionalParams::tags>("tags")
        .element_encoding(ParamStringEncoding::kUtf8)
        .max_element_utf8_bytes(4)
        .max_elements(2));

// PathParams: filesystem path field; logical type = kFilesystemPath, has_stable_shape = false,
// has_portable_codec = false — but aggregate validation still passes.
struct PathParams {
  std::filesystem::path model{"models/default.plan"};
};

HOLOSCAN_PARAM_SCHEMA(PathParams, holoscan::param_schema("holoscan.test.PathParams", 1),
                      holoscan::field<&PathParams::model>("model"));

// PathVectorParams: sequence of filesystem paths with max_elements(4); element logical type
// is kFilesystemPath, has_stable_shape = false, no portable codec.
struct PathVectorParams {
  std::vector<std::filesystem::path> models{};
};

HOLOSCAN_PARAM_SCHEMA(PathVectorParams, holoscan::param_schema("holoscan.test.PathVectorParams", 1),
                      holoscan::field<&PathVectorParams::models>("models").max_elements(4));

// LifecycleStageParams: one field per LifecycleStage enum value; verifies that every
// valid forward stage is accepted and stored in earliest_invalidated_stage.
struct LifecycleStageParams {
  int configure{};
  int discover{};
  int allocate{};
  int warm_up{};
  int arm{};
  int start{};
};

HOLOSCAN_PARAM_SCHEMA(
    LifecycleStageParams, holoscan::param_schema("holoscan.test.LifecycleStageParams", 1),
    holoscan::field<&LifecycleStageParams::configure>("configure")
        .lifecycle_mutable(LifecycleStage::kConfigure),
    holoscan::field<&LifecycleStageParams::discover>("discover")
        .lifecycle_mutable(LifecycleStage::kDiscover),
    holoscan::field<&LifecycleStageParams::allocate>("allocate")
        .lifecycle_mutable(LifecycleStage::kAllocate),
    holoscan::field<&LifecycleStageParams::warm_up>("warm_up").lifecycle_mutable(
        LifecycleStage::kWarmUp),
    holoscan::field<&LifecycleStageParams::arm>("arm").lifecycle_mutable(LifecycleStage::kArm),
    holoscan::field<&LifecycleStageParams::start>("start").lifecycle_mutable(
        LifecycleStage::kStart));

// ShapeInt32Params: 32-bit signed scalar; paired with ShapeInt64Params — differing
// bit_width must produce distinct structural_hash values (REQ: PARAM-004).
struct ShapeInt32Params {
  std::int32_t value{};
};

HOLOSCAN_PARAM_SCHEMA(ShapeInt32Params, holoscan::param_schema("holoscan.test.ShapeScalar", 1),
                      holoscan::field<&ShapeInt32Params::value>("value"));

// ShapeInt64Params: 64-bit counterpart to ShapeInt32Params; different bit_width produces a
// different structural_hash even though the schema stable_id is identical.
struct ShapeInt64Params {
  std::int64_t value{};
};

HOLOSCAN_PARAM_SCHEMA(ShapeInt64Params, holoscan::param_schema("holoscan.test.ShapeScalar", 1),
                      holoscan::field<&ShapeInt64Params::value>("value"));

// ShapeOptionalInt32Params: same stable schema ID and underlying scalar shape as
// ShapeInt32Params; explicit optionality must make its structural hash distinct.
struct ShapeOptionalInt32Params {
  std::optional<std::int32_t> value{};
};

HOLOSCAN_PARAM_SCHEMA(ShapeOptionalInt32Params,
                      holoscan::param_schema("holoscan.test.ShapeScalar", 1),
                      holoscan::field<&ShapeOptionalInt32Params::value>("value"));

// ShapeVectorIntParams: sequence of int32; verifies kSequence logical type with
// element_or_rep_type = kSignedInteger and element_or_rep_bit_width = 32.
struct ShapeVectorIntParams {
  std::vector<std::int32_t> values{};
};

HOLOSCAN_PARAM_SCHEMA(ShapeVectorIntParams, holoscan::param_schema("holoscan.test.ShapeVector", 1),
                      holoscan::field<&ShapeVectorIntParams::values>("values").max_elements(4));

// ShapeVectorFloatParams: sequence of floats; element_or_rep_type = kFloatingPoint;
// structural_hash must differ from ShapeVectorIntParams.
struct ShapeVectorFloatParams {
  std::vector<float> values{};
};

HOLOSCAN_PARAM_SCHEMA(ShapeVectorFloatParams,
                      holoscan::param_schema("holoscan.test.ShapeVector", 1),
                      holoscan::field<&ShapeVectorFloatParams::values>("values").max_elements(4));

// ShapeVectorDurationParams: sequence of durations; has_stable_shape = false because
// duration period metadata is not encodable in a portable structural hash.
struct ShapeVectorDurationParams {
  std::vector<std::chrono::milliseconds> values{};
};

HOLOSCAN_PARAM_SCHEMA(
    ShapeVectorDurationParams, holoscan::param_schema("holoscan.test.ShapeVectorDuration", 1),
    holoscan::field<&ShapeVectorDurationParams::values>("values").max_elements(4));

// ShapeMillisecondsParams: single duration field; verifies duration_period_numerator = 1
// and duration_period_denominator = 1000 in the type_shape.
struct ShapeMillisecondsParams {
  std::chrono::milliseconds value{};
};

HOLOSCAN_PARAM_SCHEMA(ShapeMillisecondsParams,
                      holoscan::param_schema("holoscan.test.ShapeDuration", 1),
                      holoscan::field<&ShapeMillisecondsParams::value>("value"));

// ShapeMicrosecondsParams: microsecond duration; denominator = 1_000_000 → different
// structural_hash from ShapeMillisecondsParams.
struct ShapeMicrosecondsParams {
  std::chrono::microseconds value{};
};

HOLOSCAN_PARAM_SCHEMA(ShapeMicrosecondsParams,
                      holoscan::param_schema("holoscan.test.ShapeDuration", 1),
                      holoscan::field<&ShapeMicrosecondsParams::value>("value"));

// ShapeBound8Params: UTF-8 string with max_utf8_bytes = 8; the bound enters the
// structural_hash so structural_hash must differ from ShapeBound16Params.
struct ShapeBound8Params {
  std::string value{};
};

HOLOSCAN_PARAM_SCHEMA(ShapeBound8Params, holoscan::param_schema("holoscan.test.ShapeBound", 1),
                      holoscan::field<&ShapeBound8Params::value>("value")
                          .encoding(ParamStringEncoding::kUtf8)
                          .max_utf8_bytes(8));

// ShapeBound16Params: UTF-8 string with max_utf8_bytes = 16; structural_hash must
// differ from ShapeBound8Params (same schema stable_id, different bound).
struct ShapeBound16Params {
  std::string value{};
};

HOLOSCAN_PARAM_SCHEMA(ShapeBound16Params, holoscan::param_schema("holoscan.test.ShapeBound", 1),
                      holoscan::field<&ShapeBound16Params::value>("value")
                          .encoding(ParamStringEncoding::kUtf8)
                          .max_utf8_bytes(16));

// EncodedBoundParams: max_encoded_bytes is a portable codec bound, not an aggregate
// constraint; validate_params must pass regardless of element count.
struct EncodedBoundParams {
  std::vector<std::int32_t> values{1, 2, 3};
};

HOLOSCAN_PARAM_SCHEMA(EncodedBoundParams,
                      holoscan::param_schema("holoscan.test.EncodedBoundParams", 1),
                      holoscan::field<&EncodedBoundParams::values>("values").max_encoded_bytes(1));

// BinaryStringParams: binary string field (no kUtf8 encoding); arbitrary byte sequences
// must pass aggregate validation without encoding checks.
struct BinaryStringParams {
  std::string bytes{};
};

HOLOSCAN_PARAM_SCHEMA(BinaryStringParams,
                      holoscan::param_schema("holoscan.test.BinaryStringParams", 1),
                      holoscan::field<&BinaryStringParams::bytes>("bytes"));

// Utf8StringVectorParams: UTF-8 string vector with max_elements and per-element max_utf8_bytes;
// exercises cardinality-before-encoding validation order.
struct Utf8StringVectorParams {
  std::vector<std::string> values{"first", "second"};
};

HOLOSCAN_PARAM_SCHEMA(Utf8StringVectorParams,
                      holoscan::param_schema("holoscan.test.Utf8StringVectorParams", 1),
                      holoscan::field<&Utf8StringVectorParams::values>("values")
                          .element_encoding(ParamStringEncoding::kUtf8)
                          .max_elements(4)
                          .max_element_utf8_bytes(8));

// BinaryStringVectorParams: binary string vector with no encoding constraint; arbitrary
// byte sequences in elements must pass validation.
struct BinaryStringVectorParams {
  std::vector<std::string> values{};
};

HOLOSCAN_PARAM_SCHEMA(BinaryStringVectorParams,
                      holoscan::param_schema("holoscan.test.BinaryStringVectorParams", 1),
                      holoscan::field<&BinaryStringVectorParams::values>("values"));

// NestedParams: small aggregate with a range-constrained count field; referenced as a nested
// schema by ValidationParams and ComposedParams to test nested field validation.
struct NestedParams {
  std::uint32_t count{1};
};

HOLOSCAN_PARAM_SCHEMA(NestedParams, holoscan::param_schema("holoscan.test.NestedParams", 1),
                      holoscan::field<&NestedParams::count>("count").range(1U, 4U));

// Custom validator for ValidationParams.even; the validator ID "holoscan.test.validator.even.v1"
// and kValidatorRejected code must propagate through validate_params.
[[nodiscard]] bool even_value(const std::uint32_t& value) noexcept {
  return value % 2U == 0U;
}

// ValidationParams: exercises every constraint category in one aggregate: numeric range,
// duration range, UTF-8 encoding, max_utf8_bytes, custom validator, and nested schema.
struct ValidationParams {
  std::int64_t signed_value{-1};
  std::chrono::milliseconds timeout{5};
  std::string utf8{"valid"};
  std::uint32_t even{2};
  NestedParams nested{};
};

HOLOSCAN_PARAM_SCHEMA(ValidationParams, holoscan::param_schema("holoscan.test.ValidationParams", 1),
                      holoscan::field<&ValidationParams::signed_value>("signed_value")
                          .range(std::int64_t{-4}, std::int64_t{4}),
                      holoscan::field<&ValidationParams::timeout>("timeout").range(
                          std::chrono::milliseconds{1}, std::chrono::milliseconds{10}),
                      holoscan::field<&ValidationParams::utf8>("utf8")
                          .encoding(ParamStringEncoding::kUtf8)
                          .max_utf8_bytes(16),
                      holoscan::field<&ValidationParams::even>("even").validator<even_value>(
                          "holoscan.test.validator.even.v1"),
                      holoscan::field<&ValidationParams::nested>("nested").nested<NestedParams>());

// DurationRangeParams: mixed-unit duration ranges (microseconds fields bounded by both
// microseconds and milliseconds); verifies cross-unit conversion to the field's canonical period.
struct DurationRangeParams {
  std::chrono::microseconds mixed_bounds{500};
  std::chrono::microseconds coarser_bounds{1500};
};

HOLOSCAN_PARAM_SCHEMA(DurationRangeParams,
                      holoscan::param_schema("holoscan.test.DurationRangeParams", 1),
                      holoscan::field<&DurationRangeParams::mixed_bounds>("mixed_bounds")
                          .range(std::chrono::microseconds{100}, std::chrono::milliseconds{2}),
                      holoscan::field<&DurationRangeParams::coarser_bounds>("coarser_bounds")
                          .range(std::chrono::milliseconds{1}, std::chrono::milliseconds{2}));

constexpr auto kFractionalDurationRange =
    holoscan::field<&DurationRangeParams::mixed_bounds>("mixed_bounds")
        .range(std::chrono::nanoseconds{1}, std::chrono::microseconds{1});
static_assert(kFractionalDurationRange.invalid());

using HugeHours = std::chrono::duration<std::uint64_t, std::ratio<3600, 1>>;
constexpr auto kOverflowingDurationRange =
    holoscan::field<&DurationRangeParams::mixed_bounds>("mixed_bounds")
        .range(HugeHours{0}, HugeHours{std::numeric_limits<std::uint64_t>::max()});
static_assert(kOverflowingDurationRange.invalid());

// ExtremeDurationRangeParams: field with an extreme-period duration; static_asserts above
// verify compile-time overflow detection in cross-unit duration range conversions.
struct ExtremeDurationRangeParams {
  std::chrono::nanoseconds timeout{0};
};

using ExtremePeriod =
    std::ratio<std::numeric_limits<std::intmax_t>::max(), std::intmax_t{999'999'937}>;
using ExtremeDuration = std::chrono::duration<std::int64_t, ExtremePeriod>;
constexpr auto kZeroExtremeDurationRange =
    holoscan::field<&ExtremeDurationRangeParams::timeout>("timeout").range(ExtremeDuration{0},
                                                                           ExtremeDuration{0});
static_assert(!kZeroExtremeDurationRange.invalid());
constexpr auto kOverflowingExtremeDurationRange =
    holoscan::field<&ExtremeDurationRangeParams::timeout>("timeout").range(ExtremeDuration{0},
                                                                           ExtremeDuration{1});
static_assert(kOverflowingExtremeDurationRange.invalid());

// ComposedParams: parent schema containing a nested NestedParams field; verifies that
// nested schema ID, version, and hashes propagate into the parent's field descriptor.
struct ComposedParams {
  NestedParams nested{};
};

HOLOSCAN_PARAM_SCHEMA(ComposedParams, holoscan::param_schema("holoscan.test.ComposedParams", 1),
                      holoscan::field<&ComposedParams::nested>("nested").nested<NestedParams>());

// OptionalComposedParams supplies a second nesting level whose mutable local traversal operation
// engages an absent optional before continuing through ComposedParams to NestedParams.
struct OptionalComposedParams {
  std::optional<ComposedParams> composed{};
};

HOLOSCAN_PARAM_SCHEMA(
    OptionalComposedParams, holoscan::param_schema("holoscan.test.OptionalComposedParams", 1),
    holoscan::field<&OptionalComposedParams::composed>("composed").nested<ComposedParams>());

// PartialNestedParent: parent schema nesting a PartialParams (partial-coverage schema);
// the nested field must have has_stable_shape = false because the nested schema is partial.
struct PartialNestedParent {
  PartialParams nested{};
};

HOLOSCAN_PARAM_SCHEMA(
    PartialNestedParent, holoscan::param_schema("holoscan.test.PartialNestedParent", 1),
    holoscan::field<&PartialNestedParent::nested>("nested").nested<PartialParams>());

// NestedShape32 / NestedShape64: nested shape pair with int32 vs int64 fields; same
// stable_id but different structural_hash — difference propagates upward into parent hashes.
struct NestedShape32 {
  std::int32_t value{};
};

HOLOSCAN_PARAM_SCHEMA(NestedShape32, holoscan::param_schema("holoscan.test.NestedShape", 1),
                      holoscan::field<&NestedShape32::value>("value"));

// NestedShape64: 64-bit counterpart to NestedShape32; different structural_hash propagates
// into NestedParent64, distinguishing it from NestedParent32 (REQ: PARAM-004).
struct NestedShape64 {
  std::int64_t value{};
};

HOLOSCAN_PARAM_SCHEMA(NestedShape64, holoscan::param_schema("holoscan.test.NestedShape", 1),
                      holoscan::field<&NestedShape64::value>("value"));

// NestedParent32: parent schema nesting NestedShape32; structural_hash must differ from
// NestedParent64 because the nested shape's structural_hash propagates upward.
struct NestedParent32 {
  NestedShape32 nested{};
};

HOLOSCAN_PARAM_SCHEMA(NestedParent32, holoscan::param_schema("holoscan.test.NestedParent", 1),
                      holoscan::field<&NestedParent32::nested>("nested").nested<NestedShape32>());

// NestedParent64: parent schema nesting NestedShape64; paired with NestedParent32 to
// confirm nested structural_hash propagation (REQ: PARAM-004).
struct NestedParent64 {
  NestedShape64 nested{};
};

HOLOSCAN_PARAM_SCHEMA(NestedParent64, holoscan::param_schema("holoscan.test.NestedParent", 1),
                      holoscan::field<&NestedParent64::nested>("nested").nested<NestedShape64>());

// ProofParamsA: schema with kAuthorAsserted proof; paired with ProofParamsB to verify that
// proof level does not affect structural, contract, or metadata hashes.
struct ProofParamsA {
  int value{};
};

HOLOSCAN_PARAM_SCHEMA(ProofParamsA, holoscan::param_schema("holoscan.test.ProofParams", 1),
                      holoscan::field<&ProofParamsA::value>("value"));

// ProofParamsB: schema with kProven proof (via ParamSchemaTestAccess::proven); same hashes
// as ProofParamsA, confirming proof level is metadata-only and hash-neutral.
struct ProofParamsB {
  int value{};
};

HOLOSCAN_PARAM_SCHEMA(ProofParamsB,
                      holoscan::detail::ParamSchemaTestAccess::proven(
                          holoscan::param_schema("holoscan.test.ProofParams", 1)),
                      holoscan::field<&ProofParamsB::value>("value"));

// Minimal Operator<CompleteParams> used in binding tests; verifies that bind_initial
// validates the aggregate before replacing it, and rejects out-of-range values atomically.
class CompleteParamOperator final : public Operator<CompleteParams> {
 public:
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Minimal Operator<RestrictedParams>; bind_initial must return kNotSupported when the
// aggregate's non-nothrow-swappable field disables staged replacement.
class RestrictedParamOperator final : public Operator<RestrictedParams> {
 public:
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Template operator that captures a pointer to its params block and the generation counter
// at compute() time; used to verify the typed callback path delivers the correct state.
template <typename ParamsT>
class ObservingParamOperator final : public Operator<ParamsT> {
 public:
  expected<void, Error> compute(ExecutionContext& context) override {
    observed = &context.params<ParamsT>();
    generation = context.params_generation();
    return {};
  }

  const ParamsT* observed{};
  ParamGeneration generation{99};
};

using CompleteParamsAlias = CompleteParams;

// Test concepts probing the public schema query surface. The static_asserts below confirm
// each query is only available for non-const, non-reference, schema-registered types.
template <typename T>
concept HasLocalFieldOperations = requires(T value) { value.local_operations; };

template <typename T>
concept HasSchemaAvailabilityQuery = requires { param_schema_availability<T>(); };

template <typename T>
concept HasSchemaDescriptorQuery = requires { param_schema_descriptor<T>(); };

template <typename T>
concept HasSchemaDocumentQuery = requires { param_schema_document<T>(); };

template <typename T>
concept HasStaticSchemaQuery = requires { param_schema<T>(); };

template <typename T>
concept HasAggregateValidationQuery =
    requires(const std::remove_cvref_t<T>& value) { validate_params<T>(value); };

template <typename T>
concept HasInitialGenerationQuery = requires { initial_param_generation<T>(); };

// REQ: PARAM-001
static_assert(std::is_aggregate_v<OpaqueParams>);
static_assert(
    std::same_as<decltype(param_type_descriptor<CompleteParams>()), const ParamTypeDescriptor&>);
static_assert(std::same_as<CompleteParamsAlias, CompleteParams>);
static_assert(HasInitialGenerationQuery<CompleteParams>);
static_assert(!HasInitialGenerationQuery<const CompleteParams>);
static_assert(!HasInitialGenerationQuery<int>);

// REQ: PARAM-002
static_assert(std::same_as<decltype(param_schema<CompleteParams>()),
                           const StaticParamSchema<CompleteParams, 3>&>);
static_assert(
    std::same_as<decltype(param_ref<&CompleteParams::gain>()), ParamRef<&CompleteParams::gain>>);
static_assert(std::same_as<decltype(param_ref<&ComposedParams::nested, &NestedParams::count>()),
                           ParamRef<&ComposedParams::nested, &NestedParams::count>>);
// REQ: PARAM-056
static_assert(std::same_as<decltype(param_ref<&OptionalComposedParams::composed,
                                              &ComposedParams::nested, &NestedParams::count>()),
                           ParamRef<&OptionalComposedParams::composed, &ComposedParams::nested,
                                    &NestedParams::count>>);

// REQ: PARAM-008
static_assert(!HasLocalFieldOperations<ParamFieldDocument>);
static_assert(HasSchemaAvailabilityQuery<CompleteParams>);
static_assert(!HasSchemaAvailabilityQuery<const CompleteParams>);
static_assert(!HasSchemaAvailabilityQuery<CompleteParams&>);
static_assert(!HasSchemaAvailabilityQuery<int>);
static_assert(HasSchemaDescriptorQuery<CompleteParams>);
static_assert(!HasSchemaDescriptorQuery<const CompleteParams>);
static_assert(!HasSchemaDescriptorQuery<int>);
static_assert(HasSchemaDocumentQuery<CompleteParams>);
static_assert(!HasSchemaDocumentQuery<CompleteParams&>);
static_assert(!HasSchemaDocumentQuery<int>);
static_assert(HasStaticSchemaQuery<CompleteParams>);
static_assert(!HasStaticSchemaQuery<const CompleteParams>);
static_assert(!HasStaticSchemaQuery<int>);
static_assert(HasAggregateValidationQuery<CompleteParams>);
static_assert(!HasAggregateValidationQuery<const CompleteParams>);
static_assert(!HasAggregateValidationQuery<int>);

// REQ: PARAM-003
static_assert(kParamAllowedValueCapacity == 16U);

// REQ: PARAM-011
static_assert(static_cast<std::uint8_t>(params::ParamProjectionStatus::kIncluded) == 0U);
static_assert(static_cast<std::uint8_t>(params::ParamProjectionStatus::kRedactedSensitive) == 1U);
static_assert(static_cast<std::uint8_t>(params::ParamProjectionStatus::kExcludedInternal) == 2U);
static_assert(static_cast<std::uint8_t>(params::ParamProjectionStatus::kExcludedTransient) == 3U);
static_assert(static_cast<std::uint8_t>(params::ParamProjectionStatus::kExcludedProfilePolicy) ==
              4U);
static_assert(static_cast<std::uint8_t>(params::ParamProjectionStatus::kExcludedUnsupportedCodec) ==
              5U);

// Runs a single compute() cycle with the given initial params and asserts that the operator
// observes the correct params pointer and a generation value of 0 (the initial generation).
template <typename ParamsT>
void expect_initial_typed_callback_path(ParamsT params) {
  ObservingParamOperator<ParamsT> op;
  test::SyntheticOperatorFixture fixture{op};
  ASSERT_TRUE(fixture.declare_ports());
  ASSERT_TRUE(fixture.bind_initial_params(std::move(params)));
  const detail::SyntheticActivationOptions initial{
      .params_generation = ParamGeneration{0},
  };
  ASSERT_TRUE(fixture.invoke_once(initial));
  EXPECT_EQ(op.observed, &op.params());
  EXPECT_EQ(op.generation, 0U);
}

// REQ: PARAM-001
TEST(ParameterType, OpaquePartialAndCompleteTypesHaveExactLocalDescriptors) {
  const auto& opaque = param_type_descriptor<OpaqueParams>();
  const auto& partial = param_type_descriptor<PartialParams>();
  const auto& complete = param_type_descriptor<CompleteParams>();

  EXPECT_NE(opaque.local_type_token, partial.local_type_token);
  EXPECT_NE(partial.local_type_token, complete.local_type_token);
  EXPECT_EQ(complete.local_type_token,
            param_type_descriptor<CompleteParamsAlias>().local_type_token);
  EXPECT_EQ(opaque.object_size, sizeof(OpaqueParams));
  EXPECT_EQ(complete.object_alignment, alignof(CompleteParams));
  EXPECT_EQ(param_schema_availability<OpaqueParams>(), ParamSchemaAvailability::kOpaqueLocal);
  EXPECT_EQ(param_schema_availability<PartialParams>(), ParamSchemaAvailability::kDeclared);
  EXPECT_EQ(param_schema_availability<CompleteParams>(), ParamSchemaAvailability::kDeclared);

  const OpaqueParams opaque_value{};
  EXPECT_TRUE(opaque.validate_aggregate(&opaque_value).ok());
}

// REQ: PARAM-001
TEST(ParameterType, NoParamsUsesTheStaticZeroGenerationDisposition) {
  const auto& descriptor = param_type_descriptor<NoParams>();
  EXPECT_EQ(descriptor.local_type_token, param_type_descriptor<NoParams>().local_type_token);
  EXPECT_FALSE(descriptor.storage.owns_dynamic_storage);
  EXPECT_TRUE(descriptor.storage.fixed_object_extent);
  EXPECT_TRUE(descriptor.storage.supports_bounded_materialization);
  EXPECT_EQ(descriptor.storage.inline_bytes, 0);
  EXPECT_EQ(initial_param_generation<NoParams>(), 0);
}

// REQ: PARAM-001
// REQ: PARAM-002
// REQ: PARAM-010
TEST(ParameterBinding, OpaquePartialAndCompleteUseTheSameTypedCallbackPath) {
  expect_initial_typed_callback_path(OpaqueParams{.threshold = 7});
  expect_initial_typed_callback_path(PartialParams{.threshold = 6, .unmanaged_debug = true});
  expect_initial_typed_callback_path(
      CompleteParams{.label = "stereo", .gain = 2.0F, .workspace_mb = 1024});
}

// REQ: PARAM-002
TEST(ParameterSchema, ModesAndIndependentPoliciesArePreserved) {
  // Key setup: retrieve descriptors for PartialParams, CompleteParams, and ComposedParams.
  const ParamSchemaDescriptor* partial = param_schema_descriptor<PartialParams>();
  const ParamSchemaDescriptor* complete = param_schema_descriptor<CompleteParams>();
  ASSERT_NE(partial, nullptr);
  ASSERT_NE(complete, nullptr);

  // Key assertion: partial schema has kDeclaredFieldsOnly; complete has kComplete with
  // per-field mutability and visibility attributes preserved across the descriptor.
  EXPECT_EQ(partial->coverage, ParamSchemaCoverage::kDeclaredFieldsOnly);
  EXPECT_EQ(partial->proof, ParamSchemaProof::kAuthorAsserted);
  ASSERT_EQ(partial->fields.size(), 1);
  EXPECT_EQ(partial->fields[0].mutability, ParamMutability::kRuntime);
  EXPECT_EQ(partial->fields[0].external_access, ParamExternalAccess::kTuning);

  EXPECT_EQ(complete->coverage, ParamSchemaCoverage::kComplete);
  ASSERT_EQ(complete->fields.size(), 3);
  EXPECT_EQ(complete->fields[0].mutability, ParamMutability::kCompileTime);
  EXPECT_EQ(complete->fields[2].mutability, ParamMutability::kLifecycle);
  EXPECT_EQ(complete->fields[2].earliest_invalidated_stage, LifecycleStage::kAllocate);
  EXPECT_EQ(complete->fields[2].visibility, ParamVisibility::kInternal);

  const ParamSchemaDescriptor* nested = param_schema_descriptor<NestedParams>();
  const ParamSchemaDescriptor* composed = param_schema_descriptor<ComposedParams>();
  ASSERT_NE(nested, nullptr);
  ASSERT_NE(composed, nullptr);
  EXPECT_EQ(nested->coverage, ParamSchemaCoverage::kComplete);
  EXPECT_EQ(nested->proof, ParamSchemaProof::kAuthorAsserted);
  EXPECT_EQ(composed->proof, ParamSchemaProof::kAuthorAsserted);
  EXPECT_TRUE(composed->fields[0].nested);
  EXPECT_EQ(composed->fields[0].nested_schema_id, nested->stable_id);
  EXPECT_EQ(composed->fields[0].nested_schema_version, nested->version);
  EXPECT_EQ(composed->fields[0].nested_structural_hash, nested->structural_hash);
  EXPECT_EQ(composed->fields[0].nested_contract_hash, nested->contract_hash);
}

// REQ: PARAM-056
TEST(ParameterSchema, LocalNestedOperationsTraverseDirectAndOptionalMembersRecursively) {
  const ParamSchemaDescriptor* root_schema = param_schema_descriptor<OptionalComposedParams>();
  const ParamSchemaDescriptor* composed_schema = param_schema_descriptor<ComposedParams>();
  const ParamSchemaDescriptor* nested_schema = param_schema_descriptor<NestedParams>();
  ASSERT_NE(root_schema, nullptr);
  ASSERT_NE(composed_schema, nullptr);
  ASSERT_NE(nested_schema, nullptr);
  ASSERT_EQ(root_schema->fields.size(), 1U);
  ASSERT_EQ(composed_schema->fields.size(), 1U);
  ASSERT_EQ(nested_schema->fields.size(), 1U);

  const ParamFieldLocalOperations& optional_ops = root_schema->fields[0].local_operations;
  ASSERT_NE(optional_ops.mutable_nested_member, nullptr);
  ASSERT_NE(optional_ops.nested_member, nullptr);
  ASSERT_NE(optional_ops.nested_schema, nullptr);
  EXPECT_EQ(optional_ops.nested_schema(), composed_schema);

  OptionalComposedParams params;
  EXPECT_EQ((detail::param_path_value<&OptionalComposedParams::composed,
                                      &ComposedParams::nested,
                                      &NestedParams::count>(params)),
            nullptr);
  EXPECT_EQ(optional_ops.nested_member(&params), nullptr);
  void* composed_value = optional_ops.mutable_nested_member(&params);
  ASSERT_NE(composed_value, nullptr);
  EXPECT_TRUE(params.composed.has_value());
  EXPECT_EQ(optional_ops.nested_member(&params), composed_value);

  const ParamFieldLocalOperations& direct_ops = composed_schema->fields[0].local_operations;
  ASSERT_NE(direct_ops.mutable_nested_member, nullptr);
  ASSERT_NE(direct_ops.nested_member, nullptr);
  ASSERT_NE(direct_ops.nested_schema, nullptr);
  EXPECT_EQ(direct_ops.nested_schema(), nested_schema);
  void* nested_value = direct_ops.mutable_nested_member(composed_value);
  ASSERT_NE(nested_value, nullptr);
  EXPECT_EQ(direct_ops.nested_member(composed_value), nested_value);

  const ParamFieldLocalOperations& count_ops = nested_schema->fields[0].local_operations;
  ASSERT_NE(count_ops.replace_member_from_scalar, nullptr);
  EXPECT_TRUE(count_ops.replace_member_from_scalar(nested_value, ParamScalarValue::from(3U)));
  // ASSERT/EXPECT control flow is not understood by the optional-access analyzer.
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(params.composed->nested.count, 3U);
  const auto* nested_count = detail::param_path_value<&OptionalComposedParams::composed,
                                                      &ComposedParams::nested,
                                                      &NestedParams::count>(params);
  ASSERT_NE(nested_count, nullptr);
  EXPECT_EQ(*nested_count, 3U);
  EXPECT_TRUE(validate_params(params).ok());

  const ParamSchemaDescriptor* scalar_schema = param_schema_descriptor<CompleteParams>();
  ASSERT_NE(scalar_schema, nullptr);
  EXPECT_EQ(scalar_schema->fields[0].local_operations.mutable_nested_member, nullptr);
  EXPECT_EQ(scalar_schema->fields[0].local_operations.nested_member, nullptr);
  EXPECT_EQ(scalar_schema->fields[0].local_operations.nested_schema, nullptr);
}

// REQ: PARAM-057
TEST(ParameterSchema, LocalAssignmentOperationsPreserveExactOptionalAndSequenceTypes) {
  const ParamSchemaDescriptor* optional_schema = param_schema_descriptor<OptionalParams>();
  const ParamSchemaDescriptor* enum_schema = param_schema_descriptor<NamedEnumParams>();
  ASSERT_NE(optional_schema, nullptr);
  ASSERT_NE(enum_schema, nullptr);
  ASSERT_EQ(optional_schema->fields.size(), 3U);
  ASSERT_EQ(enum_schema->fields.size(), 3U);

  OptionalParams optional;
  const ParamFieldLocalOperations& count = optional_schema->fields[0].local_operations;
  ASSERT_NE(count.replace_member_from_scalar, nullptr);
  ASSERT_NE(count.read_member_as_scalar, nullptr);
  ASSERT_NE(count.clear_optional_member, nullptr);
  EXPECT_TRUE(count.replace_member_from_scalar(&optional, ParamScalarValue::from(4)));
  ASSERT_EQ(optional.count, 4);
  ParamScalarValue observed_count;
  EXPECT_TRUE(count.read_member_as_scalar(&optional, observed_count));
  std::int32_t decoded_count{};
  EXPECT_TRUE(observed_count.try_as(decoded_count));
  EXPECT_EQ(decoded_count, 4);
  count.clear_optional_member(&optional);
  EXPECT_FALSE(optional.count.has_value());
  EXPECT_FALSE(count.read_member_as_scalar(&optional, observed_count));

  const ParamFieldLocalOperations& label = optional_schema->fields[1].local_operations;
  ASSERT_NE(label.replace_member_from_string, nullptr);
  ASSERT_NE(label.clear_optional_member, nullptr);
  label.replace_member_from_string(&optional, "camera");
  ASSERT_EQ(optional.label, "camera");
  label.clear_optional_member(&optional);
  EXPECT_FALSE(optional.label.has_value());

  const ParamFieldLocalOperations& tags = optional_schema->fields[2].local_operations;
  ASSERT_NE(tags.replace_member_from_string_sequence, nullptr);
  ASSERT_NE(tags.clear_optional_member, nullptr);
  const std::array<std::string_view, 2> tag_values{"rgb", "left"};
  tags.replace_member_from_string_sequence(&optional, tag_values);
  ASSERT_TRUE(optional.tags.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(*optional.tags, (std::vector<std::string>{"rgb", "left"}));
  tags.clear_optional_member(&optional);
  EXPECT_FALSE(optional.tags.has_value());

  NamedEnumParams enums;
  const ParamFieldLocalOperations& mode = enum_schema->fields[0].local_operations;
  const ParamFieldLocalOperations& fallback = enum_schema->fields[1].local_operations;
  const ParamFieldLocalOperations& modes = enum_schema->fields[2].local_operations;
  EXPECT_EQ(mode.clear_optional_member, nullptr);
  ASSERT_NE(mode.replace_member_from_scalar, nullptr);
  ASSERT_NE(fallback.replace_member_from_scalar, nullptr);
  ASSERT_NE(fallback.clear_optional_member, nullptr);
  ASSERT_NE(modes.replace_member_from_scalar_sequence, nullptr);
  EXPECT_TRUE(mode.replace_member_from_scalar(&enums, ParamScalarValue::from(NamedMode::kManual)));
  EXPECT_TRUE(
      fallback.replace_member_from_scalar(&enums, ParamScalarValue::from(NamedMode::kDisabled)));
  const std::array enum_values{ParamScalarValue::from(NamedMode::kAutomatic),
                               ParamScalarValue::from(NamedMode::kManual)};
  EXPECT_TRUE(modes.replace_member_from_scalar_sequence(&enums, enum_values));
  EXPECT_EQ(enums.mode, NamedMode::kManual);
  ASSERT_EQ(enums.fallback, NamedMode::kDisabled);
  EXPECT_EQ(enums.modes, (std::vector<NamedMode>{NamedMode::kAutomatic, NamedMode::kManual}));
  fallback.clear_optional_member(&enums);
  EXPECT_FALSE(enums.fallback.has_value());

  const auto prior_modes = enums.modes;
  const std::array wrong_kind{ParamScalarValue::from(std::int32_t{2})};
  EXPECT_FALSE(modes.replace_member_from_scalar_sequence(&enums, wrong_kind));
  EXPECT_EQ(enums.modes, prior_modes);
}

// REQ: PARAM-057
TEST(ParameterSchema, LocalSequenceAssignmentPreservesStatefulAllocators) {
  const ParamSchemaDescriptor* schema = param_schema_descriptor<StatefulAllocatorParams>();
  ASSERT_NE(schema, nullptr);
  ASSERT_EQ(schema->fields.size(), 5U);

  StatefulAllocatorParams params;
  params.scalars.push_back(41);
  params.strings.emplace_back("prior");

  const ParamFieldLocalOperations& scalars = schema->fields[0].local_operations;
  ASSERT_NE(scalars.replace_member_from_scalar_sequence, nullptr);
  const std::array scalar_values{ParamScalarValue::from(std::int32_t{2}),
                                 ParamScalarValue::from(std::int32_t{5})};
  EXPECT_TRUE(scalars.replace_member_from_scalar_sequence(&params, scalar_values));
  ASSERT_EQ(params.scalars.size(), 2U);
  EXPECT_EQ(params.scalars[0], 2);
  EXPECT_EQ(params.scalars[1], 5);
  EXPECT_EQ(params.scalars.get_allocator().identity(), 17U);

  const StatefulScalarSequence prior_scalars = params.scalars;
  const std::array wrong_kind{ParamScalarValue::from(std::uint32_t{8U})};
  EXPECT_FALSE(scalars.replace_member_from_scalar_sequence(&params, wrong_kind));
  EXPECT_EQ(params.scalars, prior_scalars);
  EXPECT_EQ(params.scalars.get_allocator().identity(), 17U);

  const ParamFieldLocalOperations& strings = schema->fields[1].local_operations;
  ASSERT_NE(strings.replace_member_from_string_sequence, nullptr);
  const std::array<std::string_view, 2> string_values{"left", "rgb"};
  strings.replace_member_from_string_sequence(&params, string_values);
  ASSERT_EQ(params.strings.size(), 2U);
  EXPECT_EQ(params.strings[0], "left");
  EXPECT_EQ(params.strings[1], "rgb");
  EXPECT_EQ(params.strings.get_allocator().identity(), 23U);

  const ParamFieldLocalOperations& optional_scalars = schema->fields[2].local_operations;
  const ParamFieldLocalOperations& optional_strings = schema->fields[3].local_operations;
  EXPECT_EQ(optional_scalars.replace_member_from_scalar_sequence, nullptr);
  EXPECT_EQ(optional_strings.replace_member_from_string_sequence, nullptr);
  EXPECT_NE(optional_scalars.clear_optional_member, nullptr);
  EXPECT_NE(optional_strings.clear_optional_member, nullptr);

  const ParamFieldLocalOperations& defaultable_optional_scalars =
      schema->fields[4].local_operations;
  ASSERT_NE(defaultable_optional_scalars.replace_member_from_scalar_sequence, nullptr);
  ASSERT_NE(defaultable_optional_scalars.clear_optional_member, nullptr);
  EXPECT_TRUE(
      defaultable_optional_scalars.replace_member_from_scalar_sequence(&params, scalar_values));
  ASSERT_TRUE(params.defaultable_optional_scalars.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(params.defaultable_optional_scalars->get_allocator().identity(), 29U);
  defaultable_optional_scalars.clear_optional_member(&params);
  EXPECT_FALSE(params.defaultable_optional_scalars.has_value());
  EXPECT_TRUE(
      defaultable_optional_scalars.replace_member_from_scalar_sequence(&params, scalar_values));
  ASSERT_TRUE(params.defaultable_optional_scalars.has_value());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  EXPECT_EQ(params.defaultable_optional_scalars->get_allocator().identity(), 0U);
}

// REQ: PARAM-003
TEST(ParameterSchema, IndependentFieldPoliciesReachLocalAndPortableViews) {
  const ParamSchemaDescriptor* descriptor = param_schema_descriptor<PolicyParams>();
  const ParamSchemaDocument* document = param_schema_document<PolicyParams>();
  ASSERT_NE(descriptor, nullptr);
  ASSERT_NE(document, nullptr);
  ASSERT_EQ(descriptor->fields.size(), 4U);
  ASSERT_EQ(document->fields.size(), descriptor->fields.size());

  const ParamFieldDescriptor& required = descriptor->fields[0];
  EXPECT_TRUE(required.required_on_load);
  EXPECT_EQ(required.alias, "legacy_required");
  EXPECT_EQ(required.external_access, ParamExternalAccess::kNone);
  EXPECT_EQ(required.persistence, ParamPersistence::kPersisted);

  const ParamFieldDescriptor& readable = descriptor->fields[1];
  EXPECT_EQ(readable.external_access, ParamExternalAccess::kReadOnly);
  EXPECT_EQ(readable.visibility, ParamVisibility::kPublic);
  EXPECT_EQ(readable.sensitivity, ParamSensitivity::kNormal);

  const ParamFieldDescriptor& secret = descriptor->fields[2];
  EXPECT_EQ(secret.sensitivity, ParamSensitivity::kSensitive);
  EXPECT_EQ(secret.external_access, ParamExternalAccess::kNone);
  EXPECT_EQ(secret.persistence, ParamPersistence::kPersisted);

  const ParamFieldDescriptor& scratch = descriptor->fields[3];
  EXPECT_EQ(scratch.persistence, ParamPersistence::kTransient);
  EXPECT_EQ(scratch.visibility, ParamVisibility::kPublic);
  EXPECT_EQ(scratch.sensitivity, ParamSensitivity::kNormal);

  EXPECT_TRUE(document->fields[0].required_on_load);
  EXPECT_EQ(document->fields[0].alias, required.alias);
  EXPECT_EQ(document->fields[1].external_access, readable.external_access);
  EXPECT_EQ(document->fields[2].sensitivity, secret.sensitivity);
  EXPECT_EQ(document->fields[3].persistence, scratch.persistence);
}

// REQ: PARAM-003
TEST(ParameterSchema, ExplicitFieldPolicyEvidencePreservesSemanticContract) {
  const ParamSchemaDescriptor* inferred = param_schema_descriptor<InferredPolicyEvidenceParams>();
  const ParamSchemaDescriptor* explicit_policy =
      param_schema_descriptor<ExplicitPolicyEvidenceParams>();
  const ParamSchemaDocument* explicit_document =
      param_schema_document<ExplicitPolicyEvidenceParams>();
  ASSERT_NE(inferred, nullptr);
  ASSERT_NE(explicit_policy, nullptr);
  ASSERT_NE(explicit_document, nullptr);
  ASSERT_EQ(inferred->fields.size(), 1U);
  ASSERT_EQ(explicit_policy->fields.size(), 1U);
  ASSERT_EQ(explicit_document->fields.size(), 1U);

  EXPECT_FALSE(inferred->fields.front().policy_evidence.complete());
  EXPECT_TRUE(explicit_policy->fields.front().policy_evidence.complete());
  EXPECT_EQ(explicit_document->fields.front().policy_evidence,
            explicit_policy->fields.front().policy_evidence);
  EXPECT_EQ(inferred->contract_hash, explicit_policy->contract_hash);
  EXPECT_NE(inferred->metadata_hash, explicit_policy->metadata_hash);
}

// REQ: PARAM-015
TEST(ParameterSchema, LastMutabilityMarkerClearsLifecycleOnlyState) {
  const ParamSchemaDescriptor* transitioned = param_schema_descriptor<MutabilityTransitionParams>();
  const ParamSchemaDescriptor* canonical = param_schema_descriptor<CanonicalMutabilityParams>();
  const ParamSchemaDocument* transitioned_document =
      param_schema_document<MutabilityTransitionParams>();
  const ParamSchemaDocument* canonical_document =
      param_schema_document<CanonicalMutabilityParams>();
  ASSERT_NE(transitioned, nullptr);
  ASSERT_NE(canonical, nullptr);
  ASSERT_NE(transitioned_document, nullptr);
  ASSERT_NE(canonical_document, nullptr);
  ASSERT_EQ(transitioned->fields.size(), 4U);

  EXPECT_EQ(transitioned->fields[0].mutability, ParamMutability::kCompileTime);
  EXPECT_EQ(transitioned->fields[1].mutability, ParamMutability::kRuntime);
  EXPECT_EQ(transitioned->fields[2].mutability, ParamMutability::kRuntime);
  EXPECT_EQ(transitioned->fields[3].mutability, ParamMutability::kRecompile);
  for (std::size_t index = 0; index < transitioned->fields.size(); ++index) {
    EXPECT_EQ(transitioned->fields[index].earliest_invalidated_stage, LifecycleStage::kConfigure);
    EXPECT_EQ(transitioned_document->fields[index].mutability,
              canonical_document->fields[index].mutability);
    EXPECT_EQ(transitioned_document->fields[index].earliest_invalidated_stage,
              canonical_document->fields[index].earliest_invalidated_stage);
  }
  EXPECT_EQ(transitioned->contract_hash, canonical->contract_hash);
  EXPECT_EQ(transitioned->metadata_hash, canonical->metadata_hash);
}

// REQ: PARAM-003
TEST(ParameterSchema, EveryForwardLifecycleStageIsAccepted) {
  const ParamSchemaDescriptor* descriptor = param_schema_descriptor<LifecycleStageParams>();
  ASSERT_NE(descriptor, nullptr);
  ASSERT_EQ(descriptor->fields.size(), 6U);
  EXPECT_EQ(descriptor->fields[0].earliest_invalidated_stage, LifecycleStage::kConfigure);
  EXPECT_EQ(descriptor->fields[1].earliest_invalidated_stage, LifecycleStage::kDiscover);
  EXPECT_EQ(descriptor->fields[2].earliest_invalidated_stage, LifecycleStage::kAllocate);
  EXPECT_EQ(descriptor->fields[3].earliest_invalidated_stage, LifecycleStage::kWarmUp);
  EXPECT_EQ(descriptor->fields[4].earliest_invalidated_stage, LifecycleStage::kArm);
  EXPECT_EQ(descriptor->fields[5].earliest_invalidated_stage, LifecycleStage::kStart);
}

// REQ: PARAM-003
TEST(ParameterSchema, EnumLogicalTypesFollowUnderlyingSignedness) {
  const ParamSchemaDescriptor* descriptor = param_schema_descriptor<EnumParams>();
  ASSERT_NE(descriptor, nullptr);
  ASSERT_EQ(descriptor->fields.size(), 2U);
  EXPECT_EQ(descriptor->fields[0].logical_type, ParamLogicalType::kSignedInteger);
  EXPECT_EQ(descriptor->fields[1].logical_type, ParamLogicalType::kUnsignedInteger);
}

// REQ: PARAM-055
TEST(ParameterSchema, NamedEnumChoicesPreserveAuthoredLabelsAndValidateEveryShape) {
  const ParamSchemaDescriptor* schema = param_schema_descriptor<NamedEnumParams>();
  const ParamSchemaDocument* document = param_schema_document<NamedEnumParams>();
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(document, nullptr);
  ASSERT_EQ(schema->fields.size(), 3U);
  ASSERT_EQ(document->fields.size(), 3U);

  for (std::size_t field_index = 0; field_index < schema->fields.size(); ++field_index) {
    const ParamFieldDescriptor& field = schema->fields[field_index];
    const ParamFieldDocument& portable = document->fields[field_index];
    ASSERT_EQ(field.enum_choices.count, 3U);
    ASSERT_EQ(portable.enum_choices.count, 3U);
    EXPECT_EQ(field.constraints.allowed_value_count, field.enum_choices.count);
    EXPECT_EQ(field.enum_choices.values[0].name, "disabled");
    EXPECT_EQ(field.enum_choices.values[1].name, "automatic");
    EXPECT_EQ(field.enum_choices.values[2].name, "manual");
    EXPECT_EQ(portable.enum_choices.values[1].name, "automatic");
    NamedMode decoded{};
    EXPECT_TRUE(field.enum_choices.values[1].value.try_as(decoded));
    EXPECT_EQ(decoded, NamedMode::kAutomatic);
  }

  NamedEnumParams params;
  EXPECT_TRUE(validate_params(params).ok());

  params.mode = static_cast<NamedMode>(1U);
  ParamValidationResult result = validate_params(params);
  EXPECT_EQ(result.code, ParamValidationCode::kNotAllowed);
  EXPECT_EQ(result.field_name, "mode");
  params.mode = NamedMode::kAutomatic;

  params.fallback = static_cast<NamedMode>(1U);
  result = validate_params(params);
  EXPECT_EQ(result.code, ParamValidationCode::kNotAllowed);
  EXPECT_EQ(result.field_name, "fallback");
  params.fallback.reset();
  EXPECT_TRUE(validate_params(params).ok());

  params.modes.push_back(static_cast<NamedMode>(1U));
  result = validate_params(params);
  EXPECT_EQ(result.code, ParamValidationCode::kNotAllowed);
  EXPECT_EQ(result.field_name, "modes");

  const ParamSchemaDescriptor* first_shape = param_schema_descriptor<NamedEnumShapeA>();
  const ParamSchemaDescriptor* renamed_shape = param_schema_descriptor<NamedEnumShapeB>();
  ASSERT_NE(first_shape, nullptr);
  ASSERT_NE(renamed_shape, nullptr);
  EXPECT_NE(first_shape->structural_hash, renamed_shape->structural_hash);
}

// REQ: PARAM-055
TEST(ParameterSchema, NamedEnumChoicesRemainOptInForExistingSchemas) {
  const ParamSchemaDescriptor* schema = param_schema_descriptor<UnnamedEnumParams>();
  const ParamSchemaDocument* document = param_schema_document<UnnamedEnumParams>();
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(document, nullptr);
  ASSERT_EQ(schema->fields.size(), 1U);
  ASSERT_EQ(document->fields.size(), 1U);

  const ParamFieldDescriptor& field = schema->fields.front();
  const ParamFieldDocument& portable = document->fields.front();
  EXPECT_EQ(field.enum_choices.count, 0U);
  EXPECT_EQ(portable.enum_choices.count, 0U);
  EXPECT_EQ(field.constraints.allowed_value_count, 2U);
  EXPECT_EQ(portable.constraints.allowed_value_count, 2U);

  UnnamedEnumParams params;
  EXPECT_TRUE(validate_params(params).ok());
  params.mode = NamedMode::kManual;
  const ParamValidationResult result = validate_params(params);
  EXPECT_EQ(result.code, ParamValidationCode::kNotAllowed);
  EXPECT_EQ(result.field_name, "mode");
}

// REQ: PARAM-004
TEST(ParameterSchema, StructuralContractAndMetadataHashesRemainSeparate) {
  const ParamSchemaDescriptor* asserted = param_schema_descriptor<ProofParamsA>();
  const ParamSchemaDescriptor* proven = param_schema_descriptor<ProofParamsB>();
  ASSERT_NE(asserted, nullptr);
  ASSERT_NE(proven, nullptr);
  EXPECT_EQ(asserted->proof, ParamSchemaProof::kAuthorAsserted);
  EXPECT_EQ(proven->proof, ParamSchemaProof::kProven);
  EXPECT_EQ(asserted->structural_hash, proven->structural_hash);
  EXPECT_EQ(asserted->contract_hash, proven->contract_hash);
  EXPECT_EQ(asserted->metadata_hash, proven->metadata_hash);
}

// REQ: PARAM-004
TEST(ParameterSchema, PortableTypeShapeSeparatesDecodingCompatibility) {
  const auto* scalar32 = param_schema_descriptor<ShapeInt32Params>();
  const auto* scalar64 = param_schema_descriptor<ShapeInt64Params>();
  const auto* optional_scalar32 = param_schema_descriptor<ShapeOptionalInt32Params>();
  const auto* vector_int = param_schema_descriptor<ShapeVectorIntParams>();
  const auto* vector_float = param_schema_descriptor<ShapeVectorFloatParams>();
  const auto* vector_duration = param_schema_descriptor<ShapeVectorDurationParams>();
  const auto* milliseconds = param_schema_descriptor<ShapeMillisecondsParams>();
  const auto* microseconds = param_schema_descriptor<ShapeMicrosecondsParams>();
  const auto* bound8 = param_schema_descriptor<ShapeBound8Params>();
  const auto* bound16 = param_schema_descriptor<ShapeBound16Params>();
  const auto* partial_nested = param_schema_descriptor<PartialNestedParent>();
  const auto* nested_parent32 = param_schema_descriptor<NestedParent32>();
  const auto* nested_parent64 = param_schema_descriptor<NestedParent64>();
  ASSERT_NE(scalar32, nullptr);
  ASSERT_NE(scalar64, nullptr);
  ASSERT_NE(optional_scalar32, nullptr);
  ASSERT_NE(vector_int, nullptr);
  ASSERT_NE(vector_float, nullptr);
  ASSERT_NE(vector_duration, nullptr);
  ASSERT_NE(milliseconds, nullptr);
  ASSERT_NE(microseconds, nullptr);
  ASSERT_NE(bound8, nullptr);
  ASSERT_NE(bound16, nullptr);
  ASSERT_NE(partial_nested, nullptr);
  ASSERT_NE(nested_parent32, nullptr);
  ASSERT_NE(nested_parent64, nullptr);

  EXPECT_EQ(scalar32->fields[0].type_shape.type, ParamLogicalType::kSignedInteger);
  EXPECT_EQ(scalar32->fields[0].type_shape.bit_width, 32U);
  EXPECT_EQ(scalar64->fields[0].type_shape.bit_width, 64U);
  EXPECT_TRUE(scalar32->fields[0].type_shape.has_stable_shape);
  EXPECT_FALSE(scalar32->fields[0].type_shape.optional);
  EXPECT_TRUE(optional_scalar32->fields[0].type_shape.optional);
  EXPECT_EQ(optional_scalar32->fields[0].logical_type, ParamLogicalType::kSignedInteger);
  EXPECT_EQ(optional_scalar32->fields[0].type_shape.bit_width, 32U);
  EXPECT_NE(scalar32->structural_hash, scalar64->structural_hash);
  EXPECT_NE(scalar32->structural_hash, optional_scalar32->structural_hash);

  EXPECT_EQ(vector_int->fields[0].type_shape.type, ParamLogicalType::kSequence);
  EXPECT_EQ(vector_int->fields[0].type_shape.element_or_rep_type, ParamLogicalType::kSignedInteger);
  EXPECT_EQ(vector_int->fields[0].type_shape.element_or_rep_bit_width, 32U);
  EXPECT_EQ(vector_float->fields[0].type_shape.element_or_rep_type,
            ParamLogicalType::kFloatingPoint);
  EXPECT_NE(vector_int->structural_hash, vector_float->structural_hash);
  EXPECT_FALSE(vector_duration->fields[0].type_shape.has_stable_shape);

  EXPECT_EQ(milliseconds->fields[0].type_shape.type, ParamLogicalType::kDuration);
  EXPECT_EQ(milliseconds->fields[0].type_shape.duration_period_numerator, 1);
  EXPECT_EQ(milliseconds->fields[0].type_shape.duration_period_denominator, 1000);
  EXPECT_EQ(microseconds->fields[0].type_shape.duration_period_denominator, 1000000);
  EXPECT_NE(milliseconds->structural_hash, microseconds->structural_hash);

  EXPECT_NE(bound8->structural_hash, bound16->structural_hash);
  EXPECT_FALSE(partial_nested->fields[0].type_shape.has_stable_shape);
  EXPECT_TRUE(nested_parent32->fields[0].type_shape.has_stable_shape);
  EXPECT_TRUE(nested_parent64->fields[0].type_shape.has_stable_shape);
  EXPECT_NE(nested_parent32->structural_hash, nested_parent64->structural_hash);
  const auto* restricted = param_schema_descriptor<RestrictedParams>();
  const auto* nested = param_schema_descriptor<ComposedParams>();
  ASSERT_NE(restricted, nullptr);
  ASSERT_NE(nested, nullptr);
  EXPECT_FALSE(restricted->fields[0].type_shape.has_stable_shape);
  EXPECT_TRUE(nested->fields[0].type_shape.has_stable_shape);
}

// REQ: PARAM-054
TEST(ParameterSchema, OptionalFieldsPreserveAbsenceAndValidatePresentValues) {
  const ParamSchemaDescriptor* schema = param_schema_descriptor<OptionalParams>();
  const ParamSchemaDocument* document = param_schema_document<OptionalParams>();
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(document, nullptr);
  ASSERT_EQ(schema->fields.size(), 3U);
  ASSERT_EQ(document->fields.size(), 3U);

  EXPECT_TRUE(schema->fields[0].type_shape.optional);
  EXPECT_EQ(schema->fields[0].logical_type, ParamLogicalType::kSignedInteger);
  EXPECT_EQ(schema->fields[0].type_shape.bit_width, 32U);
  EXPECT_TRUE(schema->fields[0].type_shape.has_stable_shape);
  EXPECT_TRUE(document->fields[0].type_shape.optional);
  EXPECT_TRUE(schema->fields[0].storage.fixed_object_extent);
  EXPECT_TRUE(schema->fields[0].storage.supports_bounded_materialization);
  EXPECT_EQ(schema->fields[0].storage.inline_bytes, sizeof(std::int32_t) + 1U);

  OptionalParams params;
  EXPECT_TRUE(validate_params(params).ok());

  params.count = 4;
  params.label = "camera";
  params.tags = std::vector<std::string>{"left", "rgb"};
  EXPECT_TRUE(validate_params(params).ok());

  params.count = 3;
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kNotAllowed);
  params.count.reset();
  EXPECT_TRUE(validate_params(params).ok());

  params.label = "too-long-label";
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kCardinalityExceeded);
  params.label.reset();

  params.tags = std::vector<std::string>{"one", "two", "three"};
  EXPECT_EQ(validate_params(params).code, ParamValidationCode::kCardinalityExceeded);
  params.tags.reset();
  EXPECT_TRUE(validate_params(params).ok());
}

// REQ: PARAM-005
TEST(ParameterSchema, AggregateValidationIsTypedAndLeavesTheInputUnchanged) {
  // Key setup: construct valid and invalid variants of multiple params types to exercise
  // every constraint category (range, duration, UTF-8, cardinality, custom validator, nested).
  const CompleteParams valid;
  const auto valid_result = validate_params(valid);
  EXPECT_TRUE(valid_result.ok());

  // Key assertion: each invalid variant produces the expected ValidationCode and field_name;
  // the input is not modified by a failed validation.
  CompleteParams invalid_gain = valid;
  invalid_gain.gain = 8.0F;
  const auto gain_result = validate_params(invalid_gain);
  EXPECT_EQ(gain_result.code, ParamValidationCode::kOutOfRange);
  EXPECT_EQ(gain_result.field_name, "gain");
  EXPECT_EQ(invalid_gain.gain, 8.0F);
  EXPECT_EQ(param_type_descriptor<CompleteParams>().validate_aggregate(&invalid_gain).code,
            ParamValidationCode::kOutOfRange);

  CompleteParams invalid_label = valid;
  invalid_label.label = "this-label-is-longer-than-sixteen-bytes";
  const auto label_result = validate_params(invalid_label);
  EXPECT_EQ(label_result.code, ParamValidationCode::kCardinalityExceeded);
  EXPECT_EQ(label_result.field_name, "label");

  ValidationParams invalid_signed;
  invalid_signed.signed_value = -5;
  EXPECT_EQ(validate_params(invalid_signed).code, ParamValidationCode::kOutOfRange);

  ValidationParams invalid_duration;
  invalid_duration.timeout = std::chrono::milliseconds{11};
  EXPECT_EQ(validate_params(invalid_duration).code, ParamValidationCode::kOutOfRange);

  ValidationParams invalid_utf8;
  invalid_utf8.utf8 = std::string{"\xC0\xAF", 2};
  EXPECT_EQ(validate_params(invalid_utf8).code, ParamValidationCode::kInvalidEncoding);

  BinaryStringParams arbitrary_bytes;
  arbitrary_bytes.bytes = std::string{"\xC0\xAF", 2};
  EXPECT_TRUE(validate_params(arbitrary_bytes).ok());

  Utf8StringVectorParams invalid_element_encoding;
  invalid_element_encoding.values[0] = std::string{"\xC0\xAF", 2};
  EXPECT_EQ(validate_params(invalid_element_encoding).code, ParamValidationCode::kInvalidEncoding);

  Utf8StringVectorParams invalid_element_size;
  invalid_element_size.values[0] = "123456789";
  EXPECT_EQ(validate_params(invalid_element_size).code, ParamValidationCode::kCardinalityExceeded);

  BinaryStringVectorParams arbitrary_elements;
  arbitrary_elements.values.push_back(std::string{"\xC0\xAF", 2});
  EXPECT_TRUE(validate_params(arbitrary_elements).ok());

  ValidationParams invalid_custom;
  invalid_custom.even = 3;
  const ParamValidationResult custom_result = validate_params(invalid_custom);
  EXPECT_EQ(custom_result.code, ParamValidationCode::kValidatorRejected);
  EXPECT_EQ(custom_result.validator_id, "holoscan.test.validator.even.v1");

  ValidationParams invalid_nested;
  invalid_nested.nested.count = 5;
  const ParamValidationResult nested_result = validate_params(invalid_nested);
  EXPECT_EQ(nested_result.code, ParamValidationCode::kOutOfRange);
  EXPECT_EQ(nested_result.field_name, "count");

  // Encoded-byte limits are portable codec metadata. Aggregate validation cannot infer encoded
  // bytes from a container's element count; HSDK5-018's qualified codec enforces this bound.
  EXPECT_TRUE(validate_params(EncodedBoundParams{}).ok());
}

// REQ: PARAM-014
TEST(ParameterSchema, CompatibleDurationRangesConvertExactlyToFieldDomain) {
  const auto* schema = param_schema_descriptor<DurationRangeParams>();
  ASSERT_NE(schema, nullptr);
  ASSERT_EQ(schema->fields.size(), 2U);

  EXPECT_EQ(schema->fields[0].constraints.range_min.kind, ParamConstraintScalarKind::kSigned);
  EXPECT_EQ(schema->fields[0].constraints.range_min.signed_value, 100);
  EXPECT_EQ(schema->fields[0].constraints.range_max.signed_value, 2000);
  EXPECT_EQ(schema->fields[1].constraints.range_min.signed_value, 1000);
  EXPECT_EQ(schema->fields[1].constraints.range_max.signed_value, 2000);

  EXPECT_TRUE(validate_params(DurationRangeParams{}).ok());
  DurationRangeParams below_range;
  below_range.mixed_bounds = std::chrono::microseconds{99};
  EXPECT_EQ(validate_params(below_range).code, ParamValidationCode::kOutOfRange);
  DurationRangeParams above_range;
  above_range.coarser_bounds = std::chrono::microseconds{2001};
  EXPECT_EQ(validate_params(above_range).code, ParamValidationCode::kOutOfRange);
}

// REQ: PARAM-005
TEST(ParameterSchema, AllowedValuesAndCardinalityPrecedeEncodingDeterministically) {
  AllowedValueParams allowed;
  EXPECT_TRUE(validate_params(allowed).ok());
  allowed.mode = 4;
  EXPECT_TRUE(validate_params(allowed).ok());
  allowed.mode = 3;
  const ParamValidationResult disallowed = validate_params(allowed);
  EXPECT_EQ(disallowed.code, ParamValidationCode::kNotAllowed);
  EXPECT_EQ(disallowed.field_name, "mode");

  const ParamSchemaDescriptor* allowed_descriptor = param_schema_descriptor<AllowedValueParams>();
  ASSERT_NE(allowed_descriptor, nullptr);
  ASSERT_EQ(allowed_descriptor->fields[0].constraints.allowed_value_count, 3U);
  EXPECT_EQ(allowed_descriptor->fields[0].constraints.allowed_values[0].signed_value, 1);
  EXPECT_EQ(allowed_descriptor->fields[0].constraints.allowed_values[1].signed_value, 2);
  EXPECT_EQ(allowed_descriptor->fields[0].constraints.allowed_values[2].signed_value, 4);

  Utf8StringVectorParams too_many_elements;
  too_many_elements.values.assign(5, std::string{"\xC0\xAF", 2});
  EXPECT_EQ(validate_params(too_many_elements).code, ParamValidationCode::kCardinalityExceeded);

  Utf8StringVectorParams oversized_invalid_element;
  oversized_invalid_element.values[0] = std::string(9, 'x');
  oversized_invalid_element.values[0][0] = static_cast<char>(0xC0);
  oversized_invalid_element.values[0][1] = static_cast<char>(0xAF);
  EXPECT_EQ(validate_params(oversized_invalid_element).code,
            ParamValidationCode::kCardinalityExceeded);

  ValidationParams oversized_invalid_string;
  oversized_invalid_string.utf8 = std::string(17, 'x');
  oversized_invalid_string.utf8[0] = static_cast<char>(0xC0);
  oversized_invalid_string.utf8[1] = static_cast<char>(0xAF);
  EXPECT_EQ(validate_params(oversized_invalid_string).code,
            ParamValidationCode::kCardinalityExceeded);
}

// REQ: PARAM-006
TEST(ParameterStorage, DynamicOwnershipAndBoundsRemainDifferentFacts) {
  const auto& opaque = param_type_descriptor<OpaqueParams>();
  const auto& complete = param_type_descriptor<CompleteParams>();
  EXPECT_TRUE(opaque.storage.owns_dynamic_storage);
  EXPECT_FALSE(opaque.storage.fixed_object_extent);
  EXPECT_TRUE(complete.storage.owns_dynamic_storage);
  EXPECT_FALSE(complete.storage.supports_bounded_materialization);

  const auto& pointer = param_type_descriptor<PointerParams>();
  EXPECT_TRUE(pointer.storage.owns_dynamic_storage);
  EXPECT_FALSE(pointer.storage.fixed_object_extent);
  EXPECT_FALSE(pointer.storage.supports_bounded_materialization);
  EXPECT_EQ(pointer.storage.inline_bytes, 0U);

  const auto* schema_descriptor = param_schema_descriptor<CompleteParams>();
  ASSERT_NE(schema_descriptor, nullptr);
  EXPECT_EQ(schema_descriptor->fields[0].constraints.max_utf8_bytes, 16);
  EXPECT_FALSE(schema_descriptor->fields[0].storage.supports_bounded_materialization);
}

// REQ: PARAM-004
// REQ: PARAM-006
TEST(ParameterStorage, FilesystemPathsRemainLegalButPortablyUnstable) {
  const ParamSchemaDescriptor* path = param_schema_descriptor<PathParams>();
  const ParamSchemaDescriptor* path_vector = param_schema_descriptor<PathVectorParams>();
  ASSERT_NE(path, nullptr);
  ASSERT_NE(path_vector, nullptr);

  EXPECT_EQ(path->fields[0].logical_type, ParamLogicalType::kFilesystemPath);
  EXPECT_EQ(path->fields[0].type_shape.type, ParamLogicalType::kFilesystemPath);
  EXPECT_FALSE(path->fields[0].type_shape.has_stable_shape);
  EXPECT_TRUE(path->fields[0].storage.owns_dynamic_storage);
  EXPECT_FALSE(path->fields[0].storage.fixed_object_extent);
  EXPECT_FALSE(path->fields[0].storage.has_portable_codec);

  EXPECT_EQ(path_vector->fields[0].type_shape.type, ParamLogicalType::kSequence);
  EXPECT_EQ(path_vector->fields[0].type_shape.element_or_rep_type,
            ParamLogicalType::kFilesystemPath);
  EXPECT_FALSE(path_vector->fields[0].type_shape.has_stable_shape);
  EXPECT_TRUE(validate_params(PathParams{}).ok());
  EXPECT_TRUE(validate_params(PathVectorParams{}).ok());
}

// REQ: PARAM-007
TEST(ParameterBinding, WholeAggregateBindingValidatesBeforeReplacement) {
  // Key setup: establish a valid baseline; subsequent bind_initial calls must verify that
  // rejection leaves the incumbent params value unchanged.
  CompleteParamOperator op;
  const CompleteParams accepted{.label = "front", .gain = 2.0F, .workspace_mb = 1024};
  ASSERT_TRUE(detail::OperatorParameterBinding::bind_initial(op, accepted));
  EXPECT_EQ(op.params().label, "front");
  EXPECT_EQ(op.params().gain, 2.0F);
  EXPECT_EQ(op.params().workspace_mb, 1024);

  // Key assertion: a rejected aggregate (gain out of range) must not modify the incumbent;
  // RestrictedParams binding must return kNotSupported (no staged replacement available).
  CompleteParams rejected = accepted;
  rejected.gain = 9.0F;
  const auto result = detail::OperatorParameterBinding::bind_initial(op, rejected);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(op.params().gain, 2.0F);

  RestrictedParamOperator restricted_op;
  const RestrictedParams unsupported{.value = RestrictedField{42}};
  const auto unsupported_result =
      detail::OperatorParameterBinding::bind_initial(restricted_op, unsupported);
  ASSERT_FALSE(unsupported_result);
  EXPECT_EQ(unsupported_result.error().code, ErrorCode::kNotSupported);
  EXPECT_EQ(restricted_op.params().value.value, 0);
}

// REQ: PARAM-008
TEST(ParameterSchema, PortableProjectionContainsNoLocalOperations) {
  const ParamSchemaDocument* document = param_schema_document<CompleteParams>();
  const ParamSchemaDescriptor* descriptor = param_schema_descriptor<CompleteParams>();
  ASSERT_NE(document, nullptr);
  ASSERT_NE(descriptor, nullptr);
  EXPECT_EQ(document->stable_id, descriptor->stable_id);
  EXPECT_EQ(document->version, descriptor->version);
  EXPECT_EQ(document->fields.size(), descriptor->fields.size());
  EXPECT_EQ(document->fields[1].canonical_name, "gain");
  EXPECT_EQ(document->fields[1].id, descriptor->fields[1].id);
}

// REQ: PARAM-011
TEST(ParameterSchema, ProjectionClassificationUsesSafeMetadataOnlyPrecedence) {
  // Key setup: copy a field document and mutate its policy attributes to step through
  // each projection status in strict precedence order.
  const ParamSchemaDocument* document = param_schema_document<CompleteParams>();
  ASSERT_NE(document, nullptr);

  ParamFieldDocument field = document->fields[0];
  // Key assertion: transient > sensitive > internal > profile-excluded > codec-excluded > included;
  // each status is reached by toggling exactly one policy field at a time.
  EXPECT_EQ(classify_param_projection(
                field, ParamProjectionContext{.profile_allows = true, .codec_available = true}),
            params::ParamProjectionStatus::kIncluded);

  field.persistence = ParamPersistence::kTransient;
  field.sensitivity = ParamSensitivity::kSensitive;
  field.visibility = ParamVisibility::kInternal;
  EXPECT_EQ(classify_param_projection(field, {}),
            params::ParamProjectionStatus::kExcludedTransient);

  field.persistence = ParamPersistence::kPersisted;
  EXPECT_EQ(classify_param_projection(field, {}),
            params::ParamProjectionStatus::kRedactedSensitive);

  field.sensitivity = ParamSensitivity::kNormal;
  EXPECT_EQ(classify_param_projection(field, {}), params::ParamProjectionStatus::kExcludedInternal);

  field.visibility = ParamVisibility::kPublic;
  EXPECT_EQ(classify_param_projection(field, {}),
            params::ParamProjectionStatus::kExcludedProfilePolicy);
  EXPECT_EQ(classify_param_projection(
                field, ParamProjectionContext{.profile_allows = false, .codec_available = false}),
            params::ParamProjectionStatus::kExcludedProfilePolicy);
  EXPECT_EQ(classify_param_projection(
                field, ParamProjectionContext{.profile_allows = true, .codec_available = false}),
            params::ParamProjectionStatus::kExcludedUnsupportedCodec);
}

// REQ: PARAM-012
TEST(ParameterSchema, LocalFieldOperationsSupportStrongQualifiedStaging) {
  // Key setup: obtain local operations for label (std::string) and gain (float) fields
  // to verify distinct type tokens, sizes, and the presence of all function pointers.
  const ParamSchemaDescriptor* descriptor = param_schema_descriptor<CompleteParams>();
  const ParamSchemaDescriptor* restricted_descriptor = param_schema_descriptor<RestrictedParams>();
  const ParamSchemaDescriptor* throwing_equality_descriptor =
      param_schema_descriptor<ThrowingEqualityParams>();
  ASSERT_NE(descriptor, nullptr);
  ASSERT_NE(restricted_descriptor, nullptr);
  ASSERT_NE(throwing_equality_descriptor, nullptr);

  const ParamFieldLocalOperations& label_operations = descriptor->fields[0].local_operations;
  const ParamFieldLocalOperations& gain_operations = descriptor->fields[1].local_operations;
  EXPECT_NE(label_operations.local_field_type_token, gain_operations.local_field_type_token);
  EXPECT_EQ(label_operations.object_size, sizeof(std::string));
  EXPECT_EQ(label_operations.object_alignment, alignof(std::string));
  ASSERT_NE(label_operations.copy_construct_from_member, nullptr);
  ASSERT_NE(label_operations.destroy_staged, nullptr);
  ASSERT_NE(label_operations.replace_member_from_staged, nullptr);

  // Key assertion: copy_construct_from_member writes the field value into aligned staging
  // storage; replace_member_from_staged swaps the staged value into the live params member.
  CompleteParams params{};
  alignas(std::string) std::array<std::byte, sizeof(std::string)> staging{};
  label_operations.copy_construct_from_member(staging.data(), &params);
  auto* staged = std::launder(reinterpret_cast<std::string*>(staging.data()));
  EXPECT_EQ(*staged, "camera");
  if (label_operations.semantic_equal_to_staged != nullptr) {
    EXPECT_TRUE(label_operations.semantic_equal_to_staged(&params, staged));
  }

  std::string replacement{"replacement"};
  label_operations.replace_member_from_staged(&params, &replacement);
  EXPECT_EQ(params.label, "replacement");
  EXPECT_EQ(replacement, "camera");
  label_operations.destroy_staged(staged);

  const ParamFieldLocalOperations& restricted_operations =
      restricted_descriptor->fields[0].local_operations;
  EXPECT_EQ(restricted_operations.replace_member_from_staged, nullptr);
  EXPECT_NE(restricted_operations.semantic_equal_to_staged, nullptr);

  static_assert(!detail::NothrowSemanticEquality<ThrowingEqualityField>);
  const ParamFieldLocalOperations& throwing_equality_operations =
      throwing_equality_descriptor->fields[0].local_operations;
  EXPECT_EQ(throwing_equality_operations.semantic_equal_to_staged, nullptr);
}

// REQ: PARAM-012
TEST(ParameterSchema, ComposedFieldsOnlyInstallValidSemanticEqualityThunks) {
  // Key setup: instantiate descriptors for pair, tuple, reference-qualified container, and
  // declaration-only standard-container equality fields whose element has no equality operator.
  const ParamSchemaDescriptor* descriptor = param_schema_descriptor<ComposedEqualityParams>();
  ASSERT_NE(descriptor, nullptr);
  ASSERT_EQ(descriptor->fields.size(), 20U);

  // Key assertion: descriptor construction succeeds and leaves every invalid optional equality
  // route absent instead of imposing an undeclared comparability requirement on the field.
  for (const ParamFieldDescriptor& field : descriptor->fields) {
    EXPECT_EQ(field.local_operations.semantic_equal_members, nullptr);
  }
}

}  // namespace
}  // namespace holoscan
