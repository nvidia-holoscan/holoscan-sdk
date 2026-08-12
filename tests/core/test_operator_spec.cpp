/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <utility>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/readiness_source.hpp>

#include <core/operator_spec_test_access.hpp>
#include <core/operator_test_binding.hpp>

namespace holoscan {

struct OperatorSpecSchemaPayload {
  std::uint32_t value{};
};

template <>
consteval SchemaIdentity schema_identity<OperatorSpecSchemaPayload>() {
  return SchemaIdentity{
      .canonical_name = "holoscan.test.OperatorSpecSchemaPayload",
      .version = 7,
      .schema_sha256 = {std::byte{0xA5}, std::byte{0x5A}},
      .id = SchemaId{.value = {std::byte{0x5A}}},
  };
}

namespace {

// Placeholder payload struct with a Tensor-adjacent field; reserved for future
// static-assert checks on schema detection traits.
struct CarrierWithTensorField {
  std::uint64_t frame_id{};
};

template <typename Spec, typename Port>
concept CanAssignInputBuilderToPort = requires(Spec& spec, Port& port) { port = spec.input(port); };

template <typename Spec, typename Port>
concept CanAssignOutputBuilderToPort =
    requires(Spec& spec, Port& port) { port = spec.output(port); };

template <typename Spec, typename Port>
concept HasInputArrayBinding = requires(Spec& spec, Port& port) { spec.input_array(port); };

template <typename Spec, typename Port>
concept HasOutputArrayBinding = requires(Spec& spec, Port& port) { spec.output_array(port); };

template <typename Builder>
concept HasTensorAllocation = requires(Builder& builder, TensorAllocationBounds bounds) {
  { builder.tensor_allocation(bounds) } -> std::same_as<Builder&>;
};

using ScalarInput = Input<std::uint32_t>;
using ScalarOutput = Output<std::uint32_t>;
using TensorOutput = Output<Tensor>;

static_assert(detail::operator_spec_payload_kind<std::uint32_t>() == PayloadKind::kInlineTyped);
static_assert(detail::operator_spec_payload_kind<OperatorSpecSchemaPayload>() ==
              PayloadKind::kSerializedBytes);
static_assert(detail::operator_spec_payload_kind<Tensor>() == PayloadKind::kDescriptorBacked);
static_assert(!detail::operator_spec_schema_identity<std::uint32_t>().has_value());
static_assert(!detail::operator_spec_schema_identity<Tensor>().has_value());
static_assert(detail::operator_spec_schema_identity<OperatorSpecSchemaPayload>()->version == 7);
static_assert(detail::operator_spec_schema_identity<OperatorSpecSchemaPayload>()->canonical_name ==
              "holoscan.test.OperatorSpecSchemaPayload");
static_assert(&detail::kOperatorSpecLocalTypeToken<std::uint32_t> !=
              &detail::kOperatorSpecLocalTypeToken<std::int32_t>);
static_assert(&detail::kOperatorSpecLocalTypeToken<OperatorSpecSchemaPayload> !=
              &detail::kOperatorSpecLocalTypeToken<Tensor>);

static_assert(
    std::same_as<decltype(std::declval<OperatorSpec&>().input(std::declval<ScalarInput&>())),
                 InputSpecBuilder&>);
static_assert(
    std::same_as<decltype(std::declval<OperatorSpec&>().output(std::declval<ScalarOutput&>())),
                 OutputSpecBuilder&>);
static_assert(
    std::same_as<decltype(std::declval<OperatorSpec&>().output(std::declval<TensorOutput&>())),
                 TensorOutputSpecBuilder&>);
static_assert(
    std::same_as<decltype(std::declval<OperatorSpec&>().compute_capture_safe()), OperatorSpec&>);

static_assert(!CanAssignInputBuilderToPort<OperatorSpec, ScalarInput>);
static_assert(!CanAssignOutputBuilderToPort<OperatorSpec, ScalarOutput>);
static_assert(!HasInputArrayBinding<OperatorSpec, ScalarInput>);
static_assert(!HasOutputArrayBinding<OperatorSpec, ScalarOutput>);

static_assert(!HasTensorAllocation<OutputSpecBuilder>);
static_assert(HasTensorAllocation<TensorOutputSpecBuilder>);
static_assert(std::same_as<decltype(std::declval<TensorOutputSpecBuilder&>()
                                        .max_emits_per_compute(2)
                                        .tensor_allocation(TensorAllocationBounds{
                                            .max_rank = 4,
                                            .max_byte_span = 4096,
                                        })
                                        .late_joiner_history_depth(0)),
                           TensorOutputSpecBuilder&>);

// Operator that registers the same Input<> field under two different names.
// Used by RejectsOnePortDeclarationBoundUnderDifferentNames to verify that a single
// port object cannot be aliased to multiple spec entries (kInvalidArgument expected).
class ReusedPortDeclarationOperator final : public Operator<> {
 public:
  Input<std::uint32_t> input_;

  void setup(OperatorSpec& spec) override {
    spec.input(input_, "first");
    spec.input(input_, "second");
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator that passes an empty string as the readiness source name.
// Used by RejectsEmptyReadinessSourceName to verify kInvalidArgument is returned
// before any registration takes place.
class EmptyReadinessNameOperator final : public Operator<> {
 public:
  ReadableSource readable_;

  void setup(OperatorSpec& spec) override { spec.readable_source(readable_, ""); }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator that registers the same NotificationSource under two distinct names.
// Used by RejectsOneReadinessDeclarationBoundUnderDifferentNames.
class ReusedReadinessDeclarationOperator final : public Operator<> {
 public:
  NotificationSource notification_;

  void setup(OperatorSpec& spec) override {
    spec.notification_source(notification_, "first");
    spec.notification_source(notification_, "second");
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator that registers two different readiness sources with the same name "ready".
// Used by RejectsDuplicateNamesAcrossReadinessSourceKinds (kDuplicateName expected).
class DuplicateReadinessNameOperator final : public Operator<> {
 public:
  ReadableSource readable_;
  DoorbellSource doorbell_;

  void setup(OperatorSpec& spec) override {
    spec.readable_source(readable_, "ready");
    spec.doorbell_source(doorbell_, "ready");
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator that gives an input port and a readiness source the same name.
// Used by RejectsDuplicateNamesAcrossPortAndReadinessKinds (kDuplicateName expected).
class DuplicatePortAndReadinessNameOperator final : public Operator<> {
 public:
  Input<std::uint32_t> input_{"ready"};
  ReadableSource readable_;

  void setup(OperatorSpec& spec) override {
    spec.input(input_);
    spec.readable_source(readable_, "ready");
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator with a clean, non-conflicting set of declarations (one input, one output,
// one readable source, one spin source).  Used by FreshSpecsAcceptTheSameStableDeclarations
// to verify that the same operator can be declared into multiple fresh specs without error.
class ValidDeclarationsOperator final : public Operator<> {
 public:
  Input<std::uint32_t> input_{"input"};
  Output<std::uint32_t> output_{"output"};
  ReadableSource readable_;
  SpinSource spin_;

  void setup(OperatorSpec& spec) override {
    spec.input(input_);
    spec.output(output_);
    spec.readable_source(readable_, "readable");
    spec.spin_source(spin_, "spin");
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator with initially unnamed ports whose stable names are assigned through setup().
// Used by SetupNamesBecomeTheStableNamesOfInitiallyUnnamedPortDeclarations to confirm
// that the name passed to spec.input/output() is stored on the Port object itself.
class SetupNamedPortsOperator final : public Operator<> {
 public:
  Input<std::uint32_t> input_;
  Output<std::uint32_t> output_;
  Output<Tensor> tensor_;

  void setup(OperatorSpec& spec) override {
    spec.input(input_, "input");
    spec.output(output_, "output");
    spec.output(tensor_, "tensor");
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator covering all three payload kinds in one spec:
//   scalar_ → kInlineTyped (plain C++ value type)
//   schema_ → kSerializedBytes (type with a consteval schema_identity<> specialization)
//   tensor_ → kDescriptorBacked (holoscan::Tensor with descriptor-backed transport)
// Representative test: AdmittedPayloadFamiliesRetainDistinctColdTypeMetadata
class TypedMetadataDeclarationsOperator final : public Operator<> {
 public:
  Input<std::uint32_t> scalar_{"scalar"};
  Output<OperatorSpecSchemaPayload> schema_{"schema"};
  Output<Tensor> tensor_{"tensor"};

  void setup(OperatorSpec& spec) override {
    spec.input(scalar_);
    spec.output(schema_);
    spec.output(tensor_);
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Operator that registers all four readiness source kinds and then configures them
// out-of-declaration-order.  This catches any implementation bug where builders are
// associated by a shared insertion index instead of the record's exact family+index.
// Representative test: InterleavedReadinessKindsRetainTheirExactBuilderFamilies
class InterleavedReadinessDeclarationsOperator final : public Operator<> {
 public:
  ReadableSource readable_;
  NotificationSource notification_;
  DoorbellSource doorbell_;
  SpinSource spin_;

  void setup(OperatorSpec& spec) override {
    auto& readable = spec.readable_source(readable_, "readable");
    auto& notification = spec.notification_source(notification_, "notification");
    auto& doorbell = spec.doorbell_source(doorbell_, "doorbell");
    auto& spin = spec.spin_source(spin_, "spin");

    // Exercise every builder after all four records exist. This detects accidental association
    // through one shared insertion index instead of the record's exact builder family and index.
    readable.capacity(1).clock(ClockKey{"readable-clock"});
    notification.capacity(4).sender_reference_capacity(3).clock(ClockKey{"notification-clock"});
    doorbell.capacity(2).clock(ClockKey{"doorbell-clock"});
    spin.capacity(1).clock(ClockKey{"spin-clock"});
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

// Convenience wrapper that constructs a fresh SyntheticOperatorBinding and invokes
// declare_ports on the provided operator.  Returns the Error from the first violation,
// or void on success.  Used by all rejection and acceptance tests in this file.
template <typename Op>
expected<void, Error> declare_once(Op& op) {
  detail::SyntheticOperatorBinding binding;
  return binding.declare_ports(op);
}

// REQ: OPERATOR-004
TEST(OperatorSpec, BindingBuildersAreColdMetadataNotPortReplacementValues) {
  EXPECT_FALSE((std::is_convertible_v<InputSpecBuilder&, ScalarInput>));
  EXPECT_FALSE((std::is_convertible_v<OutputSpecBuilder&, ScalarOutput>));
  EXPECT_FALSE((std::is_convertible_v<TensorOutputSpecBuilder&, TensorOutput>));
}

TEST(OperatorSpec, SetupNamesBecomeTheStableNamesOfInitiallyUnnamedPortDeclarations) {
  SetupNamedPortsOperator op;
  OperatorSpec spec;

  op.setup(spec);

  EXPECT_EQ(op.input_.name(), "input");
  EXPECT_EQ(op.output_.name(), "output");
  EXPECT_EQ(op.tensor_.name(), "tensor");
}

// REQ: OPERATOR-004
TEST(OperatorSpec, TensorBoundsAreFiniteValues) {
  constexpr TensorAllocationBounds bounds{
      .max_rank = 8,
      .max_byte_span = 16U * 1024U * 1024U,
  };
  EXPECT_EQ(bounds.max_rank, 8);
  EXPECT_EQ(bounds.max_byte_span, 16U * 1024U * 1024U);
}

// REQ: OPERATOR-004
TEST(OperatorSpec, RejectsOnePortDeclarationBoundUnderDifferentNames) {
  ReusedPortDeclarationOperator op;
  const auto result = declare_once(op);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
}

// REQ: OPERATOR-004
TEST(OperatorSpec, RejectsEmptyReadinessSourceName) {
  EmptyReadinessNameOperator op;
  const auto result = declare_once(op);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
}

// REQ: OPERATOR-004
TEST(OperatorSpec, RejectsOneReadinessDeclarationBoundUnderDifferentNames) {
  ReusedReadinessDeclarationOperator op;
  const auto result = declare_once(op);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kInvalidArgument);
}

// REQ: OPERATOR-004
TEST(OperatorSpec, RejectsDuplicateNamesAcrossReadinessSourceKinds) {
  DuplicateReadinessNameOperator op;
  const auto result = declare_once(op);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kDuplicateName);
}

// REQ: OPERATOR-004
TEST(OperatorSpec, RejectsDuplicateNamesAcrossPortAndReadinessKinds) {
  DuplicatePortAndReadinessNameOperator op;
  const auto result = declare_once(op);
  ASSERT_FALSE(result);
  EXPECT_EQ(result.error().code, ErrorCode::kDuplicateName);
}

// REQ: OPERATOR-004
TEST(OperatorSpec, FreshSpecsAcceptTheSameStableDeclarations) {
  ValidDeclarationsOperator op;
  EXPECT_TRUE(declare_once(op));
  EXPECT_TRUE(declare_once(op));
}

// REQ: OPERATOR-004
TEST(OperatorSpec, AdmittedPayloadFamiliesRetainDistinctColdTypeMetadata) {
  // Key setup: single operator with three ports covering kInlineTyped, kSerializedBytes,
  // and kDescriptorBacked.  Each port's cold metadata must be independently addressable.
  TypedMetadataDeclarationsOperator op;
  OperatorSpec spec;
  op.setup(spec);

  // Key assertion: local_type_token, payload_kind, and schema_identity are all
  // stored per-port and must match the template argument of each Input<>/Output<>.
  ASSERT_EQ(detail::OperatorSpecTestAccess::port_count(spec), 3);
  const auto scalar = detail::OperatorSpecTestAccess::port(spec, 0);
  EXPECT_EQ(scalar.declaration, &op.scalar_);
  EXPECT_EQ(scalar.direction, PortDirection::kInput);
  EXPECT_EQ(scalar.name, "scalar");
  EXPECT_FALSE(scalar.tensor);
  EXPECT_EQ(scalar.builder_index, 0);
  EXPECT_EQ(scalar.local_type_token, &detail::kOperatorSpecLocalTypeToken<std::uint32_t>);
  EXPECT_EQ(scalar.payload_kind, PayloadKind::kInlineTyped);
  EXPECT_FALSE(scalar.schema_identity.has_value());

  const auto schema = detail::OperatorSpecTestAccess::port(spec, 1);
  EXPECT_EQ(schema.declaration, &op.schema_);
  EXPECT_EQ(schema.direction, PortDirection::kOutput);
  EXPECT_EQ(schema.name, "schema");
  EXPECT_FALSE(schema.tensor);
  EXPECT_EQ(schema.builder_index, 0);
  EXPECT_EQ(schema.local_type_token,
            &detail::kOperatorSpecLocalTypeToken<OperatorSpecSchemaPayload>);
  EXPECT_EQ(schema.payload_kind, PayloadKind::kSerializedBytes);
  ASSERT_TRUE(schema.schema_identity.has_value());
  const SchemaIdentity schema_identity_value = schema.schema_identity.value_or(SchemaIdentity{});
  EXPECT_EQ(schema_identity_value.canonical_name, "holoscan.test.OperatorSpecSchemaPayload");
  EXPECT_EQ(schema_identity_value.version, 7);
  EXPECT_EQ(schema_identity_value.schema_sha256,
            schema_identity<OperatorSpecSchemaPayload>().schema_sha256);
  EXPECT_EQ(schema_identity_value.id, schema_identity<OperatorSpecSchemaPayload>().id);

  const auto tensor = detail::OperatorSpecTestAccess::port(spec, 2);
  EXPECT_EQ(tensor.declaration, &op.tensor_);
  EXPECT_EQ(tensor.direction, PortDirection::kOutput);
  EXPECT_EQ(tensor.name, "tensor");
  EXPECT_TRUE(tensor.tensor);
  EXPECT_EQ(tensor.builder_index, 0);
  EXPECT_EQ(tensor.local_type_token, &detail::kOperatorSpecLocalTypeToken<Tensor>);
  EXPECT_EQ(tensor.payload_kind, PayloadKind::kDescriptorBacked);
  EXPECT_FALSE(tensor.schema_identity.has_value());
}

// REQ: OPERATOR-006
TEST(OperatorSpec, InterleavedReadinessKindsRetainTheirExactBuilderFamilies) {
  InterleavedReadinessDeclarationsOperator op;
  OperatorSpec spec;
  op.setup(spec);

  using Family = detail::OperatorSpecTestAccess::SourceBuilderFamily;
  ASSERT_EQ(detail::OperatorSpecTestAccess::source_count(spec), 4);
  const auto readable = detail::OperatorSpecTestAccess::source(spec, 0);
  EXPECT_EQ(readable.declaration, &op.readable_);
  EXPECT_EQ(readable.name, "readable");
  EXPECT_EQ(readable.kind, ReadinessSourceKind::kReadable);
  EXPECT_EQ(readable.builder_family, Family::kSingleWitness);
  EXPECT_EQ(readable.builder_index, 0);

  const auto notification = detail::OperatorSpecTestAccess::source(spec, 1);
  EXPECT_EQ(notification.declaration, &op.notification_);
  EXPECT_EQ(notification.name, "notification");
  EXPECT_EQ(notification.kind, ReadinessSourceKind::kNotification);
  EXPECT_EQ(notification.builder_family, Family::kNotification);
  EXPECT_EQ(notification.builder_index, 0);

  const auto doorbell = detail::OperatorSpecTestAccess::source(spec, 2);
  EXPECT_EQ(doorbell.declaration, &op.doorbell_);
  EXPECT_EQ(doorbell.name, "doorbell");
  EXPECT_EQ(doorbell.kind, ReadinessSourceKind::kDoorbell);
  EXPECT_EQ(doorbell.builder_family, Family::kDoorbell);
  EXPECT_EQ(doorbell.builder_index, 0);

  const auto spin = detail::OperatorSpecTestAccess::source(spec, 3);
  EXPECT_EQ(spin.declaration, &op.spin_);
  EXPECT_EQ(spin.name, "spin");
  EXPECT_EQ(spin.kind, ReadinessSourceKind::kSpin);
  EXPECT_EQ(spin.builder_family, Family::kSingleWitness);
  EXPECT_EQ(spin.builder_index, 1);
}

}  // namespace
}  // namespace holoscan
