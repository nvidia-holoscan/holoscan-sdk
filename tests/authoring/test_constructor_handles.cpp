/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/authoring/constructor_handle.hpp>
#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "../../src/authoring/constructor_binding_factory.hpp"
#include "../../src/core/graph_state.hpp"
#include "../common/allocation_counter.hpp"

namespace {

using holoscan::authoring::ConstructorHandle;
using holoscan::authoring::ConstructorHandleAssignment;
using holoscan::authoring::ConstructorHandleDescriptor;
using holoscan::authoring::ConstructorHandleInput;
using holoscan::authoring::ConstructorHandleType;
using holoscan::authoring::OperatorConstructionInputs;
using holoscan::authoring::ParameterAssignment;
using holoscan::authoring::ParameterValue;

// Stands in for a provider connection such as a network data channel.
// Representative test: AuthoringConstructorHandles.RetainsOwnersThroughCompileAndGraphRetirement.
struct TestChannel {
  int id{};
};

// Stands in for a provider camera sensor whose lifecycle a receiver drives through two callbacks.
// Representative test: AuthoringConstructorHandles.AdaptsTheSensorOwnerIntoNativeCallbacks.
struct TestSensor {
  void start() { starts.fetch_add(1, std::memory_order_relaxed); }
  void stop() { stops.fetch_add(1, std::memory_order_relaxed); }
  std::atomic<int> starts{};
  std::atomic<int> stops{};
};

// One descriptor per interface type, published only through typed accessors, as a provider would.
constexpr ConstructorHandleDescriptor kTestChannelDescriptor{
    .name = "holoscan.test.TestChannel", .description = "Synthetic provider data channel."};
constexpr ConstructorHandleDescriptor kTestSensorDescriptor{
    .name = "holoscan.test.TestSensor", .description = "Synthetic provider camera sensor."};

const ConstructorHandleType<TestChannel>& test_channel_handle_type() {
  static const ConstructorHandleType<TestChannel> type{kTestChannelDescriptor};
  return type;
}

const ConstructorHandleType<TestSensor>& test_sensor_handle_type() {
  static const ConstructorHandleType<TestSensor> type{kTestSensorDescriptor};
  return type;
}

struct ReceiverParams {
  std::uint32_t frame_size{64U};
};

HOLOSCAN_PARAM_SCHEMA(ReceiverParams, holoscan::param_schema("holoscan.test.ReceiverParams", 1),
                      holoscan::field<&ReceiverParams::frame_size>("frame_size").range(1U, 4096U));

std::atomic<int> receiver_constructions{};
std::atomic<int> receiver_description_constructions{};

// Selects the description-only constructor used by hardware-free discovery.
struct DescriptionOnly {};

// Requires a live channel and a callback pair at construction, like a device-backed receiver.
// Representative test: AuthoringConstructorHandles.ConstructsFromTwoTypedHandlesAndValues.
class HandleReceiverOp final : public holoscan::Operator<ReceiverParams> {
 public:
  HandleReceiverOp(std::shared_ptr<TestChannel> channel, std::function<void()> device_start,
                   std::function<void()> device_stop)
      : channel_(std::move(channel)),
        device_start_(std::move(device_start)),
        device_stop_(std::move(device_stop)) {
    if (!channel_) {
      throw std::invalid_argument("a data channel is required");
    }
    receiver_constructions.fetch_add(1, std::memory_order_relaxed);
  }

  explicit HandleReceiverOp(DescriptionOnly) {
    receiver_description_constructions.fetch_add(1, std::memory_order_relaxed);
  }

  void setup(holoscan::OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract result;
    result.trigger(holoscan::OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  void start() override {
    if (device_start_) {
      device_start_();
    }
  }

  void stop() override {
    if (device_stop_) {
      device_stop_();
    }
  }

  [[nodiscard]] const std::shared_ptr<TestChannel>& channel() const noexcept { return channel_; }

  holoscan::Output<std::uint32_t> output;

 private:
  std::shared_ptr<TestChannel> channel_;
  std::function<void()> device_start_;
  std::function<void()> device_stop_;
};

// Consumes the receiver output so the fixture graph compiles.
// Representative test: AuthoringConstructorHandles.RetainsOwnersThroughCompileAndGraphRetirement.
class ReceiverSink final : public holoscan::Operator<> {
 public:
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input"); }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    return {};
  }

  holoscan::Input<std::uint32_t> input;
};

struct ReceiverThunks final
    : holoscan::authoring::ManualOperatorRegistrationThunks<HandleReceiverOp> {
  using ManualOperatorRegistrationThunks::make_operator_ref;
  using ManualOperatorRegistrationThunks::typed_handle;

  static void set_params(const holoscan::authoring::OpaqueOperatorRef& op,
                         const holoscan::authoring::MaterializedParameters& params) {
    if (params.type() == nullptr ||
        params.type()->local_type_token !=
            holoscan::param_type_descriptor<ReceiverParams>().local_type_token) {
      throw holoscan::RuntimeError{holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                                                   "AUTHORING_PARAMETER_TYPE_MISMATCH"}};
    }
    typed_handle(op).set_params(*static_cast<const ReceiverParams*>(params.data()));
  }
};

// Recovers both typed owners, adapts the sensor into two owning callbacks, and applies the
// declared value configuration as one compound declaration.
holoscan::expected<holoscan::authoring::OpaqueOperatorRef, holoscan::Error>
add_receiver_from_inputs(holoscan::Graph& graph, std::string_view name,
                         const OperatorConstructionInputs& inputs) {
  auto channel = inputs.require_shared("data_channel", test_channel_handle_type());
  if (!channel) {
    return holoscan::make_unexpected(std::move(channel).error());
  }
  auto sensor = inputs.require_shared("camera_sensor", test_sensor_handle_type());
  if (!sensor) {
    return holoscan::make_unexpected(std::move(sensor).error());
  }
  auto params = holoscan::authoring::materialize_parameters(
      holoscan::param_type_descriptor<ReceiverParams>(),
      holoscan::param_schema_descriptor<ReceiverParams>(),
      inputs.values);
  if (!params) {
    return holoscan::make_unexpected(std::move(params).error());
  }
  return ReceiverThunks::invoke_add([&] {
    auto op = graph.op<HandleReceiverOp>(
        name,
        *channel,
        [owner = *sensor] { owner->start(); },
        [owner = *sensor] { owner->stop(); });
    op.set_params(*static_cast<const ReceiverParams*>(params->data()));
    return ReceiverThunks::make_operator_ref(op);
  });
}

holoscan::expected<holoscan::authoring::OpaqueOperatorRef, holoscan::Error>
add_receiver_for_description(holoscan::Graph& graph, std::string_view name) {
  return ReceiverThunks::invoke_add([&] {
    return ReceiverThunks::make_operator_ref(graph.op<HandleReceiverOp>(name, DescriptionOnly{}));
  });
}

constexpr std::array kReceiverHandles{
    ConstructorHandleInput{.name = "data_channel",
                           .description = "Discovered data channel.",
                           .type = &kTestChannelDescriptor},
    ConstructorHandleInput{.name = "camera_sensor",
                           .description = "Sensor started and stopped with the receiver.",
                           .type = &kTestSensorDescriptor},
    ConstructorHandleInput{.name = "diagnostics_channel",
                           .description = "Optional second channel of the same type.",
                           .type = &kTestChannelDescriptor,
                           .required = false},
};

constexpr holoscan::authoring::RegisteredConstructorInputs kReceiverInputs{
    .handles = kReceiverHandles,
    .add_to_graph_from_inputs = &add_receiver_from_inputs,
    .add_for_description = &add_receiver_for_description,
};

holoscan::authoring::RegisteredOperatorType make_receiver_row(
    std::string_view name,
    const holoscan::authoring::RegisteredConstructorInputs* inputs = &kReceiverInputs) {
  using Thunks = ReceiverThunks;
  return holoscan::authoring::RegisteredOperatorType{
      .name = name,
      .param_type = &holoscan::param_type_descriptor<ReceiverParams>(),
      .param_schema = holoscan::param_schema_descriptor<ReceiverParams>(),
      .parameter_introspection = holoscan::authoring::OperatorParameterIntrospection::kDeclared,
      .add_to_graph = &Thunks::add_without_constructor_handles,
      .add_to_graph_from_assignments = &Thunks::reject_without_constructor_handles,
      .set_params = &Thunks::set_params,
      .port = &Thunks::port,
      .ports = &Thunks::ports,
      .region_add = &Thunks::region_add,
      .partition_add = &Thunks::partition_add,
      .lifecycle_before = &Thunks::lifecycle_before,
      .lifecycle_allow_self_restart = &Thunks::lifecycle_allow_self_restart,
      .lifecycle_allow_external_restart = &Thunks::lifecycle_allow_external_restart,
      .lifecycle_allow_restart_request = &Thunks::lifecycle_allow_restart_request,
      .constructor_inputs = inputs,
  };
}

// Reusable Graph whose compose() authors the handle receiver through its registry row, the way a
// Python Graph subclass would through Graph.op. Its constructor arguments are the caller's owners.
// Representative test: AuthoringConstructorHandles.EmbeddingKeepsTheBindingRecordAndForkGuard.
class HandleReceiverGraph final : public holoscan::Graph {
 public:
  HandleReceiverGraph(std::shared_ptr<TestChannel> channel, std::shared_ptr<TestSensor> sensor)
      : channel_(std::move(channel)), sensor_(std::move(sensor)) {}

  holoscan::Output<std::uint32_t> output{"output"};

  [[nodiscard]] holoscan::ClockKey clock_import() const { return clock_import_; }

 protected:
  void compose() override;

 private:
  std::shared_ptr<TestChannel> channel_;
  std::shared_ptr<TestSensor> sensor_;
  holoscan::ClockKey clock_import_;
};

template <typename T>
ConstructorHandle make_handle(const ConstructorHandleType<T>& type, std::shared_ptr<T> owner) {
  auto handle = ConstructorHandle::from_shared(type, std::move(owner));
  if (!handle) {
    throw holoscan::RuntimeError{handle.error()};
  }
  return std::move(*handle);
}

const holoscan::authoring::detail::ConstructorBindingFactory* binding_factory(
    holoscan::Graph& graph, std::string_view name) {
  const auto state = holoscan::detail::GraphPayloadRegistrationAccess::state(graph).lock();
  const auto found =
      std::ranges::find(state->operators_, name, &holoscan::detail::OperatorStorage::name);
  if (found == state->operators_.end()) {
    return nullptr;
  }
  return dynamic_cast<const holoscan::authoring::detail::ConstructorBindingFactory*>(
      found->factory.get());
}

std::string token_of(const holoscan::Error& error) {
  return error.message.substr(0, error.message.find(' '));
}

// REQ: AUTHOR-024
TEST(AuthoringConstructorHandles, CarrierPreservesOwnershipAndExactType) {
  // Key setup: Wrap one owner, then ask for it under the right and a same-named foreign descriptor.
  auto channel = std::make_shared<TestChannel>(TestChannel{.id = 7});
  const auto handle = ConstructorHandle::from_shared(test_channel_handle_type(), channel);
  ASSERT_TRUE(handle) << handle.error().message;
  static constexpr ConstructorHandleDescriptor kImpostor{.name = "holoscan.test.TestChannel"};
  const ConstructorHandleType<TestChannel> impostor{kImpostor};

  const auto recovered = handle->as_shared(test_channel_handle_type());
  const auto rejected = handle->as_shared(impostor);
  const auto null_owner =
      ConstructorHandle::from_shared(test_channel_handle_type(), std::shared_ptr<TestChannel>{});

  // Key assertion: Recovery shares the original control block; a name alone never authorizes it.
  ASSERT_TRUE(recovered) << recovered.error().message;
  EXPECT_EQ(recovered->get(), channel.get());
  EXPECT_FALSE(recovered->owner_before(channel) || channel.owner_before(*recovered));
  ASSERT_FALSE(rejected);
  EXPECT_EQ(token_of(rejected.error()), "AUTHORING_CONSTRUCTOR_HANDLE_TYPE_MISMATCH");
  ASSERT_FALSE(null_owner);
  EXPECT_EQ(token_of(null_owner.error()), "AUTHORING_CONSTRUCTOR_HANDLE_NULL");
  EXPECT_TRUE(handle->created_in_current_process());
}

// REQ: AUTHOR-024
TEST(AuthoringConstructorHandles, RegistrationValidatesHandleInputsAndTypeNames) {
  // Key setup: Build rows whose handle tables are malformed, collide with values, or reuse a stable
  // type name for a different descriptor.
  static constexpr std::array kDuplicate{
      ConstructorHandleInput{.name = "data_channel", .type = &kTestChannelDescriptor},
      ConstructorHandleInput{.name = "data_channel", .type = &kTestChannelDescriptor}};
  static constexpr std::array kValueCollision{
      ConstructorHandleInput{.name = "frame_size", .type = &kTestChannelDescriptor}};
  static constexpr ConstructorHandleDescriptor kConflicting{.name = "holoscan.test.TestChannel"};
  static constexpr std::array kConflictingType{
      ConstructorHandleInput{.name = "data_channel", .type = &kConflicting}};
  static constexpr holoscan::authoring::RegisteredConstructorInputs kDuplicateInputs{
      .handles = kDuplicate,
      .add_to_graph_from_inputs = &add_receiver_from_inputs,
      .add_for_description = &add_receiver_for_description};
  static constexpr holoscan::authoring::RegisteredConstructorInputs kCollisionInputs{
      .handles = kValueCollision,
      .add_to_graph_from_inputs = &add_receiver_from_inputs,
      .add_for_description = &add_receiver_for_description};
  static constexpr holoscan::authoring::RegisteredConstructorInputs kMissingDescription{
      .handles = kReceiverHandles, .add_to_graph_from_inputs = &add_receiver_from_inputs};
  // Two distinct descriptors under one stable name that no earlier row has published.
  static constexpr ConstructorHandleDescriptor kRowFirst{.name = "holoscan.test.RowLocalType"};
  static constexpr ConstructorHandleDescriptor kRowSecond{.name = "holoscan.test.RowLocalType"};
  static constexpr std::array kRowConflictingTypes{
      ConstructorHandleInput{.name = "primary", .type = &kRowFirst},
      ConstructorHandleInput{.name = "secondary", .type = &kRowSecond}};
  static constexpr holoscan::authoring::RegisteredConstructorInputs kRowConflictInputs{
      .handles = kRowConflictingTypes,
      .add_to_graph_from_inputs = &add_receiver_from_inputs,
      .add_for_description = &add_receiver_for_description};
  static constexpr holoscan::authoring::RegisteredConstructorInputs kConflictInputs{
      .handles = kConflictingType,
      .add_to_graph_from_inputs = &add_receiver_from_inputs,
      .add_for_description = &add_receiver_for_description};
  holoscan::authoring::AuthoringRegistry registry;

  const auto duplicate = registry.register_operator(make_receiver_row("dup", &kDuplicateInputs));
  const auto collision = registry.register_operator(make_receiver_row("col", &kCollisionInputs));
  const auto missing = registry.register_operator(make_receiver_row("miss", &kMissingDescription));
  const auto first = registry.register_operator(make_receiver_row("first"));
  const auto second = registry.register_operator(make_receiver_row("second"));
  const auto conflict = registry.register_operator(make_receiver_row("conflict", &kConflictInputs));
  const auto row_conflict =
      registry.register_operator(make_receiver_row("row-conflict", &kRowConflictInputs));

  // Key assertion: Malformed tables name the constructor_inputs field, identical descriptors are
  // shared across rows, and a second descriptor under the same stable name is rejected.
  for (const auto* result : {&duplicate, &collision, &missing}) {
    ASSERT_FALSE(*result);
    EXPECT_EQ(result->error().message,
              "AUTHORING_REGISTRY_ROW_INVALID kind=operator field=constructor_inputs");
  }
  EXPECT_TRUE(first) << first.error().message;
  EXPECT_TRUE(second) << second.error().message;
  ASSERT_FALSE(conflict);
  EXPECT_EQ(token_of(conflict.error()), "AUTHORING_CONSTRUCTOR_HANDLE_TYPE_CONFLICT");
  EXPECT_FALSE(registry.operator_type("conflict"));
  ASSERT_FALSE(row_conflict);
  EXPECT_EQ(token_of(row_conflict.error()), "AUTHORING_CONSTRUCTOR_HANDLE_TYPE_CONFLICT");
  EXPECT_FALSE(registry.operator_type("row-conflict"));
  EXPECT_FALSE(registry.constructor_handle_type("holoscan.test.RowLocalType"));
  const auto found = registry.constructor_handle_type("holoscan.test.TestChannel");
  ASSERT_TRUE(found) << found.error().message;
  EXPECT_EQ(&found->get(), &kTestChannelDescriptor);
}

// REQ: AUTHOR-024
TEST(AuthoringConstructorHandles, AllocationFailureDuringRegistrationPublishesNothing) {
  // Key setup: Count the allocations of one successful registration, then fail each of them in
  // turn, through both direct and automatic registration.
  std::size_t allocations{};
  {
    holoscan::authoring::AuthoringRegistry probe;
    const holoscan::test::ScopedAllocationCounter counter;
    ASSERT_TRUE(probe.register_operator(make_receiver_row("holoscan.test.AllocationReceiver")));
    allocations = counter.count();
  }
  ASSERT_GT(allocations, 0U);
  std::size_t observed_failures{};
  for (const bool automatic : {false, true}) {
    for (std::size_t checkpoint = 0U; checkpoint < allocations; ++checkpoint) {
      SCOPED_TRACE(::testing::Message()
                   << "automatic=" << automatic << " checkpoint=" << checkpoint);
      holoscan::authoring::AuthoringRegistry registry;
      bool registered = false;
      {
        const holoscan::test::ScopedAllocationFailure failure{checkpoint};
        const auto row = make_receiver_row("holoscan.test.AllocationReceiver");
        try {
          registered = automatic ? registry.register_operator_automatically(row)
                                 : registry.register_operator(row).has_value();
        } catch (const std::bad_alloc&) {
          registered = false;
        }
      }
      if (registered) {
        continue;
      }
      ++observed_failures;

      // Key assertion: A failed registration publishes neither the Operator nor any type name,
      // and the same row can then be registered.
      EXPECT_FALSE(registry.operator_type("holoscan.test.AllocationReceiver"));
      EXPECT_FALSE(registry.constructor_handle_type(kTestChannelDescriptor.name));
      EXPECT_FALSE(registry.constructor_handle_type(kTestSensorDescriptor.name));
      EXPECT_TRUE(
          registry.register_operator(make_receiver_row("holoscan.test.AllocationReceiver")));
    }
  }
  EXPECT_GT(observed_failures, 0U);
}

// REQ: AUTHOR-024
// REQ: AUTHOR-025
// REQ: AUTHOR-026
TEST(AuthoringConstructorHandles, DescribesPortsAndHandlesWithoutAnyHandle) {
  // Key setup: Describe a row whose real constructor rejects a missing channel.
  holoscan::authoring::AuthoringRegistry registry;
  ASSERT_TRUE(registry.register_operator(make_receiver_row("holoscan.test.HandleReceiver")));
  const int constructions = receiver_constructions.load();
  const int description_constructions = receiver_description_constructions.load();

  const auto described = registry.describe_operator("holoscan.test.HandleReceiver");

  // Key assertion: Discovery uses the description route and publishes typed handle metadata.
  ASSERT_TRUE(described) << described.error().message;
  EXPECT_EQ(receiver_constructions.load(), constructions);
  EXPECT_GT(receiver_description_constructions.load(), description_constructions);
  ASSERT_EQ(described->ports.size(), 1U);
  EXPECT_EQ(described->ports[0].name, "output");
  ASSERT_EQ(described->constructor_handles.size(), 3U);
  EXPECT_EQ(described->constructor_handles[0].name, "data_channel");
  EXPECT_EQ(described->constructor_handles[0].type_name, "holoscan.test.TestChannel");
  EXPECT_TRUE(described->constructor_handles[0].required);
  EXPECT_EQ(described->constructor_handles[1].type_name, "holoscan.test.TestSensor");
  EXPECT_FALSE(described->constructor_handles[2].required);
}

// REQ: AUTHOR-024
TEST(AuthoringConstructorHandles, ConstructsFromTwoTypedHandlesAndValues) {
  // Graph:
  //
  // receiver
  //
  // Key setup: Supply both required handles in reverse declaration order, the same channel owner
  // under the optional second name, and one declared value.
  const auto row = make_receiver_row("holoscan.test.HandleReceiver");
  auto channel = std::make_shared<TestChannel>(TestChannel{.id = 3});
  auto sensor = std::make_shared<TestSensor>();
  const std::array handles{
      ConstructorHandleAssignment{"camera_sensor", make_handle(test_sensor_handle_type(), sensor)},
      ConstructorHandleAssignment{"diagnostics_channel",
                                  make_handle(test_channel_handle_type(), channel)},
      ConstructorHandleAssignment{"data_channel", make_handle(test_channel_handle_type(), channel)},
  };
  const std::array values{
      ParameterAssignment{"frame_size", ParameterValue::from(std::uint32_t{128U})}};
  holoscan::Graph graph{"handle-receiver"};

  const auto added = holoscan::authoring::add_registered_operator(
      graph, row, "receiver", OperatorConstructionInputs{.values = values, .handles = handles});

  // Key assertion: The Operator holds the caller's channel, the Params update is applied, and the
  // factory records every supplied handle in declaration order, independent of argument order.
  ASSERT_TRUE(added) << added.error().message;
  const auto typed = ReceiverThunks::typed_handle(added->op);
  EXPECT_EQ(typed->channel().get(), channel.get());
  EXPECT_EQ(typed->params().frame_size, 128U);
  const auto* factory = binding_factory(graph, "receiver");
  ASSERT_NE(factory, nullptr);
  const auto& bindings = factory->record().bindings;
  ASSERT_EQ(bindings.size(), 3U);
  EXPECT_EQ(bindings[0].name, "data_channel");
  EXPECT_EQ(bindings[1].name, "camera_sensor");
  EXPECT_EQ(bindings[2].name, "diagnostics_channel");
  EXPECT_EQ(bindings[0].owner.owner().get(), bindings[2].owner.owner().get());
  EXPECT_EQ(sensor->starts.load(), 0);
}

// REQ: AUTHOR-024
TEST(AuthoringConstructorHandles, AdaptsTheSensorOwnerIntoNativeCallbacks) {
  // Graph:
  //
  // receiver
  //
  // Key setup: Author the receiver, then drop every caller-owned sensor reference.
  const auto row = make_receiver_row("holoscan.test.HandleReceiver");
  auto sensor = std::make_shared<TestSensor>();
  const std::weak_ptr<TestSensor> observed = sensor;
  holoscan::Graph graph{"sensor-callbacks"};
  {
    const std::array handles{
        ConstructorHandleAssignment{
            "data_channel",
            make_handle(test_channel_handle_type(), std::make_shared<TestChannel>())},
        ConstructorHandleAssignment{"camera_sensor",
                                    make_handle(test_sensor_handle_type(), std::move(sensor))},
    };
    ASSERT_TRUE(holoscan::authoring::add_registered_operator(
        graph, row, "receiver", OperatorConstructionInputs{.handles = handles}));
  }
  ASSERT_FALSE(observed.expired());
  const int starts = observed.lock()->starts.load();

  auto replayed = binding_factory(graph, "receiver")->create(nullptr);
  ASSERT_TRUE(replayed) << replayed.error().message;
  auto* receiver = dynamic_cast<HandleReceiverOp*>(replayed->get());
  ASSERT_NE(receiver, nullptr);
  receiver->start();
  receiver->stop();

  // Key assertion: Authoring never invokes the callbacks; a factory replay's callbacks reach the
  // same sensor after the caller's owner is gone.
  EXPECT_EQ(starts, 0);
  EXPECT_EQ(observed.lock()->starts.load(), 1);
  EXPECT_EQ(observed.lock()->stops.load(), 1);
}

// REQ: AUTHOR-024
// REQ: AUTHOR-025
TEST(AuthoringConstructorHandles, RejectsInvalidHandleCallsBeforeMutation) {
  // Graph:
  //
  // existing
  //
  // Key setup: Reject unknown, duplicate, empty, swapped, missing, and value-colliding handles, and
  // any handle supplied to a row without handle inputs.
  const auto row = make_receiver_row("holoscan.test.HandleReceiver");
  auto channel = std::make_shared<TestChannel>();
  auto sensor = std::make_shared<TestSensor>();
  const auto channel_handle = make_handle(test_channel_handle_type(), channel);
  const auto sensor_handle = make_handle(test_sensor_handle_type(), sensor);
  holoscan::Graph graph{"invalid-handles"};
  auto existing = graph.op<ReceiverSink>("existing");
  const auto before = graph.view();
  const auto valid = std::array{ConstructorHandleAssignment{"data_channel", channel_handle},
                                ConstructorHandleAssignment{"camera_sensor", sensor_handle}};

  struct Case {
    std::vector<ConstructorHandleAssignment> handles;
    std::vector<ParameterAssignment> values;
    std::string_view token;
  };
  const std::array cases{
      Case{{valid[0], valid[1], {"extra", channel_handle}},
           {},
           "AUTHORING_CONSTRUCTOR_HANDLE_UNKNOWN"},
      Case{{valid[0], valid[0], valid[1]}, {}, "AUTHORING_CONSTRUCTOR_HANDLE_DUPLICATE"},
      Case{{valid[0], {"camera_sensor", ConstructorHandle{}}},
           {},
           "AUTHORING_CONSTRUCTOR_HANDLE_NULL"},
      Case{{{"data_channel", sensor_handle}, {"camera_sensor", channel_handle}},
           {},
           "AUTHORING_CONSTRUCTOR_HANDLE_TYPE_MISMATCH"},
      Case{{valid[0]}, {}, "AUTHORING_CONSTRUCTOR_HANDLE_REQUIRED"},
      Case{{valid[0], valid[1]},
           {ParameterAssignment{"data_channel", ParameterValue::from(std::uint32_t{1U})}},
           "AUTHORING_CONSTRUCTOR_HANDLE_VALUE_CONFLICT"},
  };
  const int constructions = receiver_constructions.load();
  // Snapshot after the case table holds its own handle copies.
  const long channel_owners = channel.use_count();
  const long sensor_owners = sensor.use_count();

  for (const Case& call : cases) {
    SCOPED_TRACE(call.token);
    const auto added = holoscan::authoring::add_registered_operator(
        graph,
        row,
        "receiver",
        OperatorConstructionInputs{.values = call.values, .handles = call.handles});
    ASSERT_FALSE(added);
    EXPECT_EQ(token_of(added.error()), call.token);
  }
  const auto plain_row =
      holoscan::authoring::authoring_registry().operator_type("holoscan.test.RegistryOperator");
  if (plain_row) {
    const auto unsupported = holoscan::authoring::add_registered_operator(
        graph, plain_row->get(), "plain", OperatorConstructionInputs{.handles = valid});
    ASSERT_FALSE(unsupported);
    EXPECT_EQ(unsupported.error().message, "AUTHORING_CONSTRUCTOR_HANDLE_UNSUPPORTED");
  }

  // Key assertion: No constructor ran, the Graph is unchanged, and no temporary owner leaked.
  EXPECT_EQ(receiver_constructions.load(), constructions);
  EXPECT_EQ(graph.view().identity(), before.identity());
  EXPECT_EQ(channel.use_count(), channel_owners);
  EXPECT_EQ(sensor.use_count(), sensor_owners);
  EXPECT_TRUE(holoscan::authoring::add_registered_operator(
      graph, row, "receiver", OperatorConstructionInputs{.handles = valid}));
}

// REQ: AUTHOR-024
TEST(AuthoringConstructorHandles, RetainsOwnersThroughCompileAndGraphRetirement) {
  // Graph:
  //
  // receiver --> sink
  //
  // Key setup: Author with temporary owners, compile, then retire the Graph before the plan.
  const auto row = make_receiver_row("holoscan.test.HandleReceiver");
  std::weak_ptr<TestChannel> channel_observer;
  std::weak_ptr<TestSensor> sensor_observer;
  std::optional<holoscan::ExecutionPlan> plan;
  const int constructions = receiver_constructions.load();
  {
    auto channel = std::make_shared<TestChannel>();
    auto sensor = std::make_shared<TestSensor>();
    channel_observer = channel;
    sensor_observer = sensor;
    holoscan::Graph graph{"handle-lifetime"};
    const std::array handles{
        ConstructorHandleAssignment{"data_channel",
                                    make_handle(test_channel_handle_type(), std::move(channel))},
        ConstructorHandleAssignment{"camera_sensor",
                                    make_handle(test_sensor_handle_type(), std::move(sensor))},
    };
    const auto added = holoscan::authoring::add_registered_operator(
        graph, row, "receiver", OperatorConstructionInputs{.handles = handles});
    ASSERT_TRUE(added) << added.error().message;
    auto sink = graph.op<ReceiverSink>("sink");
    graph.add_flow(ReceiverThunks::typed_handle(added->op)->output, sink->input);
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
    plan.emplace(holoscan::compile(graph));
    ASSERT_TRUE(plan->ok()) << plan->json();
  }

  // Key assertion: Compilation replayed the constructor with the same owners, and the accepted
  // plan keeps both owners alive after the Graph is gone; retiring the plan releases them.
  EXPECT_GT(receiver_constructions.load(), constructions + 1);
  EXPECT_FALSE(channel_observer.expired());
  EXPECT_FALSE(sensor_observer.expired());
  EXPECT_EQ(sensor_observer.lock()->starts.load(), 0);
  plan.reset();
  EXPECT_TRUE(channel_observer.expired());
  EXPECT_TRUE(sensor_observer.expired());
}

// REQ: AUTHOR-024
TEST(AuthoringConstructorHandles, ParameterReplacementKeepsTheBindingRecord) {
  // Graph:
  //
  // receiver
  //
  // Key setup: Replace the receiver's Params after authoring it with handles.
  const auto row = make_receiver_row("holoscan.test.HandleReceiver");
  auto channel = std::make_shared<TestChannel>();
  const std::array handles{
      ConstructorHandleAssignment{"data_channel", make_handle(test_channel_handle_type(), channel)},
      ConstructorHandleAssignment{
          "camera_sensor", make_handle(test_sensor_handle_type(), std::make_shared<TestSensor>())},
  };
  holoscan::Graph graph{"handle-params"};
  const auto added = holoscan::authoring::add_registered_operator(
      graph, row, "receiver", OperatorConstructionInputs{.handles = handles});
  ASSERT_TRUE(added) << added.error().message;

  auto typed = ReceiverThunks::typed_handle(added->op);
  typed.set_params(ReceiverParams{.frame_size = 256U});

  // Key assertion: The replacement factory still carries the record and the original owner.
  const auto* factory = binding_factory(graph, "receiver");
  ASSERT_NE(factory, nullptr);
  ASSERT_EQ(factory->record().bindings.size(), 2U);
  EXPECT_EQ(factory->record().bindings[0].owner.owner().get(), channel.get());
  EXPECT_EQ(typed->channel().get(), channel.get());
  EXPECT_EQ(typed->params().frame_size, 256U);
}

void HandleReceiverGraph::compose() {
  // Embedded Graphs do not inherit the parent's default clock; the parent binds this import.
  clock_import_ = import_clock("clock");
  set_default_clock(clock_import_);
  static const auto row = make_receiver_row("holoscan.test.EmbeddedHandleReceiver");
  const std::array handles{
      ConstructorHandleAssignment{"data_channel",
                                  make_handle(test_channel_handle_type(), channel_)},
      ConstructorHandleAssignment{"camera_sensor", make_handle(test_sensor_handle_type(), sensor_)},
  };
  const auto added = holoscan::authoring::add_registered_operator(
      *this, row, "receiver", OperatorConstructionInputs{.handles = handles});
  if (!added) {
    throw holoscan::RuntimeError{added.error()};
  }
  export_output(output, ReceiverThunks::typed_handle(added->op)->output);
}

// REQ: AUTHOR-024
TEST(AuthoringConstructorHandles, EmbeddingKeepsTheBindingRecordAndForkGuard) {
  // Graph:
  //
  // pipeline[receiver] --> sink
  //
  // Key setup: Embed a reusable Graph that authors the handle receiver in compose(), then drop
  // the caller's owners.
  std::weak_ptr<TestChannel> channel_observer;
  std::weak_ptr<TestSensor> sensor_observer;
  std::optional<holoscan::ExecutionPlan> plan;
  {
    auto channel = std::make_shared<TestChannel>();
    auto sensor = std::make_shared<TestSensor>();
    channel_observer = channel;
    sensor_observer = sensor;
    holoscan::Graph graph{"handle-embedding"};
    auto pipeline =
        graph.embed<HandleReceiverGraph>("pipeline", std::move(channel), std::move(sensor));
    auto sink = graph.op<ReceiverSink>("sink");
    graph.add_flow(pipeline->output, sink->input);
    const auto clock = graph.add_clock<holoscan::RealtimeClock>("clock");
    graph.set_default_clock(clock);
    graph.bind_clock(pipeline, pipeline->clock_import(), clock);
    plan.emplace(holoscan::compile(graph));
    ASSERT_TRUE(plan->ok()) << plan->json();

    // A forked child inherits the parent Graph, including the embedded receiver's factory copy.
    const pid_t child = fork();
    ASSERT_GE(child, 0);
    if (child == 0) {
      const int constructions = receiver_constructions.load();
      int failures = 0;
      try {
        if (holoscan::compile(graph).ok()) {
          failures |= 1;
        }
      } catch (...) {  // NOLINT(bugprone-empty-catch)
        // A rejected construction may also surface as a compile exception.
      }
      if (receiver_constructions.load() != constructions) {
        failures |= 2;
      }
      _exit(failures);
    }
    int status = 0;
    ASSERT_EQ(waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);
  }

  // Key assertion: The embedded copy kept the record: the fork guard refused replay in the child,
  // the plan keeps both owners alive after the Graph is gone, and retiring it releases them.
  EXPECT_FALSE(channel_observer.expired());
  EXPECT_FALSE(sensor_observer.expired());
  plan.reset();
  EXPECT_TRUE(channel_observer.expired());
  EXPECT_TRUE(sensor_observer.expired());
}

// REQ: AUTHOR-024
TEST(AuthoringConstructorHandles, ForkedChildCannotReplayOrReuseInheritedHandles) {
  // Graph:
  //
  // receiver --> sink
  //
  // Key setup: Author in the parent, then fork a child that inherits the Graph and a parent handle.
  const auto row = make_receiver_row("holoscan.test.HandleReceiver");
  const auto channel_handle =
      make_handle(test_channel_handle_type(), std::make_shared<TestChannel>());
  const auto sensor_handle = make_handle(test_sensor_handle_type(), std::make_shared<TestSensor>());
  const std::array handles{ConstructorHandleAssignment{"data_channel", channel_handle},
                           ConstructorHandleAssignment{"camera_sensor", sensor_handle}};
  holoscan::Graph graph{"handle-fork"};
  const auto added = holoscan::authoring::add_registered_operator(
      graph, row, "receiver", OperatorConstructionInputs{.handles = handles});
  ASSERT_TRUE(added) << added.error().message;
  auto sink = graph.op<ReceiverSink>("sink");
  graph.add_flow(ReceiverThunks::typed_handle(added->op)->output, sink->input);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    int failures = 0;
    const int constructions = receiver_constructions.load();
    const auto created = binding_factory(graph, "receiver")->create(nullptr);
    if (created || created.error().message != "CONSTRUCTOR_HANDLE_PROCESS_MISMATCH") {
      failures |= 1;
    }
    try {
      if (holoscan::compile(graph).ok()) {
        failures |= 2;
      }
    } catch (...) {  // NOLINT(bugprone-empty-catch)
      // A rejected construction may also surface as a compile exception.
    }
    const auto reused = holoscan::authoring::add_registered_operator(
        graph, row, "second", OperatorConstructionInputs{.handles = handles});
    if (reused || token_of(reused.error()) != "AUTHORING_CONSTRUCTOR_HANDLE_PROCESS_MISMATCH") {
      failures |= 4;
    }
    if (receiver_constructions.load() != constructions) {
      failures |= 8;
    }
    _exit(failures);
  }
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);

  // Key assertion: The child never ran the constructor, and the parent can still compile.
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0);
  EXPECT_TRUE(holoscan::compile(graph).ok());
}

// REQ: AUTHOR-027
TEST(AuthoringConstructorHandles, CallbackInputsShareOneCallableAndAllowOmission) {
  // Key setup: Supply one C++ callback under a callback input and omit another.
  int calls = 0;
  auto callback =
      std::make_shared<const holoscan::authoring::ConstructorCallback>([&calls] { ++calls; });
  const std::array handles{
      ConstructorHandleAssignment{
          "device_start", make_handle(holoscan::authoring::callback_handle_type(), callback)},
      ConstructorHandleAssignment{
          "data_channel", make_handle(test_channel_handle_type(), std::make_shared<TestChannel>())},
  };
  const OperatorConstructionInputs inputs{.handles = handles};

  auto start = inputs.callback("device_start");
  const auto omitted = inputs.callback("device_stop");
  const auto wrong = inputs.callback("data_channel");

  // Key assertion: Copies of the returned function share the supplied callable, an omitted input
  // is an empty function, and a non-callback handle is a type mismatch.
  ASSERT_TRUE(start) << start.error().message;
  const holoscan::authoring::ConstructorCallback copy = *start;
  (*start)();
  copy();
  EXPECT_EQ(calls, 2);
  ASSERT_TRUE(omitted) << omitted.error().message;
  EXPECT_FALSE(*omitted);
  ASSERT_FALSE(wrong);
  EXPECT_EQ(token_of(wrong.error()), "AUTHORING_CONSTRUCTOR_HANDLE_TYPE_MISMATCH");
  EXPECT_EQ(holoscan::authoring::callback_handle_type().descriptor().name, "holoscan.VoidCallback");
}

// REQ: AUTHOR-027
TEST(AuthoringConstructorHandles, CoreReservesTheCallbackTypeNameInEveryRegistry) {
  // Key setup: Try to publish a provider descriptor under the built-in callback name, before and
  // after a row that uses Core's descriptor, in fresh registries.
  static constexpr ConstructorHandleDescriptor kForeign{.name = "holoscan.VoidCallback"};
  static constexpr std::array kForeignHandles{
      ConstructorHandleInput{.name = "data_channel", .type = &kTestChannelDescriptor},
      ConstructorHandleInput{.name = "device_start", .type = &kForeign, .required = false}};
  static constexpr holoscan::authoring::RegisteredConstructorInputs kForeignInputs{
      .handles = kForeignHandles,
      .add_to_graph_from_inputs = &add_receiver_from_inputs,
      .add_for_description = &add_receiver_for_description};
  static const std::array kCanonicalHandles{
      ConstructorHandleInput{.name = "data_channel", .type = &kTestChannelDescriptor},
      ConstructorHandleInput{.name = "device_start",
                             .type = &holoscan::authoring::callback_handle_type().descriptor(),
                             .required = false}};
  static const holoscan::authoring::RegisteredConstructorInputs kCanonicalInputs{
      .handles = kCanonicalHandles,
      .add_to_graph_from_inputs = &add_receiver_from_inputs,
      .add_for_description = &add_receiver_for_description};

  const holoscan::authoring::AuthoringRegistry fresh;
  const auto initial = fresh.constructor_handle_type("holoscan.VoidCallback");
  ASSERT_TRUE(initial) << initial.error().message;
  EXPECT_EQ(&initial->get(), &holoscan::authoring::callback_handle_type().descriptor());

  holoscan::authoring::AuthoringRegistry foreign_first;
  const auto foreign_before =
      foreign_first.register_operator(make_receiver_row("foreign", &kForeignInputs));
  const auto canonical_after =
      foreign_first.register_operator(make_receiver_row("canonical", &kCanonicalInputs));
  holoscan::authoring::AuthoringRegistry canonical_first;
  const auto canonical_before =
      canonical_first.register_operator(make_receiver_row("canonical", &kCanonicalInputs));
  const auto canonical_again =
      canonical_first.register_operator(make_receiver_row("canonical-2", &kCanonicalInputs));
  const auto foreign_after =
      canonical_first.register_operator(make_receiver_row("foreign", &kForeignInputs));

  // Key assertion: The foreign descriptor is rejected in either order, rows using Core's
  // descriptor are accepted and may share it, and lookup resolves the name to Core's descriptor.
  ASSERT_FALSE(foreign_before);
  EXPECT_EQ(token_of(foreign_before.error()), "AUTHORING_CONSTRUCTOR_HANDLE_TYPE_CONFLICT");
  EXPECT_TRUE(canonical_after) << canonical_after.error().message;
  EXPECT_TRUE(canonical_before) << canonical_before.error().message;
  EXPECT_TRUE(canonical_again) << canonical_again.error().message;
  ASSERT_FALSE(foreign_after);
  EXPECT_EQ(token_of(foreign_after.error()), "AUTHORING_CONSTRUCTOR_HANDLE_TYPE_CONFLICT");
  const auto found = foreign_first.constructor_handle_type("holoscan.VoidCallback");
  ASSERT_TRUE(found) << found.error().message;
  EXPECT_EQ(&found->get(), &holoscan::authoring::callback_handle_type().descriptor());
}

}  // namespace
