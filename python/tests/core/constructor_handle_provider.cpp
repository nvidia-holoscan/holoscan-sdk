/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// A synthetic, independently built provider of native constructor handles. It is compiled as its
// own nanobind extension in a separate nanobind domain, so Holoscan's binding shares no
// binding-library type registry with it; only the versioned capsule exchange connects them.

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/authoring/constructor_handle.hpp>
#include <holoscan/authoring/register_operator.hpp>
#include <holoscan/authoring/registry.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/python/constructor_handle_capsule.hpp>

namespace nb = nanobind;

namespace holoscan::python::test {
namespace {

std::atomic<int> live_channels{};
std::atomic<int> live_sensors{};
std::atomic<int> receiver_constructions{};

// Provider data channel, created as a native owner so its lifetime is independent of Python.
struct TestChannel : std::enable_shared_from_this<TestChannel> {
  explicit TestChannel(int channel_id) : id(channel_id) { live_channels.fetch_add(1); }
  TestChannel(const TestChannel&) = delete;
  TestChannel& operator=(const TestChannel&) = delete;
  virtual ~TestChannel() { live_channels.fetch_sub(1); }
  int id{};
};

// Negative recipe: a Python-owned channel whose exporter converts `self` through nanobind's
// shared_ptr caster, which creates an owner that keeps the Python object alive.
struct AnchoredChannel final : TestChannel {
  using TestChannel::TestChannel;
};

// Provider camera sensor driven by the receiver's two lifecycle callbacks.
struct TestSensor : std::enable_shared_from_this<TestSensor> {
  TestSensor() { live_sensors.fetch_add(1); }
  TestSensor(const TestSensor&) = delete;
  TestSensor& operator=(const TestSensor&) = delete;
  ~TestSensor() { live_sensors.fetch_sub(1); }
  void start() { starts.fetch_add(1); }
  void stop() { stops.fetch_add(1); }
  std::atomic<int> starts{};
  std::atomic<int> stops{};
};

constexpr authoring::ConstructorHandleDescriptor kChannelDescriptor{
    .name = "holoscan.test.py.TestChannel", .description = "Synthetic provider data channel."};
constexpr authoring::ConstructorHandleDescriptor kSensorDescriptor{
    .name = "holoscan.test.py.TestSensor", .description = "Synthetic provider camera sensor."};

const authoring::ConstructorHandleType<TestChannel>& channel_handle_type() {
  static const authoring::ConstructorHandleType<TestChannel> type{kChannelDescriptor};
  return type;
}

const authoring::ConstructorHandleType<TestSensor>& sensor_handle_type() {
  static const authoring::ConstructorHandleType<TestSensor> type{kSensorDescriptor};
  return type;
}

struct ReceiverParams {
  std::uint32_t frame_size{64U};
};

HOLOSCAN_PARAM_SCHEMA(ReceiverParams, param_schema("holoscan.test.py.ReceiverParams", 1),
                      field<&ReceiverParams::frame_size>("frame_size").range(1U, 4096U));

struct DescriptionOnly {};

// Requires a live channel and a callback pair at construction, like a device-backed receiver.
class HandleReceiverOp final : public Operator<ReceiverParams> {
 public:
  HandleReceiverOp(std::shared_ptr<TestChannel> channel, std::function<void()> device_start,
                   std::function<void()> device_stop)
      : channel_(std::move(channel)),
        device_start_(std::move(device_start)),
        device_stop_(std::move(device_stop)) {
    if (!channel_) {
      throw std::invalid_argument("a data channel is required");
    }
    receiver_constructions.fetch_add(1);
  }

  explicit HandleReceiverOp(DescriptionOnly) {}

  void setup(OperatorSpec& spec) override { spec.output(output, "output"); }

  [[nodiscard]] Contract contract() const override {
    Contract result;
    result.trigger(OnClock{.period = std::chrono::milliseconds{1}});
    return result;
  }

  [[nodiscard]] expected<void, Error> compute(ExecutionContext&) override { return {}; }

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

  Output<std::uint32_t> output;

 private:
  std::shared_ptr<TestChannel> channel_;
  std::function<void()> device_start_;
  std::function<void()> device_stop_;
};

struct ReceiverThunks final : authoring::ManualOperatorRegistrationThunks<HandleReceiverOp> {
  static void set_params(const authoring::OpaqueOperatorRef& op,
                         const authoring::MaterializedParameters& params) {
    if (params.type() == nullptr || params.type()->local_type_token !=
                                        param_type_descriptor<ReceiverParams>().local_type_token) {
      throw RuntimeError{Error{ErrorCode::kInvalidArgument, "AUTHORING_PARAMETER_TYPE_MISMATCH"}};
    }
    typed_handle(op).set_params(*static_cast<const ReceiverParams*>(params.data()));
  }

  static expected<authoring::OpaqueOperatorRef, Error> add_from_inputs(
      Graph& graph, std::string_view name, const authoring::OperatorConstructionInputs& inputs) {
    auto channel = inputs.require_shared("data_channel", channel_handle_type());
    if (!channel) {
      return make_unexpected(std::move(channel).error());
    }
    auto sensor = inputs.optional_shared("camera_sensor", sensor_handle_type());
    if (!sensor) {
      return make_unexpected(std::move(sensor).error());
    }
    auto params = authoring::materialize_parameters(param_type_descriptor<ReceiverParams>(),
                                                    param_schema_descriptor<ReceiverParams>(),
                                                    inputs.values);
    if (!params) {
      return make_unexpected(std::move(params).error());
    }
    std::function<void()> device_start;
    std::function<void()> device_stop;
    if (*sensor) {
      device_start = [owner = *sensor] { owner->start(); };
      device_stop = [owner = *sensor] { owner->stop(); };
    }
    return invoke_add([&] {
      auto op = graph.op<HandleReceiverOp>(name, *channel, device_start, device_stop);
      op.set_params(*static_cast<const ReceiverParams*>(params->data()));
      return make_operator_ref(op);
    });
  }

  // Reads the sensor actions as SDK callback handles, which Python callables can supply.
  static expected<authoring::OpaqueOperatorRef, Error> add_from_callbacks(
      Graph& graph, std::string_view name, const authoring::OperatorConstructionInputs& inputs) {
    auto channel = inputs.require_shared("data_channel", channel_handle_type());
    if (!channel) {
      return make_unexpected(std::move(channel).error());
    }
    auto device_start = inputs.callback("device_start");
    if (!device_start) {
      return make_unexpected(std::move(device_start).error());
    }
    auto device_stop = inputs.callback("device_stop");
    if (!device_stop) {
      return make_unexpected(std::move(device_stop).error());
    }
    return invoke_add([&] {
      return make_operator_ref(
          graph.op<HandleReceiverOp>(name, *channel, *device_start, *device_stop));
    });
  }

  static expected<authoring::OpaqueOperatorRef, Error> add_for_description(Graph& graph,
                                                                           std::string_view name) {
    return invoke_add(
        [&] { return make_operator_ref(graph.op<HandleReceiverOp>(name, DescriptionOnly{})); });
  }
};

constexpr std::array kReceiverHandles{
    authoring::ConstructorHandleInput{.name = "data_channel",
                                      .description = "Discovered data channel.",
                                      .type = &kChannelDescriptor},
    authoring::ConstructorHandleInput{.name = "camera_sensor",
                                      .description = "Sensor driven with the receiver.",
                                      .type = &kSensorDescriptor,
                                      .required = false},
};

constexpr authoring::RegisteredConstructorInputs kReceiverInputs{
    .handles = kReceiverHandles,
    .add_to_graph_from_inputs = &ReceiverThunks::add_from_inputs,
    .add_for_description = &ReceiverThunks::add_for_description,
};

// The callback descriptor comes from a Core function, so this table is built at run time.
const authoring::RegisteredConstructorInputs& callback_receiver_inputs() {
  static const std::array handles{
      authoring::ConstructorHandleInput{.name = "data_channel",
                                        .description = "Discovered data channel.",
                                        .type = &kChannelDescriptor},
      authoring::ConstructorHandleInput{.name = "device_start",
                                        .description = "Called when the receiver starts.",
                                        .type = &authoring::callback_handle_type().descriptor(),
                                        .required = false},
      authoring::ConstructorHandleInput{.name = "device_stop",
                                        .description = "Called when the receiver stops.",
                                        .type = &authoring::callback_handle_type().descriptor(),
                                        .required = false},
  };
  static const authoring::RegisteredConstructorInputs inputs{
      .handles = handles,
      .add_to_graph_from_inputs = &ReceiverThunks::add_from_callbacks,
      .add_for_description = &ReceiverThunks::add_for_description,
  };
  return inputs;
}

authoring::RegisteredOperatorType make_receiver_row(std::string_view name, bool declared) {
  using Thunks = ReceiverThunks;
  return authoring::RegisteredOperatorType{
      .name = name,
      .param_type = &param_type_descriptor<ReceiverParams>(),
      .param_schema = declared ? param_schema_descriptor<ReceiverParams>() : nullptr,
      .parameter_introspection = declared ? authoring::OperatorParameterIntrospection::kDeclared
                                          : authoring::OperatorParameterIntrospection::kPassThrough,
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
      .constructor_inputs = &kReceiverInputs,
  };
}

authoring::RegisteredOperatorType make_callback_receiver_row() {
  auto row = make_receiver_row("holoscan.test.py.CallbackReceiver", false);
  row.constructor_inputs = &callback_receiver_inputs();
  return row;
}

void register_rows() {
  auto& registry = authoring::authoring_registry();
  for (const auto& [name, declared] :
       {std::pair{"holoscan.test.py.HandleReceiver", true},
        std::pair{"holoscan.test.py.PassThroughHandleReceiver", false}}) {
    if (registry.operator_type(name)) {
      continue;
    }
    auto registered = registry.register_operator(make_receiver_row(name, declared));
    if (!registered) {
      throw RuntimeError{registered.error()};
    }
  }
  if (!registry.operator_type("holoscan.test.py.CallbackReceiver")) {
    auto registered = registry.register_operator(make_callback_receiver_row());
    if (!registered) {
      throw RuntimeError{registered.error()};
    }
  }
}

nb::object export_handle(PyObject* capsule) {
  if (capsule == nullptr) {
    throw nb::python_error();
  }
  return nb::steal(capsule);
}

}  // namespace
}  // namespace holoscan::python::test

NB_MODULE(holoscan_constructor_handle_provider, module) {
  using namespace holoscan::python::test;  // NOLINT(build/namespaces)
  nb::class_<TestChannel>(module, "TestChannel", nb::is_weak_referenceable())
      .def(nb::new_([](int channel_id) { return std::make_shared<TestChannel>(channel_id); }),
           nb::arg("channel_id"))
      .def_prop_ro("id", [](const TestChannel& self) { return self.id; })
      .def("__holoscan_constructor_handle__", [](TestChannel& self, const std::string& type_name) {
        return export_handle(holoscan::python::export_constructor_handle(
            type_name, channel_handle_type(), self.shared_from_this()));
      });
  nb::class_<AnchoredChannel>(module, "AnchoredChannel", nb::is_weak_referenceable())
      .def(nb::init<int>(), nb::arg("channel_id"))
      .def("__holoscan_constructor_handle__",
           [](std::shared_ptr<AnchoredChannel> self, const std::string& type_name) {
             return export_handle(holoscan::python::export_constructor_handle(
                 type_name, channel_handle_type(), std::shared_ptr<TestChannel>(std::move(self))));
           });
  nb::class_<TestSensor>(module, "TestSensor")
      .def(nb::new_([] { return std::make_shared<TestSensor>(); }))
      .def_prop_ro("starts", [](const TestSensor& self) { return self.starts.load(); })
      .def_prop_ro("stops", [](const TestSensor& self) { return self.stops.load(); })
      .def("__holoscan_constructor_handle__", [](TestSensor& self, const std::string& type_name) {
        return export_handle(holoscan::python::export_constructor_handle(
            type_name, sensor_handle_type(), self.shared_from_this()));
      });
  // Register on import, as a real provider package does, so stub generation can discover the rows.
  register_rows();
  module.def("register_rows", &register_rows);
  module.def("export_null_channel", [](const std::string& type_name) {
    return export_handle(holoscan::python::export_constructor_handle(
        type_name, channel_handle_type(), std::shared_ptr<TestChannel>{}));
  });
  module.def("live_channels", [] { return live_channels.load(); });
  module.def("live_sensors", [] { return live_sensors.load(); });
  module.def("receiver_constructions", [] { return receiver_constructions.load(); });
}
