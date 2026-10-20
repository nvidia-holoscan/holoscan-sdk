/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <Python.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
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

// REQ: PY-API-024
namespace {

std::atomic<int> live_channels{};

// A C++-owned provider object, kept in a provider-side table keyed by an integer for this test.
struct Channel {
  explicit Channel(int channel_id) : id(channel_id) { live_channels.fetch_add(1); }
  Channel(const Channel&) = delete;
  Channel& operator=(const Channel&) = delete;
  ~Channel() { live_channels.fetch_sub(1); }
  int id{};
};

std::mutex channels_mutex;
std::map<long, std::shared_ptr<Channel>> channels;
long next_channel_key{};

constexpr holoscan::authoring::ConstructorHandleDescriptor kChannelDescriptor{
    .name = "holoscan.installed.Channel", .description = "Installed-consumer channel."};

const holoscan::authoring::ConstructorHandleType<Channel>& channel_handle_type() {
  static const holoscan::authoring::ConstructorHandleType<Channel> type{kChannelDescriptor};
  return type;
}

struct DescriptionOnly {};

class ChannelSource final : public holoscan::Operator<> {
 public:
  explicit ChannelSource(std::shared_ptr<Channel> channel) : channel_(std::move(channel)) {}
  explicit ChannelSource(DescriptionOnly) {}

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

  holoscan::Output<std::uint32_t> output;

 private:
  std::shared_ptr<Channel> channel_;
};

struct SourceThunks final : holoscan::authoring::ManualOperatorRegistrationThunks<ChannelSource> {
  static void set_params(const holoscan::authoring::OpaqueOperatorRef&,
                         const holoscan::authoring::MaterializedParameters&) {
    throw holoscan::RuntimeError{holoscan::Error{holoscan::ErrorCode::kNotSupported,
                                                 "AUTHORING_MANUAL_ROW_PARAMETERS_IMMUTABLE"}};
  }

  static holoscan::expected<holoscan::authoring::OpaqueOperatorRef, holoscan::Error>
  add_from_inputs(holoscan::Graph& graph, std::string_view name,
                  const holoscan::authoring::OperatorConstructionInputs& inputs) {
    auto channel = inputs.require_shared("channel", channel_handle_type());
    if (!channel) {
      return holoscan::make_unexpected(std::move(channel).error());
    }
    return invoke_add([&] { return make_operator_ref(graph.op<ChannelSource>(name, *channel)); });
  }

  static holoscan::expected<holoscan::authoring::OpaqueOperatorRef, holoscan::Error>
  add_for_description(holoscan::Graph& graph, std::string_view name) {
    return invoke_add(
        [&] { return make_operator_ref(graph.op<ChannelSource>(name, DescriptionOnly{})); });
  }
};

constexpr std::array kHandles{holoscan::authoring::ConstructorHandleInput{
    .name = "channel", .description = "Provider channel.", .type = &kChannelDescriptor}};

constexpr holoscan::authoring::RegisteredConstructorInputs kInputs{
    .handles = kHandles,
    .add_to_graph_from_inputs = &SourceThunks::add_from_inputs,
    .add_for_description = &SourceThunks::add_for_description,
};

PyObject* register_rows(PyObject*, PyObject*) {
  auto& registry = holoscan::authoring::authoring_registry();
  constexpr std::string_view kName{"holoscan.installed.ChannelSource"};
  if (!registry.operator_type(kName)) {
    using Thunks = SourceThunks;
    auto registered = registry.register_operator(holoscan::authoring::RegisteredOperatorType{
        .name = kName,
        .param_type = &holoscan::param_type_descriptor<holoscan::NoParams>(),
        .parameter_introspection =
            holoscan::authoring::OperatorParameterIntrospection::kPassThrough,
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
        .constructor_inputs = &kInputs,
    });
    if (!registered) {
      PyErr_SetString(PyExc_RuntimeError, registered.error().message.c_str());
      return nullptr;
    }
  }
  Py_RETURN_NONE;
}

PyObject* new_channel(PyObject*, PyObject* args) {
  int channel_id = 0;
  if (PyArg_ParseTuple(args, "i", &channel_id) == 0) {
    return nullptr;
  }
  const std::lock_guard lock(channels_mutex);
  const long key = ++next_channel_key;
  channels.emplace(key, std::make_shared<Channel>(channel_id));
  return PyLong_FromLong(key);
}

PyObject* forget_channel(PyObject*, PyObject* args) {
  long key = 0;
  if (PyArg_ParseTuple(args, "l", &key) == 0) {
    return nullptr;
  }
  const std::lock_guard lock(channels_mutex);
  channels.erase(key);
  Py_RETURN_NONE;
}

PyObject* export_channel(PyObject*, PyObject* args) {
  long key = 0;
  const char* type_name = nullptr;
  if (PyArg_ParseTuple(args, "ls", &key, &type_name) == 0) {
    return nullptr;
  }
  std::shared_ptr<Channel> owner;
  {
    const std::lock_guard lock(channels_mutex);
    const auto found = channels.find(key);
    if (found == channels.end()) {
      PyErr_SetString(PyExc_KeyError, "unknown channel");
      return nullptr;
    }
    owner = found->second;
  }
  return holoscan::python::export_constructor_handle(type_name, channel_handle_type(), owner);
}

PyObject* live(PyObject*, PyObject*) {
  return PyLong_FromLong(live_channels.load());
}

std::array<PyMethodDef, 6> methods{{
    {"register_rows", &register_rows, METH_NOARGS, nullptr},
    {"new_channel", &new_channel, METH_VARARGS, nullptr},
    {"forget_channel", &forget_channel, METH_VARARGS, nullptr},
    {"export_channel", &export_channel, METH_VARARGS, nullptr},
    {"live_channels", &live, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
}};

PyModuleDef module_definition{PyModuleDef_HEAD_INIT,
                              "installed_handle_provider",
                              nullptr,
                              -1,
                              methods.data(),
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr};

}  // namespace

PyMODINIT_FUNC PyInit_installed_handle_provider() {
  return PyModule_Create(&module_definition);
}
