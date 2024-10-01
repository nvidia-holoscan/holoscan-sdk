/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "holoscan/core/executors/gxf/gxf_executor.hpp"

#include <signal.h>

#include <algorithm>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <common/assert.hpp>
#include <common/logger.hpp>

#include "holoscan/core/application.hpp"
#include "holoscan/core/arg.hpp"
#include "holoscan/core/condition.hpp"
#include "holoscan/core/conditions/gxf/downstream_affordable.hpp"
#include "holoscan/core/conditions/gxf/expiring_message.hpp"
#include "holoscan/core/conditions/gxf/message_available.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/domain/tensor.hpp"
#include "holoscan/core/errors.hpp"
#include "holoscan/core/fragment.hpp"
#include "holoscan/core/graph.hpp"
#include "holoscan/core/graphs/flow_graph.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/gxf/gxf_extension_registrar.hpp"
#include "holoscan/core/gxf/gxf_network_context.hpp"
#include "holoscan/core/gxf/gxf_operator.hpp"
#include "holoscan/core/gxf/gxf_resource.hpp"
#include "holoscan/core/gxf/gxf_scheduler.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/core/gxf/gxf_wrapper.hpp"
#include "holoscan/core/message.hpp"
#include "holoscan/core/messagelabel.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/resource.hpp"
#include "holoscan/core/resources/gxf/annotated_double_buffer_receiver.hpp"
#include "holoscan/core/resources/gxf/annotated_double_buffer_transmitter.hpp"
#include "holoscan/core/resources/gxf/dfft_collector.hpp"
#include "holoscan/core/resources/gxf/double_buffer_receiver.hpp"
#include "holoscan/core/resources/gxf/double_buffer_transmitter.hpp"
#include "holoscan/core/services/common/forward_op.hpp"
#include "holoscan/core/services/common/virtual_operator.hpp"
#include "holoscan/core/signal_handler.hpp"

#include "gxf/app/arg.hpp"
#include "gxf/std/clock.hpp"
#include "gxf/std/default_extension.hpp"
#include "gxf/std/extension_factory_helper.hpp"
#include "gxf/std/monitor.hpp"
#include "gxf/test/components/entity_monitor.hpp"

namespace holoscan::gxf {

namespace {
std::pair<uint64_t, uint64_t> get_capacity_and_policy(
    nvidia::gxf::Handle<nvidia::gxf::Component> component) {
  uint64_t capacity = 1;
  uint64_t policy = holoscan::gxf::get_default_queue_policy();
  if (component.is_null()) {
    HOLOSCAN_LOG_ERROR("Null component handle");
    return std::make_pair(capacity, policy);
  }
  auto maybe_capacity = component->getParameter<uint64_t>("capacity");
  if (maybe_capacity) {
    capacity = maybe_capacity.value();
  } else {
    HOLOSCAN_LOG_ERROR("Failed to get capacity, using default value of {}", capacity);
  }
  auto maybe_policy = component->getParameter<uint64_t>("policy");
  if (maybe_policy) {
    policy = maybe_policy.value();
  } else {
    HOLOSCAN_LOG_ERROR("Failed to get policy, using default value of {}", policy);
  }
  return std::make_pair(capacity, policy);
}

bool has_ucx_connector(std::shared_ptr<nvidia::gxf::GraphEntity> graph_entity) {
  auto has_ucx_receiver = graph_entity->try_get("nvidia::gxf::UcxReceiver");
  auto has_ucx_transmitter = graph_entity->try_get("nvidia::gxf::UcxTransmitter");
  return has_ucx_receiver || has_ucx_transmitter;
}

}  // namespace

static const std::vector<std::string> kDefaultGXFExtensions{
    "libgxf_std.so",
    "libgxf_cuda.so",
    "libgxf_multimedia.so",
    "libgxf_serialization.so",
    "libgxf_ucx.so",  // UcxContext, UcxReceiver, UcxTransmitter, etc.
};

static const std::vector<std::string> kDefaultHoloscanGXFExtensions{
    "libgxf_ucx_holoscan.so",  // serialize holoscan::Message
};

static nvidia::Severity s_gxf_log_level = nvidia::Severity::INFO;

void gxf_logging_holoscan_format(const char* file, int line, nvidia::Severity severity,
                                 const char* log, void*) {
  if (severity == nvidia::Severity::ALL || severity == nvidia::Severity::COUNT) {
    HOLOSCAN_LOG_ERROR("Invalid severity level ({}): Log severity cannot be 'ALL' or 'COUNT'.",
                       static_cast<int>(severity));
  }

  // Ignore severity if requested
  if (s_gxf_log_level == nvidia::Severity::NONE || severity > s_gxf_log_level) { return; }

  LogLevel holoscan_log_level = LogLevel::INFO;

  switch (severity) {
    case nvidia::Severity::VERBOSE:
      holoscan_log_level = LogLevel::TRACE;
      break;
    case nvidia::Severity::DEBUG:
      holoscan_log_level = LogLevel::DEBUG;
      break;
    case nvidia::Severity::INFO:
      holoscan_log_level = LogLevel::INFO;
      break;
    case nvidia::Severity::WARNING:
      holoscan_log_level = LogLevel::WARN;
      break;
    case nvidia::Severity::ERROR:
      holoscan_log_level = LogLevel::ERROR;
      break;
    case nvidia::Severity::PANIC:
      holoscan_log_level = LogLevel::CRITICAL;
      break;
    default:
      holoscan_log_level = LogLevel::INFO;
  }

  std::string_view file_str(file);
  std::string_view file_base = file_str.substr(file_str.find_last_of("/") + 1);

  holoscan::log_message(file_base.data(), line, "", holoscan_log_level, log);
}

static void setup_gxf_logging() {
  LogLevel holoscan_log_level = holoscan::log_level();
  nvidia::Severity gxf_log_level = nvidia::Severity::INFO;

  // If HOLOSCAN_EXECUTOR_LOG_LEVEL was defined set it based on that, otherwise
  // set it based on the current Holoscan log level.
  const char* gxf_log_env_name = "HOLOSCAN_EXECUTOR_LOG_LEVEL";
  const char* gxf_log_env_value = std::getenv(gxf_log_env_name);

  if (gxf_log_env_value) {
    std::string log_level(gxf_log_env_value);
    std::transform(log_level.begin(), log_level.end(), log_level.begin(), [](unsigned char c) {
      return std::toupper(c);
    });
    if (log_level == "TRACE") {
      gxf_log_level = nvidia::Severity::VERBOSE;
    } else if (log_level == "DEBUG") {
      gxf_log_level = nvidia::Severity::DEBUG;
    } else if (log_level == "INFO") {
      gxf_log_level = nvidia::Severity::INFO;
    } else if (log_level == "WARN") {
      gxf_log_level = nvidia::Severity::WARNING;
    } else if (log_level == "ERROR") {
      gxf_log_level = nvidia::Severity::ERROR;
    } else if (log_level == "CRITICAL") {
      gxf_log_level = nvidia::Severity::PANIC;
    } else if (log_level == "OFF") {
      gxf_log_level = nvidia::Severity::NONE;
    }
  } else {
    switch (holoscan_log_level) {
      case LogLevel::TRACE:
        gxf_log_level = nvidia::Severity::VERBOSE;
        break;
      case LogLevel::DEBUG:
        gxf_log_level = nvidia::Severity::DEBUG;
        break;
      case LogLevel::INFO:
        gxf_log_level = nvidia::Severity::INFO;
        break;
      case LogLevel::WARN:
        gxf_log_level = nvidia::Severity::WARNING;
        break;
      case LogLevel::ERROR:
        gxf_log_level = nvidia::Severity::ERROR;
        break;
      case LogLevel::CRITICAL:
        gxf_log_level = nvidia::Severity::PANIC;
        break;
      case LogLevel::OFF:
        gxf_log_level = nvidia::Severity::NONE;
    }
  }

  s_gxf_log_level = gxf_log_level;

  nvidia::LoggingFunction = gxf_logging_holoscan_format;
}

GXFExecutor::GXFExecutor(holoscan::Fragment* fragment, bool create_gxf_context)
    : Executor(fragment) {
  if (fragment == nullptr) { throw std::runtime_error("Fragment is nullptr"); }

  if (create_gxf_context) {
    setup_gxf_logging();

    bool trace_enable = AppDriver::get_bool_env_var("HOLOSCAN_ENABLE_PROFILE", false);
    holoscan::profiler::trace(trace_enable);

    Application* application = fragment->application();

    // TODO(gbae): make shared context work
    // Note:: Do not create shared context for now as it can cause segmentation fault while
    // multiple fragments are activated at the same time.
    // We don't have a way to prevent this from happening yet without modifying GXF.

    // if (application) {
    //   GXF_LOG_INFO("Creating a sharing context");
    //   gxf_context_t shared_context = nullptr;
    //   HOLOSCAN_GXF_CALL_FATAL(
    //       GxfGetSharedContext(application->executor().context(), &shared_context));
    //   HOLOSCAN_GXF_CALL_FATAL(GxfContextCreate1(shared_context, &context_));
    // } else {
    auto frag_name_display = fragment_->name();
    if (!frag_name_display.empty()) { frag_name_display = "[" + frag_name_display + "] "; }
    HOLOSCAN_LOG_INFO("{}Creating context", frag_name_display);
    HOLOSCAN_GXF_CALL_FATAL(GxfContextCreate(&context_));
    // }
    own_gxf_context_ = true;
    extension_manager_ = std::make_shared<GXFExtensionManager>(context_);
    // Register extensions for holoscan (GXFWrapper codelet)
    register_extensions();

    // When we use the GXF shared context, entity name collisions can occur if multiple fragments
    // are initialized at the same time.
    // To avoid this, we prefix the entity names with the fragment name.
    if (application != fragment) { entity_prefix_ = fmt::format("{}__", fragment_->name()); }
    HOLOSCAN_LOG_DEBUG("Entity prefix for fragment '{}': '{}'", fragment_->name(), entity_prefix_);
  }
}

GXFExecutor::~GXFExecutor() {
  implicit_broadcast_entities_.clear();
  util_entity_.reset();
  gpu_device_entity_.reset();
  scheduler_entity_.reset();
  network_context_entity_.reset();
  connections_entity_.reset();

  // Deinitialize GXF context only if `own_gxf_context_` is true
  if (own_gxf_context_) {
    auto frag_name_display = fragment_->name();
    if (!frag_name_display.empty()) { frag_name_display = "[" + frag_name_display + "] "; }
    try {
      HOLOSCAN_LOG_INFO("{}Destroying context", frag_name_display);
    } catch (const std::exception& e) {}

    // Unregister signal handlers if any
    try {
      SignalHandler::unregister_signal_handler(context_, SIGINT);
      SignalHandler::unregister_signal_handler(context_, SIGTERM);
    } catch (const std::exception& e) {
      try {
        HOLOSCAN_LOG_ERROR("Failed to unregister signal handlers: {}", e.what());
      } catch (const std::exception& e) {}
    }
    try {
      HOLOSCAN_GXF_CALL(GxfContextDestroy(context_));
    } catch (const std::exception& e) {}
  }

  // Delete GXF Holoscan Extension
  if (gxf_holoscan_extension_) {
    delete gxf_holoscan_extension_;
    gxf_holoscan_extension_ = nullptr;
  }
}

void GXFExecutor::initialize_gxf_resources(
    std::unordered_map<std::string, std::shared_ptr<Resource>>& resources, gxf_uid_t eid,
    std::shared_ptr<nvidia::gxf::GraphEntity> graph_entity) {
  for (const auto& [name, resource] : resources) {
    // Note: native resources are only supported on Operator, not for NetworkContext or Scheduler
    auto gxf_resource = std::dynamic_pointer_cast<gxf::GXFResource>(resource);
    // Initialize GXF component if it is not already initialized.
    if (gxf_resource->gxf_context() == nullptr) {
      gxf_resource->fragment(fragment());
      if (graph_entity) { gxf_resource->gxf_graph_entity(graph_entity); }
      gxf_resource->gxf_eid(eid);  // set GXF entity id
      gxf_resource->initialize();
    } else {
      HOLOSCAN_LOG_ERROR("Resource '{}' is not a holoscan::gxf::GXFResource and will be ignored",
                         name);
    }
  }
}

void GXFExecutor::add_operator_to_entity_group(gxf_context_t context, gxf_uid_t entity_group_gid,
                                               std::shared_ptr<Operator> op) {
  auto graph_entity = op->graph_entity();
  if (!graph_entity) {
    HOLOSCAN_LOG_ERROR("null GraphEntity found during add_operator_to_entity_group");
    return;
  }
  gxf_uid_t op_eid = graph_entity->eid();
  HOLOSCAN_LOG_DEBUG("Adding operator eid '{}' to entity group '{}'", op_eid, entity_group_gid);
  HOLOSCAN_GXF_CALL_FATAL(GxfUpdateEntityGroup(context, entity_group_gid, op_eid));
}

void GXFExecutor::run(OperatorGraph& graph) {
  if (!initialize_gxf_graph(graph)) {
    HOLOSCAN_LOG_ERROR("Failed to initialize GXF graph");
    return;
  }

  // Note that run_gxf_graph() can raise an exception.
  run_gxf_graph();
}

std::future<void> GXFExecutor::run_async(OperatorGraph& graph) {
  if (!is_gxf_graph_initialized_) { initialize_gxf_graph(graph); }

  return std::async(std::launch::async, [this, &graph]() {
    // Note that run_gxf_graph() can raise an exception.
    this->run_gxf_graph();
  });
}

void GXFExecutor::interrupt() {
  if (context_) {
    gxf_result_t code = GxfGraphInterrupt(context_);
    if (code != GXF_SUCCESS) {
      HOLOSCAN_LOG_ERROR("GxfGraphInterrupt Error: {}", GxfResultStr(code));
    }
  }
}

void GXFExecutor::context(void* context) {
  context_ = context;
  extension_manager_ = std::make_shared<GXFExtensionManager>(context_);
}

std::shared_ptr<ExtensionManager> GXFExecutor::extension_manager() {
  return extension_manager_;
}

namespace {
/* @brief Utility function used internally by the GXFExecutor::create_input_port static method.
 *
 * This function will only be called in the case of a GXF application that is wrapping a
 * holoscan::Operator as a GXF codelet (as in examples/wrap_operator_as_gxf_extension).
 *
 * It will update the native operator's connectors to use the existing GXF receiver.
 *
 * Note: cannot use GXF GraphEntity C++ APIs here as the Operator here wraps a codelet which does
 * not have a GraphEntity data member.
 */
void bind_input_port(Fragment* fragment, gxf_context_t gxf_context, gxf_uid_t eid, IOSpec* io_spec,
                     const char* rx_name, IOSpec::ConnectorType rx_type, Operator* op) {
  // Can't currently use GraphEntity API for this OperatorWrapper/bind_port code path
  if (rx_type != IOSpec::ConnectorType::kDefault) {
    // TODO: update bind_port code path for types other than ConnectorType::kDefault
    throw std::runtime_error(fmt::format(
        "Unable to support types other than ConnectorType::kDefault (rx_name: '{}')", rx_name));
  }
  const char* entity_name = "";
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentName(gxf_context, eid, &entity_name));

  gxf_tid_t receiver_find_tid{};
  HOLOSCAN_GXF_CALL_FATAL(
      GxfComponentTypeId(gxf_context, "nvidia::gxf::Receiver", &receiver_find_tid));

  gxf_uid_t receiver_cid = 0;
  HOLOSCAN_GXF_CALL_FATAL(
      GxfComponentFind(gxf_context, eid, receiver_find_tid, rx_name, nullptr, &receiver_cid));

  gxf_tid_t receiver_tid{};
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentType(gxf_context, receiver_cid, &receiver_tid));

  gxf_tid_t double_buffer_receiver_tid{};

  if (fragment->data_flow_tracker()) {
    HOLOSCAN_GXF_CALL_FATAL(GxfComponentTypeId(
        gxf_context, "holoscan::AnnotatedDoubleBufferReceiver", &double_buffer_receiver_tid));
  } else {
    HOLOSCAN_GXF_CALL_FATAL(GxfComponentTypeId(
        gxf_context, "nvidia::gxf::DoubleBufferReceiver", &double_buffer_receiver_tid));
  }

  if (receiver_tid == double_buffer_receiver_tid) {
    // It could be made more succinct by casting appropriately at the
    // std::make_shared call, but, I don't have an example to test if it is working
    if (fragment->data_flow_tracker()) {
      holoscan::AnnotatedDoubleBufferReceiver* double_buffer_receiver_ptr = nullptr;
      HOLOSCAN_GXF_CALL_FATAL(
          GxfComponentPointer(gxf_context,
                              receiver_cid,
                              receiver_tid,
                              reinterpret_cast<void**>(&double_buffer_receiver_ptr)));

      if (double_buffer_receiver_ptr) {
        auto receiver =
            std::make_shared<holoscan::DoubleBufferReceiver>(rx_name, double_buffer_receiver_ptr);
        // Set the existing DoubleBufferReceiver for this input
        io_spec->connector(receiver);
        double_buffer_receiver_ptr->op(op);
      } else {
        HOLOSCAN_LOG_ERROR(
            "Unable to get AnnotatedDoubleBufferReceiver pointer for the handle: '{}' in '{}' "
            "entity",
            rx_name,
            entity_name);
      }
    } else {
      nvidia::gxf::DoubleBufferReceiver* double_buffer_receiver_ptr = nullptr;
      GxfComponentPointer(gxf_context,
                          receiver_cid,
                          receiver_tid,
                          reinterpret_cast<void**>(&double_buffer_receiver_ptr));

      if (double_buffer_receiver_ptr) {
        auto receiver =
            std::make_shared<holoscan::DoubleBufferReceiver>(rx_name, double_buffer_receiver_ptr);
        // Set the existing DoubleBufferReceiver for this input
        io_spec->connector(receiver);
      } else {
        HOLOSCAN_LOG_ERROR(
            "Unable to get DoubleBufferReceiver pointer for the handle: '{}' in '{}' entity",
            rx_name,
            entity_name);
      }
    }
  } else {
    HOLOSCAN_LOG_ERROR(
        "Unsupported GXF receiver type for the handle: '{}' in '{}' entity", rx_name, entity_name);
  }
  return;
}
}  // namespace

void GXFExecutor::create_input_port(Fragment* fragment, gxf_context_t gxf_context, gxf_uid_t eid,
                                    IOSpec* io_spec, bool bind_port, Operator* op) {
  const char* rx_name = io_spec->name().c_str();  // input port name
  auto rx_type = io_spec->connector_type();

  if (fragment->data_flow_tracker()) {
    if ((rx_type != IOSpec::ConnectorType::kDefault) &&
        (rx_type != IOSpec::ConnectorType::kDoubleBuffer)) {
      throw std::runtime_error(
          "Currently the data flow tracking feature requires ConnectorType::kDefault or "
          "ConnectorType::kDoubleBuffer.");
    }
  }
  auto graph_entity = op->graph_entity();

  // If this executor is used by OperatorWrapper (bind_port == true) to wrap Native Operator,
  // then we need to call `bind_input_port` to set the existing GXF Receiver for this input.
  if (bind_port) {
    bind_input_port(fragment, gxf_context, eid, io_spec, rx_name, rx_type, op);
    return;
  }

  int64_t queue_size = io_spec->queue_size();
  if (queue_size == static_cast<int64_t>(IOSpec::kAnySize)) {
    // Do not create a receiver for this as we are using the parameterized receiver method.
    return;
  }

  // If the queue size is 0 (IOSpec::kPrecedingCount), then we need to calculate the default queue
  // size based on the number of preceding connections.
  if (queue_size == static_cast<int64_t>(IOSpec::kPrecedingCount)) {
    auto& flow_graph = fragment->graph();
    auto node = flow_graph.find_node(op->name());
    if (node) {
      // Count the number of connections to this input port
      int connection_count = 0;
      auto prev_nodes = flow_graph.get_previous_nodes(node);
      for (const auto& prev_node : prev_nodes) {
        auto port_map = flow_graph.get_port_map(prev_node, node);
        if (port_map.has_value()) {
          const auto& port_map_val = port_map.value();
          // Iterate over the set of target ports
          for (const auto& [preceding_out_port_name, target_ports] : *port_map_val) {
            // Count the number of connections to this input port.
            connection_count += target_ports.count(rx_name);
          }
        }

        // Set the queue size to the number of preceding connections
        queue_size = connection_count;
      }
    } else {
      HOLOSCAN_LOG_ERROR("Failed to find node for operator '{}'", op->name());
      throw std::runtime_error(fmt::format("Failed to find node for operator '{}'", op->name()));
    }
  }

  if (queue_size < 1) {
    HOLOSCAN_LOG_ERROR(
        "Invalid queue size: {} (op: '{}', input port: '{}')", queue_size, op->name(), rx_name);
    throw std::runtime_error(fmt::format(
        "Invalid queue size: {} (op: '{}', input port: '{}')", queue_size, op->name(), rx_name));
  }

  auto connector = std::dynamic_pointer_cast<Receiver>(io_spec->connector());
  if (connector && (connector->gxf_cptr() != nullptr)) {
    auto gxf_receiver = std::dynamic_pointer_cast<holoscan::gxf::GXFResource>(connector);
    if (gxf_receiver && graph_entity) {
      gxf_receiver->gxf_eid(graph_entity->eid());
      gxf_receiver->gxf_graph_entity(graph_entity);
    }
  } else {
    // Create Receiver component for this input
    std::shared_ptr<Receiver> rx_resource;
    switch (rx_type) {
      case IOSpec::ConnectorType::kDefault:
        HOLOSCAN_LOG_DEBUG("creating input port using DoubleBufferReceiver");
        rx_resource = std::make_shared<DoubleBufferReceiver>();
        // Set the capacity of the DoubleBufferReceiver with the queue_size
        rx_resource->add_arg(Arg("capacity", queue_size));
        if (fragment->data_flow_tracker()) {
          std::dynamic_pointer_cast<DoubleBufferReceiver>(rx_resource)->track();
        }
        break;
      case IOSpec::ConnectorType::kDoubleBuffer:
        rx_resource = std::dynamic_pointer_cast<Receiver>(io_spec->connector());
        if (fragment->data_flow_tracker()) {
          std::dynamic_pointer_cast<DoubleBufferReceiver>(rx_resource)->track();
        }
        break;
      case IOSpec::ConnectorType::kUCX:
        rx_resource = std::dynamic_pointer_cast<Receiver>(io_spec->connector());
        if (fragment->data_flow_tracker()) {
          HOLOSCAN_LOG_ERROR("data flow tracking not implemented for UCX ports");
        }
        break;
      default:
        HOLOSCAN_LOG_ERROR("Unsupported GXF connector_type: '{}'", static_cast<int>(rx_type));
    }
    rx_resource->name(rx_name);
    rx_resource->fragment(fragment);
    auto rx_spec = std::make_shared<ComponentSpec>(fragment);
    rx_resource->setup(*rx_spec);
    rx_resource->spec(std::move(rx_spec));

    // Note: had to make sure GXFComponent calls addComponent and not addReceiver or addTransmitter
    //       or errors will occur as follows:
    // [error] [component.hpp:160] Expression 'parameter_registrar_->getComponentParameterInfoPtr(
    //             tid, key)' failed with error 'GXF_ENTITY_COMPONENT_NOT_FOUND'.
    // [error] [graph_entity.cpp:52] Expression 'codelet_->getParameterInfo(rx_name)' failed with
    //             error 'GXF_ENTITY_COMPONENT_NOT_FOUND'.
    // [error] [gxf_component.cpp:112] Failed to add component 'values:27' of type:
    //             'nvidia::gxf::DoubleBufferReceiver'
    // [info] [gxf_component.cpp:119] Initializing component '__condition_input__1' in entity
    //             '370' via GxfComponentAdd
    // [error] [gxf_condition.cpp:97] GXF call ::holoscan::gxf::GXFParameterAdaptor::set_param(
    //              gxf_context_, gxf_cid_, key.c_str(), param_wrap) in line 97 of file

    // Add to the same entity as the operator and initialize
    // Note: it is important that GXFComponent calls addComponent and not addTransmitter for this
    rx_resource->add_to_graph_entity(op);

    if (fragment->data_flow_tracker()) {
      holoscan::AnnotatedDoubleBufferReceiver* dbl_ptr;
      switch (rx_type) {
        case IOSpec::ConnectorType::kDefault:
        case IOSpec::ConnectorType::kDoubleBuffer:
          dbl_ptr =
              reinterpret_cast<holoscan::AnnotatedDoubleBufferReceiver*>(rx_resource->gxf_cptr());
          dbl_ptr->op(op);
          break;
        case IOSpec::ConnectorType::kUCX:
          HOLOSCAN_LOG_ERROR("UCX-based receiver doesn't currently support data flow tracking");
          break;
        default:
          HOLOSCAN_LOG_ERROR(
              "Annotated data flow tracking not implemented for GXF "
              "connector_type: '{}'",
              static_cast<int>(rx_type));
      }
    }

    // Set the connector for this input
    connector = rx_resource;
    io_spec->connector(connector);
  }

  // Set the default scheduling term for this input
  if (io_spec->conditions().empty()) {
    ArgList args;
    args.add(Arg("min_size") = static_cast<uint64_t>(queue_size));
    io_spec->condition(
        ConditionType::kMessageAvailable, Arg("receiver") = io_spec->connector(), args);
  }

  // Initialize conditions for this input
  int condition_index = 0;
  for (const auto& [condition_type, condition] : io_spec->conditions()) {
    ++condition_index;
    switch (condition_type) {
      case ConditionType::kMessageAvailable: {
        std::shared_ptr<MessageAvailableCondition> message_available_condition =
            std::dynamic_pointer_cast<MessageAvailableCondition>(condition);
        // Note: GraphEntity::addSchedulingTerm requires a unique name here
        std::string cond_name =
            fmt::format("__{}_{}_cond_{}", op->name(), rx_name, condition_index);
        message_available_condition->receiver(connector);
        message_available_condition->name(cond_name);
        message_available_condition->fragment(fragment);
        auto rx_condition_spec = std::make_shared<ComponentSpec>(fragment);
        message_available_condition->setup(*rx_condition_spec);
        message_available_condition->spec(std::move(rx_condition_spec));
        // Add to the same entity as the operator and initialize
        message_available_condition->add_to_graph_entity(op);
        break;
      }
      case ConditionType::kExpiringMessageAvailable: {
        std::shared_ptr<ExpiringMessageAvailableCondition> expiring_message_available_condition =
            std::dynamic_pointer_cast<ExpiringMessageAvailableCondition>(condition);
        // Note: GraphEntity::addSchedulingTerm requires a unique name here
        std::string cond_name =
            fmt::format("__{}_{}_cond_{}", op->name(), rx_name, condition_index);
        expiring_message_available_condition->receiver(connector);
        expiring_message_available_condition->name(cond_name);
        expiring_message_available_condition->fragment(fragment);
        auto rx_condition_spec = std::make_shared<ComponentSpec>(fragment);
        expiring_message_available_condition->setup(*rx_condition_spec);
        expiring_message_available_condition->spec(std::move(rx_condition_spec));
        // Add to the same entity as the operator and initialize
        expiring_message_available_condition->add_to_graph_entity(op);
        break;
      }
      case ConditionType::kNone:
        // No condition
        break;
      default:
        throw std::runtime_error("Unsupported condition type");  // TODO: use std::expected
    }
  }
}

namespace {
/* @brief Utility function used internally by the GXFExecutor::create_output_port static method.
 *
 * This function will only be called in the case of a GXF application that is wrapping a
 * holoscan::Operator as a GXF codelet (as in examples/wrap_operator_as_gxf_extension).
 *
 * It will update the native operator's connectors to use the existing GXF transmitter/
 *
 * Note: cannot use GXF GraphEntity C++ APIs here as the Operator here wraps a codelet which does
 * not have a GraphEntity data member.
 */
void bind_output_port(Fragment* fragment, gxf_context_t gxf_context, gxf_uid_t eid, IOSpec* io_spec,
                      const char* tx_name, IOSpec::ConnectorType tx_type, Operator* op) {
  if (tx_type != IOSpec::ConnectorType::kDefault) {
    // TODO: update bind_port code path for types other than ConnectorType::kDefault
    throw std::runtime_error(fmt::format(
        "Unable to support types other than ConnectorType::kDefault (tx_name: '{}')", tx_name));
  }
  const char* entity_name = "";
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentName(gxf_context, eid, &entity_name));

  gxf_tid_t transmitter_find_tid{};
  HOLOSCAN_GXF_CALL_FATAL(
      GxfComponentTypeId(gxf_context, "nvidia::gxf::Transmitter", &transmitter_find_tid));

  gxf_uid_t transmitter_cid = 0;
  HOLOSCAN_GXF_CALL_FATAL(
      GxfComponentFind(gxf_context, eid, transmitter_find_tid, tx_name, nullptr, &transmitter_cid));

  gxf_tid_t transmitter_tid{};
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentType(gxf_context, transmitter_cid, &transmitter_tid));

  gxf_tid_t double_buffer_transmitter_tid{};

  if (fragment->data_flow_tracker()) {
    HOLOSCAN_GXF_CALL_FATAL(GxfComponentTypeId(
        gxf_context, "holoscan::AnnotatedDoubleBufferTransmitter", &double_buffer_transmitter_tid));
  } else {
    HOLOSCAN_GXF_CALL_FATAL(GxfComponentTypeId(
        gxf_context, "nvidia::gxf::DoubleBufferTransmitter", &double_buffer_transmitter_tid));
  }

  if (transmitter_tid == double_buffer_transmitter_tid) {
    if (fragment->data_flow_tracker()) {
      holoscan::AnnotatedDoubleBufferTransmitter* double_buffer_transmitter_ptr = nullptr;
      HOLOSCAN_GXF_CALL_FATAL(
          GxfComponentPointer(gxf_context,
                              transmitter_cid,
                              transmitter_tid,
                              reinterpret_cast<void**>(&double_buffer_transmitter_ptr)));

      if (double_buffer_transmitter_ptr) {
        auto transmitter = std::make_shared<holoscan::DoubleBufferTransmitter>(
            tx_name, double_buffer_transmitter_ptr);
        // Set the existing DoubleBufferTransmitter for this output
        io_spec->connector(transmitter);
        double_buffer_transmitter_ptr->op(op);
      } else {
        HOLOSCAN_LOG_ERROR(
            "Unable to get AnnotatedDoubleBufferTransmitter pointer for the handle: '{}' in '{}' "
            "entity",
            tx_name,
            entity_name);
      }
    } else {
      nvidia::gxf::DoubleBufferTransmitter* double_buffer_transmitter_ptr = nullptr;
      GxfComponentPointer(gxf_context,
                          transmitter_cid,
                          transmitter_tid,
                          reinterpret_cast<void**>(&double_buffer_transmitter_ptr));

      if (double_buffer_transmitter_ptr) {
        auto transmitter = std::make_shared<holoscan::DoubleBufferTransmitter>(
            tx_name, double_buffer_transmitter_ptr);
        // Set the existing DoubleBufferTransmitter for this output
        io_spec->connector(transmitter);
      } else {
        HOLOSCAN_LOG_ERROR(
            "Unable to get DoubleBufferTransmitter pointer for the handle: '{}' in '{}' entity",
            tx_name,
            entity_name);
      }
    }
  } else {
    HOLOSCAN_LOG_ERROR("Unsupported GXF transmitter type for the handle: '{}' in '{}' entity",
                       tx_name,
                       entity_name);
  }
}
}  // namespace

void GXFExecutor::create_output_port(Fragment* fragment, gxf_context_t gxf_context, gxf_uid_t eid,
                                     IOSpec* io_spec, bool bind_port, Operator* op) {
  const char* tx_name = io_spec->name().c_str();
  auto tx_type = io_spec->connector_type();

  if (fragment->data_flow_tracker()) {
    if ((tx_type != IOSpec::ConnectorType::kDefault) &&
        (tx_type != IOSpec::ConnectorType::kDoubleBuffer)) {
      throw std::runtime_error(
          "Currently the data flow tracking feature requires ConnectorType::kDefault or "
          "ConnectorType::kDoubleBuffer.");
    }
  }
  auto graph_entity = op->graph_entity();
  // If this executor is used by OperatorWrapper (bind_port == true) to wrap Native Operator,
  // then we need to call `bind_output_port` to set the existing GXF Transmitter for this output.
  if (bind_port) {
    bind_output_port(fragment, gxf_context, eid, io_spec, tx_name, tx_type, op);
    return;
  }

  auto connector = std::dynamic_pointer_cast<Transmitter>(io_spec->connector());
  if (connector && (connector->gxf_cptr() != nullptr)) {
    auto gxf_transmitter = std::dynamic_pointer_cast<holoscan::gxf::GXFResource>(connector);
    if (gxf_transmitter && graph_entity) {
      gxf_transmitter->gxf_eid(graph_entity->eid());
      gxf_transmitter->gxf_graph_entity(graph_entity);
    }
  } else {
    // Create Transmitter component for this output
    std::shared_ptr<Transmitter> tx_resource;
    switch (tx_type) {
      case IOSpec::ConnectorType::kDefault:
        HOLOSCAN_LOG_DEBUG("creating output port using DoubleBufferReceiver");
        tx_resource = std::make_shared<DoubleBufferTransmitter>();
        if (fragment->data_flow_tracker()) {
          std::dynamic_pointer_cast<DoubleBufferTransmitter>(tx_resource)->track();
        }
        break;
      case IOSpec::ConnectorType::kDoubleBuffer:
        tx_resource = std::dynamic_pointer_cast<Transmitter>(io_spec->connector());
        if (fragment->data_flow_tracker()) {
          std::dynamic_pointer_cast<DoubleBufferTransmitter>(tx_resource)->track();
        }
        break;
      case IOSpec::ConnectorType::kUCX:
        tx_resource = std::dynamic_pointer_cast<Transmitter>(io_spec->connector());
        if (fragment->data_flow_tracker()) {
          HOLOSCAN_LOG_ERROR("data flow tracking not implemented for UCX ports");
        }
        break;
      default:
        HOLOSCAN_LOG_ERROR("Unsupported GXF connector_type: '{}'", static_cast<int>(tx_type));
    }
    tx_resource->name(tx_name);
    tx_resource->fragment(fragment);
    auto tx_spec = std::make_shared<ComponentSpec>(fragment);
    tx_resource->setup(*tx_spec);
    tx_resource->spec(std::move(tx_spec));
    // add to the same entity as the operator and initialize
    // Note: it is important that GXFComponent calls addComponent and not addTransmitter for this
    tx_resource->add_to_graph_entity(op);

    if (fragment->data_flow_tracker()) {
      holoscan::AnnotatedDoubleBufferTransmitter* dbl_ptr;
      switch (tx_type) {
        case IOSpec::ConnectorType::kDefault:
        case IOSpec::ConnectorType::kDoubleBuffer:
          dbl_ptr = reinterpret_cast<holoscan::AnnotatedDoubleBufferTransmitter*>(
              tx_resource->gxf_cptr());
          dbl_ptr->op(op);
          break;
        case IOSpec::ConnectorType::kUCX:
          HOLOSCAN_LOG_ERROR("UCX-based receiver doesn't currently support data flow tracking");
          break;
        default:
          HOLOSCAN_LOG_ERROR(
              "Annotated data flow tracking not implemented for GXF "
              "connector_type: '{}'",
              static_cast<int>(tx_type));
      }
    }

    // Set the connector for this output
    connector = tx_resource;
    io_spec->connector(connector);
  }

  // Set the default scheduling term for this output
  // For the UCX connector, we shouldn't set kDownstreamMessageAffordable condition.
  if (io_spec->conditions().empty() && (tx_type != IOSpec::ConnectorType::kUCX)) {
    io_spec->condition(ConditionType::kDownstreamMessageAffordable,
                       Arg("transmitter") = io_spec->connector(),
                       Arg("min_size") = 1UL);
  }

  // Initialize conditions for this output
  int condition_index = 0;
  for (const auto& [condition_type, condition] : io_spec->conditions()) {
    ++condition_index;
    switch (condition_type) {
      case ConditionType::kDownstreamMessageAffordable: {
        std::shared_ptr<DownstreamMessageAffordableCondition> downstream_msg_affordable_condition =
            std::dynamic_pointer_cast<DownstreamMessageAffordableCondition>(condition);
        // Note: GraphEntity::addSchedulingTerm requires a unique name here
        std::string cond_name =
            fmt::format("__{}_{}_cond_{}", op->name(), tx_name, condition_index);
        downstream_msg_affordable_condition->transmitter(connector);
        downstream_msg_affordable_condition->name(cond_name);
        downstream_msg_affordable_condition->fragment(fragment);
        auto tx_condition_spec = std::make_shared<ComponentSpec>(fragment);
        downstream_msg_affordable_condition->setup(*tx_condition_spec);
        downstream_msg_affordable_condition->spec(std::move(tx_condition_spec));
        // add to the same entity as the operator and initialize
        downstream_msg_affordable_condition->add_to_graph_entity(op);
        break;
      }
      case ConditionType::kNone:
        // No condition
        break;
      default:
        throw std::runtime_error("Unsupported condition type");  // TODO: use std::expected
    }
  }
}

namespace {  // unnamed namespace for implementation details

using ConnectionMapType = std::unordered_map<
    holoscan::OperatorGraph::NodeType,
    std::unordered_map<std::string, std::vector<std::shared_ptr<holoscan::ConnectionItem>>>>;

ConnectionMapType generate_connection_map(
    OperatorGraph& graph,
    std::vector<std::shared_ptr<holoscan::ConnectionItem>>& connection_items) {
  // Construct name-to-operator map
  std::unordered_map<std::string, holoscan::OperatorGraph::NodeType> name_to_op;
  for (const auto& op : graph.get_nodes()) { name_to_op[op->name()] = op; }

  /// Construct connection item map
  ConnectionMapType connection_map;
  for (auto& connection_item : connection_items) {
    auto& connection_item_name = connection_item->name;
    auto [operator_name, port_name] = Operator::parse_port_name(connection_item_name);
    auto op = name_to_op[operator_name];
    auto op_find = connection_map.find(op);
    if (op_find == connection_map.end()) {
      connection_map[op] =
          std::unordered_map<std::string, std::vector<std::shared_ptr<holoscan::ConnectionItem>>>();
    }

    auto& port_map = connection_map[op];
    auto port_find = port_map.find(port_name);
    if (port_find == port_map.end()) {
      port_map[port_name] = std::vector<std::shared_ptr<holoscan::ConnectionItem>>();
    }

    auto& op_port_connections = port_map[port_name];
    op_port_connections.push_back(connection_item);
  }
  return connection_map;
}

/**
 * @brief Populate virtual_ops vector and add corresponding connections to fragment.
 *
 * When VirtualTransmitterOp is created, UcxTransmitter's 'local_address' is set by the environment
 * variable 'HOLOSCAN_UCX_SOURCE_ADDRESS' so that UcxTransmitter can create a UCX client endpoint
 * using the local IP address.
 * HOLOSCAN_UCX_SOURCE_ADDRESS may or may not have a port number (`<ip>:<port>`), but the port
 * number would be ignored because there can be multiple UcxTransmitters in the fragments that are
 * running on the same node, so specifying the port number is error-prone.
 */
void create_virtual_operators_and_connections(
    Fragment* fragment, const ConnectionMapType& connection_map,
    std::vector<std::shared_ptr<ops::VirtualOperator>>& virtual_ops) {
  // Get the local source address from the environment variable 'HOLOSCAN_UCX_SOURCE_ADDRESS'.
  std::string source_ip = "0.0.0.0";

  const char* source_address = std::getenv("HOLOSCAN_UCX_SOURCE_ADDRESS");
  if (source_address != nullptr && source_address[0] != '\0') {
    HOLOSCAN_LOG_DEBUG("The environment variable 'HOLOSCAN_UCX_SOURCE_ADDRESS' is set to '{}'",
                       source_address);
    std::string source_address_str(source_address);
    auto [ip, _] = holoscan::CLIOptions::parse_address(source_address_str, "0.0.0.0", "0");
    // Convert port string 'port' to int32_t.
    source_ip = std::move(ip);
  }

  for (auto& [op, port_map] : connection_map) {
    for (auto& [port_name, connections] : port_map) {
      int connection_index = 0;
      for (auto& connection : connections) {
        auto io_type = connection->io_type;

        std::shared_ptr<ops::VirtualOperator> virtual_op;
        if (io_type == IOSpec::IOType::kOutput) {
          // Update local_address and local_port of the UcxTransmitter based on
          // the `source_address` from the environment variable
          // 'HOLOSCAN_UCX_SOURCE_ADDRESS' (issue 4233845).
          // The `source_port` would be ignored.
          HOLOSCAN_LOG_DEBUG("Updating 'local_address' of the UcxTransmitter in '{}.{}' to '{}'",
                             fragment->name(),
                             connection->name,
                             source_ip);
          connection->args.add(Arg("local_address", source_ip));
          virtual_op = std::make_shared<ops::VirtualTransmitterOp>(
              port_name, IOSpec::ConnectorType::kUCX, connection->args);
        } else {
          virtual_op = std::make_shared<ops::VirtualReceiverOp>(
              port_name, IOSpec::ConnectorType::kUCX, connection->args);
        }
        virtual_ops.push_back(virtual_op);

        virtual_op->name(
            fmt::format("virtual_{}_{}_{}", op->name(), port_name, connection_index++));
        virtual_op->fragment(fragment);
        auto spec = std::make_shared<OperatorSpec>(fragment);
        virtual_op->setup(*spec.get());
        virtual_op->spec(spec);

        if (io_type == IOSpec::IOType::kOutput) {
          // Connect op.port_name to virtual_op.port_name
          fragment->add_flow(op, virtual_op, {{port_name, port_name}});
        } else {
          int param_index = -1;
          // If we cannot find the port_name in the op's input or IOSpec's queue size equals
          // IOSpec::kAnySize, it means that
          // the port name is the parameter name and the parameter type is
          // 'std::vector<holoscan::IOSpec*>'.
          // In this case, we need to use the indexed input port name to avoid
          // name conflict ('<port name>:<index>').
          auto op_spec = op->spec();
          auto& op_spec_inputs = op_spec->inputs();
          auto op_spec_input_it = op_spec_inputs.find(port_name);
          if (op_spec_input_it == op_spec_inputs.end() ||
              (op_spec_input_it->second->queue_size() == static_cast<int64_t>(IOSpec::kAnySize))) {
            auto& op_params = op_spec->params();
            const std::any& any_value = op_params[port_name].value();
            auto& param = *std::any_cast<Parameter<std::vector<holoscan::IOSpec*>>*>(any_value);
            auto iospec_vector = param.try_get();
            if (iospec_vector == std::nullopt) {
              param_index = 0;
            } else {
              param_index = iospec_vector.value().size();
            }
          }

          // Create and insert a forward operator to connect virtual_op.port_name to op.port_name
          const std::string forward_op_name =
              param_index == -1
                  ? fmt::format("forward_{}_{}", op->name(), port_name)
                  : fmt::format("forward_{}_{}:{}", op->name(), port_name, param_index);
          auto forward_op = fragment->make_operator<ops::ForwardOp>(forward_op_name);
          auto& in_spec =
              forward_op->spec()->inputs()["in"];  // get the input spec of the forward op

          // Create the connector for in_spec from the virtual_op
          in_spec->connector(virtual_op->connector_type(), virtual_op->arg_list());

          // Connect virtual_op.port_name to forward_op.in
          fragment->add_flow(virtual_op, forward_op, {{port_name, "in"}});

          // Connect forward_op.out  to op.port_name
          fragment->add_flow(forward_op, op, {{"out", port_name}});
        }
      }
    }
  }
}

void connect_ucx_transmitters_to_virtual_ops(
    Fragment* fragment, std::vector<std::shared_ptr<ops::VirtualOperator>>& virtual_ops) {
  auto& graph = fragment->graph();

  // If a port corresponding to the VirtualTransmitterOp is not connected to multiple destination
  // ports, we can create UCX transmitter directly.
  for (auto& virtual_op : virtual_ops) {
    // If virtual_op is VirtualTransmitterOp
    switch (virtual_op->io_type()) {
      case IOSpec::IOType::kOutput: {
        // It should have only one predecessor.
        auto last_transmitter_op = graph.get_previous_nodes(virtual_op)[0];
        auto& port_name = virtual_op->port_name();

        // Count connections from <operator name>.<port name> to the corresponding operators
        // including virtual operators. This is used to determine if direct UcxTransmitter should be
        // created or not.
        int connection_count = 0;
        auto connected_ops = graph.get_next_nodes(last_transmitter_op);
        for (auto& op : connected_ops) {
          auto port_map = graph.get_port_map(last_transmitter_op, op);
          if (port_map.has_value()) {
            const auto& port_map_val = port_map.value();
            connection_count += port_map_val->count(port_name);
          }
        }

        if (connection_count == 1) {
          auto& out_spec = last_transmitter_op->spec()->outputs()[port_name];
          // Create the connector for out_spec from the virtual_op
          out_spec->connector(virtual_op->connector_type(), virtual_op->arg_list());
        }
      } break;
      case IOSpec::IOType::kInput: {
        // Nothing to do. Handled in create_virtual_operators_and_connections().
      } break;
    }
  }
}

}  // unnamed namespace

gxf_result_t GXFExecutor::add_connection(gxf_uid_t source_cid, gxf_uid_t target_cid) {
  gxf_result_t code;

  auto connection = connections_entity_->addComponent("nvidia::gxf::Connection");
  if (!connection) {
    HOLOSCAN_LOG_ERROR(
        "Failed to add nvidia::gxf::Connection between source cid['{}'] and target cid['{}']",
        source_cid,
        target_cid);
    return GXF_FAILURE;
  }
  // Use C API instead of Connection::setReceiver and Connection::setTransmitter since we don't
  // already have Handle<Resource> for source and target.
  gxf_uid_t connect_cid = connection->cid();
  gxf_context_t context = connections_entity_->context();
  HOLOSCAN_GXF_CALL(GxfParameterSetHandle(context, connect_cid, "source", source_cid));
  code = GxfParameterSetHandle(context, connect_cid, "target", target_cid);
  return code;
}

void GXFExecutor::connect_broadcast_to_previous_op(
    const BroadcastEntityMapType& broadcast_entities, holoscan::OperatorGraph::NodeType op,
    holoscan::OperatorGraph::NodeType prev_op, holoscan::OperatorGraph::EdgeDataType port_map_val) {
  auto op_type = op->operator_type();

  // counter to ensure unique broadcast component names as required by nvidia::gxf::GraphEntity
  static uint32_t btx_count = 0;

  HOLOSCAN_LOG_DEBUG(
      "connecting broadcast codelet from previous_op {} to op {}", prev_op->name(), op->name());

  // A Broadcast component was added for prev_op
  for (const auto& [port_name, broadcast_entity] : broadcast_entities.at(prev_op)) {
    // Find the Broadcast component's source port name in the port-map.
    if (port_map_val->find(port_name) != port_map_val->end()) {
      // There is an output port of the prev_op that is associated with a Broadcast component.
      // Create a transmitter for the current Operator's input port in the Broadcast
      // entity and add a connection from the transmitter to the current Operator's input
      // port.
      auto target_ports = port_map_val->at(port_name);
      for (const auto& target_port : target_ports) {
        // Create a Transmitter in the Broadcast entity.
        auto& prev_op_io_spec = prev_op->spec()->outputs()[port_name];
        auto prev_connector_type = prev_op_io_spec->connector_type();
        auto prev_connector = prev_op_io_spec->connector();

        // If prev_connector_type is kDefault, decide which transmitter to create based on
        // the current Operator's type.
        if (prev_connector_type == IOSpec::ConnectorType::kDefault) {
          if (op_type == Operator::OperatorType::kVirtual) {
            prev_connector_type = IOSpec::ConnectorType::kUCX;
          } else {
            prev_connector_type = IOSpec::ConnectorType::kDoubleBuffer;
          }
        }

        // Find prev_connector's capacity and policy.
        uint64_t prev_connector_capacity = 1;
        uint64_t prev_connector_policy = holoscan::gxf::get_default_queue_policy();

        // Create a transmitter based on the prev_connector_type.
        switch (prev_connector_type) {
          case IOSpec::ConnectorType::kDoubleBuffer: {
            // We don't create a AnnotatedDoubleBufferTransmitter even if DFFT is on because
            // we don't want to annotate a message at the Broadcast component.

            // Clone the capacity and policy from previous connector
            auto prev_ucx_connector =
                std::dynamic_pointer_cast<DoubleBufferTransmitter>(prev_connector);
            if (prev_ucx_connector) {
              prev_connector_capacity = prev_ucx_connector->capacity_;
              prev_connector_capacity = prev_ucx_connector->policy_;
            } else {
              HOLOSCAN_LOG_ERROR(
                  "Failed to cast connector to DoubleBufferTransmitter, using default capacity and "
                  "policy");
            }

            // Note: have to use add<T> instead of addTransmitter<T> because the
            //       Transmitter is not a Parameter on the Broadcast codelet.
            std::string btx_name = fmt::format("btx_{}", btx_count);
            auto btx_handle = broadcast_entity->add<nvidia::gxf::DoubleBufferTransmitter>(
                btx_name.c_str(),
                nvidia::gxf::Arg("capacity", prev_connector_capacity),
                nvidia::gxf::Arg("policy", prev_connector_policy));
            if (!btx_handle) {
              HOLOSCAN_LOG_ERROR("Failed to create broadcast transmitter for entity {}",
                                 broadcast_entity->name());
            }
            btx_count += 1;  // increment to ensure unique names

            // 1. Find the output port's condition.
            //    (ConditionType::kDownstreamMessageAffordable)
            std::shared_ptr<holoscan::Condition> prev_condition;
            for (const auto& [condition_type, condition] : prev_op_io_spec->conditions()) {
              if (condition_type == ConditionType::kDownstreamMessageAffordable) {
                prev_condition = condition;
                break;
              }
            }
            HOLOSCAN_LOG_DEBUG(
                "Connected with Broadcast source : {} -> target : {}", port_name, target_port);

            // 2. If it exists, clone it and set it as the transmitter's condition unless
            //    the connector type is kUCX.
            uint64_t prev_min_size = 1;
            if (prev_condition) {
              auto prev_downstream_condition =
                  std::dynamic_pointer_cast<DownstreamMessageAffordableCondition>(prev_condition);
              prev_min_size = prev_downstream_condition->min_size();

              // use add<T> to get the specific Handle so setTransmitter method can be used
              std::string btx_term_name = fmt::format("btx_sched_term_{}", btx_count);
              auto btx_term_handle =
                  broadcast_entity->add<nvidia::gxf::DownstreamReceptiveSchedulingTerm>(
                      btx_term_name.c_str(), nvidia::gxf::Arg("min_size", prev_min_size));
              if (!btx_term_handle) {
                HOLOSCAN_LOG_ERROR(
                    "Failed to create broadcast transmitter scheduling term for entity {}",
                    broadcast_entity->name());
              }
              btx_term_handle->setTransmitter(btx_handle);
            }

            // Get the current Operator's input port
            auto target_gxf_resource = std::dynamic_pointer_cast<GXFResource>(
                op->spec()->inputs()[target_port]->connector());
            gxf_uid_t target_cid = target_gxf_resource->gxf_cid();

            // Connect the newly created Transmitter with current operator's input port
            add_connection(btx_handle->cid(), target_cid);
            HOLOSCAN_LOG_DEBUG(
                "Connected DownstreamReceptiveSchedulingTerm for Broadcast source : {} -> "
                "target : {}",
                port_name,
                target_port);
          } break;
          case IOSpec::ConnectorType::kUCX: {
            // Create UcxTransmitter resource temporary to create a GXF component.
            std::shared_ptr<UcxTransmitter> transmitter;

            // If the current Operator is a VirtualTransmitterOp, we create a UcxTransmitter
            // from the current Operator's arguments.
            if (op_type == Operator::OperatorType::kVirtual) {
              auto& arg_list = static_cast<ops::VirtualOperator*>(op.get())->arg_list();
              transmitter =
                  std::make_shared<UcxTransmitter>(Arg("capacity", prev_connector_capacity),
                                                   Arg("policy", prev_connector_policy),
                                                   arg_list);
            } else {
              auto prev_ucx_connector = std::dynamic_pointer_cast<UcxTransmitter>(prev_connector);
              if (!prev_ucx_connector) {
                throw std::runtime_error("failed to cast connector to UcxTransmitter");
              }
              // could also get these via prev_tx_handle->getParameter<T>(name) calls
              transmitter = std::make_shared<UcxTransmitter>(
                  Arg("capacity", prev_ucx_connector->capacity_),
                  Arg("policy", prev_ucx_connector->policy_),
                  Arg("receiver_address", prev_ucx_connector->receiver_address()),
                  Arg("port", prev_ucx_connector->port()),
                  Arg("local_address", prev_ucx_connector->local_address()),
                  Arg("local_port", prev_ucx_connector->local_port()));
            }
            auto broadcast_out_port_name = fmt::format("{}_{}", op->name(), port_name);
            transmitter->name(broadcast_out_port_name);
            transmitter->fragment(fragment_);
            auto spec = std::make_shared<ComponentSpec>(fragment_);
            transmitter->setup(*spec.get());
            transmitter->spec(spec);
            // Set eid to the broadcast entity's eid so that this component is bound to the
            // broadcast entity.
            transmitter->gxf_eid(broadcast_entity->eid());
            transmitter->gxf_graph_entity(broadcast_entity);
            // Create a transmitter in the broadcast entity.
            transmitter->initialize();
          } break;
          default:
            HOLOSCAN_LOG_ERROR("Unrecognized connector_type '{}' for source name '{}'",
                               static_cast<int>(prev_connector_type),
                               port_name);
        }
      }

      // Now delete the key
      port_map_val->erase(port_name);
    }
  }
}

void GXFExecutor::create_broadcast_components(holoscan::OperatorGraph::NodeType op,
                                              BroadcastEntityMapType& broadcast_entities,
                                              const TargetConnectionsMapType& connections) {
  auto& op_name = op->name();
  auto context = context_;
  auto entity_prefix = entity_prefix_;

  for (const auto& [source_cid, target_info] : connections) {
    auto& [source_cname, connector_type, target_ports] = target_info;
    if (target_ports.empty()) {
      HOLOSCAN_LOG_ERROR("No target component found for source_id: {}", source_cid);
      continue;
    } else if (target_ports.size() == 1) {
      continue;
    }
    // Insert GXF's Broadcast component if source port is connected to multiple targets
    std::string rx_type_name;

    uint64_t curr_min_size = 1;
    uint64_t curr_connector_capacity = 1;
    uint64_t curr_connector_policy = holoscan::gxf::get_default_queue_policy();

    // Create a corresponding condition of the op's output port and set it as the
    // receiver's condition for the broadcast entity.
    auto& op_io_spec = op->spec()->outputs()[source_cname];

    // 1. Find the output port's condition.
    //    (ConditionType::kDownstreamMessageAffordable)
    std::shared_ptr<holoscan::Condition> curr_condition;
    for (auto& [condition_type, condition] : op_io_spec->conditions()) {
      if (condition_type == ConditionType::kDownstreamMessageAffordable) {
        curr_condition = condition;
        break;
      }
    }
    // 2. If it exists, store min_size parameter of the condition.
    if (curr_condition) {
      auto curr_downstream_condition =
          std::dynamic_pointer_cast<DownstreamMessageAffordableCondition>(curr_condition);
      curr_min_size = curr_downstream_condition->min_size();
    }

    auto broadcast_entity = std::make_shared<nvidia::gxf::GraphEntity>();
    auto broadcast_entity_name =
        fmt::format("{}_broadcast_{}_{}", entity_prefix, op_name, source_cname);
    auto maybe = broadcast_entity->setup(context, broadcast_entity_name.c_str());
    if (!maybe) {
      throw std::runtime_error(
          fmt::format("Failed to create broadcast entity: '{}'", broadcast_entity_name));
    }
    // Add the broadcast_entity to the list of implicit broadcast entities
    implicit_broadcast_entities_.push_back(broadcast_entity);

    // Add the broadcast_entity for the current operator and the source port name
    broadcast_entities[op][source_cname] = broadcast_entity;

    switch (connector_type) {
      case IOSpec::ConnectorType::kDefault:
      case IOSpec::ConnectorType::kDoubleBuffer:
      case IOSpec::ConnectorType::kUCX:  // In any case, need to add doubleBufferReceiver.
      {
        // We don't create a holoscan::AnnotatedDoubleBufferReceiver even if data flow
        // tracking is on because we don't want to mark annotations for the Broadcast
        // component.
        rx_type_name = "nvidia::gxf::DoubleBufferReceiver";
        auto curr_tx_handle =
            op->graph_entity()->get<nvidia::gxf::DoubleBufferTransmitter>(source_cname.c_str());
        if (curr_tx_handle.is_null()) {
          HOLOSCAN_LOG_ERROR(
              "Failed to get nvidia::gxf::DoubleBufferTransmitter, a default receive capacity "
              "and policy will be used for the inserted broadcast component.");
        } else {
          HOLOSCAN_LOG_TRACE("getting capacity and policy from curr_tx_handle");
          auto p = get_capacity_and_policy(curr_tx_handle);
          curr_connector_capacity = p.first;
          curr_connector_policy = p.second;
        }
      } break;
      default:
        HOLOSCAN_LOG_ERROR("Unrecognized connector_type '{}' for source name '{}'",
                           static_cast<int>(connector_type),
                           source_cname);
    }
    auto broadcast_component_name =
        fmt::format("{}_broadcast_component_{}_{}", entity_prefix, op_name, source_cname);
    auto broadcast_codelet =
        broadcast_entity->addCodelet("nvidia::gxf::Broadcast", broadcast_component_name.c_str());
    if (broadcast_codelet.is_null()) {
      HOLOSCAN_LOG_ERROR("Failed to create broadcast codelet for entity: {}",
                         broadcast_entity->name());
    }
    // Broadcast component's receiver Parameter is named "source" so have to use that here
    auto broadcast_rx = broadcast_entity->addReceiver(rx_type_name.c_str(), "source");
    if (broadcast_rx.is_null()) {
      HOLOSCAN_LOG_ERROR("Failed to create receiver for broadcast component: {}",
                         broadcast_entity->name());
    }
    broadcast_entity->configReceiver(
        "source", curr_connector_capacity, curr_connector_policy, curr_min_size);

    // Connect Broadcast entity's receiver with the transmitter of the current operator
    add_connection(source_cid, broadcast_rx->cid());
  }
}

bool GXFExecutor::initialize_fragment() {
  HOLOSCAN_LOG_DEBUG("Initializing Fragment.");

  // Initialize the GXF graph by creating GXF entities related to the Holoscan operators in a
  // topologically sorted order. Operators are created as nodes in the graph of the fragment are
  // visited.
  // The direct connections between the operators are only created when a destination operator
  // is being visited.
  // The Broadcast component is created when a source operator is determined to be
  // connected to multiple targets. However, the transmitters in the Broadcast entity are not added
  // until the destination operator is visited and initialized.

  auto& graph = fragment_->graph();

  ConnectionMapType connection_map = generate_connection_map(graph, connection_items_);

  // Iterate connection map and create virtual receiver operators and connections
  std::vector<std::shared_ptr<ops::VirtualOperator>> virtual_ops;
  virtual_ops.reserve(connection_items_.size());

  // Populate virtual_ops and add connections to the fragment.
  create_virtual_operators_and_connections(fragment_, connection_map, virtual_ops);
  connect_ucx_transmitters_to_virtual_ops(fragment_, virtual_ops);

  auto operators = graph.get_nodes();

  // Create a list of nodes in the graph to iterate in topological order
  std::deque<holoscan::OperatorGraph::NodeType> worklist;
  // Create a list of the indegrees of all the nodes in the graph
  std::unordered_map<holoscan::OperatorGraph::NodeType, int> indegrees;

  // Create a set of visited nodes to avoid visiting the same node more than once.
  std::unordered_set<holoscan::OperatorGraph::NodeType> visited_nodes;
  visited_nodes.reserve(operators.size());

  // Keep a list of all the nvidia::gxf::GraphEntity entities holding broadcast codelets, if an
  // operator's output port is connected to multiple inputs. The map is indexed by the operators.
  // Each value in the map is indexed by the source port name.
  BroadcastEntityMapType broadcast_entities;

  // Initialize the indegrees of all nodes in the graph and add root operators to the worklist.
  for (auto& node : operators) {
    indegrees[node] = graph.get_previous_nodes(node).size();
    if (indegrees[node] == 0) {
      // Insert a root node as indegree is 0
      // node is not moved with std::move because operators may be used later
      worklist.push_back(node);
    }
  }

  while (true) {
    if (worklist.empty()) {
      // If the worklist is empty, we check if we have visited all nodes.
      if (visited_nodes.size() == operators.size()) {
        // If we have visited all nodes, we are done.
        break;
      } else {
        // If we have not visited all nodes, we have a cycle in the graph.
        HOLOSCAN_LOG_DEBUG(
            "Worklist is empty, but not all nodes have been visited. There is a cycle.");

        for (auto& node : operators) {
          if (indegrees[node]) {  // if indegrees is still positive add to the worklist
            indegrees[node] = 0;  // artificially breaking the cycle
            worklist.push_back(node);
          }
        }
      }
    }
    const auto& op = worklist.front();
    worklist.pop_front();

    auto op_spec = op->spec();
    auto& op_name = op->name();

    // Check if we have already visited this node
    if (visited_nodes.find(op) != visited_nodes.end()) { continue; }
    visited_nodes.insert(op);

    HOLOSCAN_LOG_DEBUG("Operator: {}", op_name);
    // Initialize the operator while we are visiting a node in the graph
    try {
      op->initialize();
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR(
          "Exception occurred during initialization of operator: '{}' - {}", op->name(), e.what());
      throw;
    }

    auto op_type = op->operator_type();

    HOLOSCAN_LOG_DEBUG("Connecting earlier operators of Op: {}", op_name);
    // Add the connections from the previous operator to the current operator, for both direct and
    // Broadcast connections.
    auto prev_operators = graph.get_previous_nodes(op);

    for (auto& prev_op : prev_operators) {
      auto port_map = graph.get_port_map(prev_op, op);
      if (!port_map.has_value()) {
        HOLOSCAN_LOG_ERROR("Could not find port map for {} -> {}", prev_op->name(), op->name());
        return false;
      }

      const auto& port_map_val = port_map.value();

      // If the previous operator is found to be one that is connected to the current operator via
      // a Broadcast component, then add the connection between the Broadcast component and the
      // current operator's input port. Only ports with inter-fragment connections use broadcasting.
      if (broadcast_entities.find(prev_op) != broadcast_entities.end()) {
        // Add transmitter to the prev_op's broadcast component and connect it to op's input port.
        // Any connected ports are removed from port_map_val.
        connect_broadcast_to_previous_op(broadcast_entities, op, prev_op, port_map_val);
      }

      if (port_map_val->size()) {
        // If there are any more mappings in the input_port_map after Broadcast entity was added, or
        // if there was no Broadcast entity added, then there must be some direct connections

        auto prev_op_type = prev_op->operator_type();

        // If previous or current operator's type is virtual operator, we don't need to connect it.
        if (prev_op_type == Operator::OperatorType::kVirtual ||
            op_type == Operator::OperatorType::kVirtual) {
          continue;
        }

        for (const auto& [source_port, target_ports] : *port_map_val) {
          gxf_uid_t source_cid = -1;
          // Only if previous operator is initialized, then source edge cid is valid
          // For cycles, a previous operator may not have been initialized yet
          if (prev_op->id() != -1) {  // id of an operator is -1 if it is not initialized
            auto source_gxf_resource = std::dynamic_pointer_cast<GXFResource>(
                prev_op->spec()->outputs()[source_port]->connector());
            source_cid = source_gxf_resource->gxf_cid();
          }

          // GXF Connection component should not be added for types using a NetworkContext
          auto connector_type = prev_op->spec()->outputs()[source_port]->connector_type();
          if (connector_type != IOSpec::ConnectorType::kUCX) {
            // const auto& target_port = target_ports.begin();
            for (const auto& target_port : target_ports) {
              auto target_gxf_resource = std::dynamic_pointer_cast<GXFResource>(
                  op->spec()->inputs()[target_port]->connector());
              // For cycles, a previous operator may not have been initialized yet, so we don't
              // connect them here. We connect them as a forward/downstream connection when we
              // visit the operator which is connected to the current operator.
              if (prev_op->id() != -1) {
                gxf_uid_t target_cid = target_gxf_resource->gxf_cid();
                add_connection(source_cid, target_cid);
                HOLOSCAN_LOG_DEBUG(
                    "Connected directly source : {} -> target : {}", source_port, target_port);
              } else {
                HOLOSCAN_LOG_DEBUG("Connection source: {} -> target: {} will be added later",
                                   source_port,
                                   target_port);
              }
            }
          } else if (target_ports.size() > 1) {
            HOLOSCAN_LOG_ERROR(
                "Source port with UCX connector is connected to multiple target ports without a "
                "Broadcast component. Op: {} source name: {}",
                op->name(),
                source_port);
            return false;
          }
        }
      }
    }

    HOLOSCAN_LOG_DEBUG("Checking next operators of Op: {}", op_name);

    // Collect the downstream connections to create necessary Broadcast components

    // Map of connections indexed by source port uid and stores a pair of the target operator name
    // and target port name
    TargetConnectionsMapType connections;

    for (auto& next_op : graph.get_next_nodes(op)) {
      auto& next_op_name = next_op->name();
      HOLOSCAN_LOG_DEBUG("  Next operator: {}", next_op_name);
      auto port_map_opt = graph.get_port_map(op, next_op);
      if (!port_map_opt.has_value()) {
        HOLOSCAN_LOG_ERROR("Could not find port map for {} -> {}", op_name, next_op_name);
        continue;
      }
      const auto& port_map = port_map_opt.value();

      for (const auto& [source_port, target_ports] : *port_map) {
        for (const auto& target_port : target_ports) {
          HOLOSCAN_LOG_DEBUG("    Port: {} -> {}", source_port, target_port);

          // If current operator's type is virtual operator, we don't need to connect it.
          if (op_type != Operator::OperatorType::kVirtual) {
            auto source_gxf_resource = std::dynamic_pointer_cast<GXFResource>(
                op_spec->outputs()[source_port]->connector());
            gxf_uid_t source_cid = source_gxf_resource->gxf_cid();
            std::string source_cname = source_gxf_resource->name();

            auto connector_type = op_spec->outputs()[source_port]->connector_type();
            if (connections.find(source_cid) == connections.end()) {
              connections[source_cid] =
                  TargetsInfo{source_cname, connector_type, std::set<TargetPort>{}};
            }
            // For the source port, add a target in the tuple form (next operator, receiving port
            // name)
            std::get<std::set<TargetPort>>(connections[source_cid])
                .insert(std::make_pair(next_op, target_port));
          }
        }
      }

      // Decrement the indegree of the next operator as the current operator's connection is
      // processed
      indegrees[next_op] -= 1;
      // Add next operator to worklist if all the previous operators have been processed
      if (!indegrees[next_op]) {
        worklist.push_back(
            std::move(next_op));  // next_op is moved because get_next_nodes returns a new vector
      }
    }
    // Iterate through downstream connections and find the direct ones to connect, only if
    // downstream operator is already initialized. This is to handle cycles in the graph.
    bool target_op_has_ucx_connector = false;
    for (auto [source_cid, target_info] : connections) {
      auto& [source_cname, connector_type, target_ports] = target_info;
      if (connector_type == IOSpec::ConnectorType::kUCX) {
        target_op_has_ucx_connector = true;
        continue;  // Connection components are only for non-UCX connections
      }
      for (auto& [tmp_next_op, target_port_name] : target_ports) {
        if (tmp_next_op->id() != -1) {
          // Operator is already initialized
          HOLOSCAN_LOG_DEBUG("next op {} is already initialized, due to a cycle.",
                             tmp_next_op->name());
          auto target_gxf_resource = std::dynamic_pointer_cast<GXFResource>(
              tmp_next_op->spec()->inputs()[target_port_name]->connector());
          gxf_uid_t target_cid = target_gxf_resource->gxf_cid();
          add_connection(source_cid, target_cid);
          HOLOSCAN_LOG_TRACE(
              "Next Op {} is connected to the current Op {} as a downstream connection due to a "
              "cycle.",
              tmp_next_op->name(),
              op_name);
        }
      }
    }

    if (!target_op_has_ucx_connector) {
      for (auto& next_op : graph.get_next_nodes(op)) {
        if (next_op->operator_type() == Operator::OperatorType::kVirtual) {
          target_op_has_ucx_connector = true;
          break;
        }
      }
    }

    if (target_op_has_ucx_connector) {
      HOLOSCAN_LOG_DEBUG("At least one target of op {} has a UCX connector.", op_name);
      // Create the Broadcast components and add their IDs to broadcast_entities, but do not add
      // any transmitter to the Broadcast entity. The transmitters will be added later when the
      // incoming edges to the respective operators are processed.
      create_broadcast_components(op, broadcast_entities, connections);
      if (op_type != Operator::OperatorType::kVirtual) {
        for (auto& next_op : graph.get_next_nodes(op)) {
          if (next_op->id() != -1 && next_op->operator_type() != Operator::OperatorType::kVirtual) {
            HOLOSCAN_LOG_DEBUG(
                "next_op of {} is {}. It is already initialized.", op_name, next_op->name());
            // next operator is already initialized, so connect the broadcast component to the
            // next operator's input port, if there is any
            auto port_map = graph.get_port_map(op, next_op);
            if (!port_map.has_value()) {
              HOLOSCAN_LOG_ERROR("Could not find port map for {} -> {}", op_name, next_op->name());
              return false;
            }
            if (broadcast_entities.find(op) != broadcast_entities.end()) {
              connect_broadcast_to_previous_op(broadcast_entities, next_op, op, port_map.value());
            }
          }
        }
      }
    } else {
      HOLOSCAN_LOG_DEBUG("No target of op {} has a UCX connector.", op_name);
    }
  }
  return true;
}

bool GXFExecutor::initialize_operator(Operator* op) {
  if (own_gxf_context_ && !is_gxf_graph_initialized_) {
    HOLOSCAN_LOG_ERROR(
        "Fragment graph is not composed yet. Operator should not be initialized in GXFExecutor. "
        "Op: {}.",
        op->name());
    return false;
  } else if (!own_gxf_context_) {  // GXF context was created outside
    HOLOSCAN_LOG_DEBUG("Not an own GXF context. Op: {}", op->name());
  }

  // Skip if the operator is already initialized
  if (op->is_initialized_) {
    HOLOSCAN_LOG_DEBUG("Operator '{}' is already initialized. Skipping initialization.",
                       op->name());
    return true;
  }

  HOLOSCAN_LOG_DEBUG("Initializing Operator '{}'", op->name());

  if (!op->spec()) {
    HOLOSCAN_LOG_ERROR("No operator spec for GXFOperator '{}'", op->name());
    return false;
  }

  auto& spec = *(op->spec());

  // op_eid_ should only be nonzero if OperatorWrapper wraps a codelet created by GXF.
  // In that case GXF has already created the entity and we can't create a GraphEntity.
  gxf_uid_t eid = (op_eid_ == 0) ? op->initialize_graph_entity(context_, entity_prefix_) : op_eid_;

  // Create Codelet component if `op_cid_` is 0
  gxf_uid_t codelet_cid = (op_cid_ == 0) ? op->add_codelet_to_graph_entity() : op_cid_;

  // Set GXF Codelet ID as the ID of the operator
  op->id(codelet_cid);

  // Create Components for input
  const auto& inputs = spec.inputs();
  for (const auto& [name, io_spec] : inputs) {
    gxf::GXFExecutor::create_input_port(fragment(), context_, eid, io_spec.get(), op_eid_ != 0, op);
  }

  // Create Components for output
  const auto& outputs = spec.outputs();
  for (const auto& [name, io_spec] : outputs) {
    gxf::GXFExecutor::create_output_port(
        fragment(), context_, eid, io_spec.get(), op_eid_ != 0, op);
  }

  HOLOSCAN_LOG_TRACE("Configuring operator: {}", op->name());

  // add Component(s) and/or Resource(s) added as Arg/ArgList to the graph entity
  add_component_args_to_graph_entity(op->args(), op->graph_entity());

  // Initialize components and resources (and add any GXF components to the Operator's graph_entity)
  op->initialize_conditions();
  op->initialize_resources();

  // Set any parameters based on the specified arguments and parameter value defaults.
  op->set_parameters();

  // Set the operator is initialized
  op->is_initialized_ = true;
  return true;
}

bool GXFExecutor::add_receivers(const std::shared_ptr<Operator>& op,
                                const std::string& receivers_name,
                                std::vector<std::string>& new_input_labels,
                                std::vector<holoscan::IOSpec*>& iospec_vector) {
  const auto downstream_op_spec = op->spec();

  // Create input port for the receivers parameter in the spec

  // Create a new input port label
  const std::string& new_input_label = fmt::format("{}:{}", receivers_name, iospec_vector.size());
  HOLOSCAN_LOG_TRACE("add_receivers: Creating new input port with label '{}'", new_input_label);
  auto& input_port = downstream_op_spec->input<holoscan::gxf::Entity>(new_input_label);

  // Add the new input port to the vector.
  iospec_vector.push_back(&input_port);

  // IOSpec vector is added, the parameters will be initialized in the initialize_operator()
  // function, when all the parameters are initialized.

  // Add new label to the label vector so that the port map of the graph edge can be updated.
  new_input_labels.push_back(new_input_label);

  return true;
}

bool GXFExecutor::is_holoscan() const {
  bool zero_eid = (op_eid_ == 0);
  bool zero_cid = (op_cid_ == 0);
  if (zero_eid ^ zero_cid) {
    // Both will be zero for Holoscan applications, but nonzero for GXF applications
    HOLOSCAN_LOG_ERROR(
        "Both op_eid_ and op_cid_ should be zero or nonzero. op_eid_: {}, op_cid_: {}",
        op_eid_,
        op_cid_);
    return false;
  }
  return zero_eid && zero_cid;
}

bool GXFExecutor::initialize_gxf_graph(OperatorGraph& graph) {
  if (is_gxf_graph_initialized_) {
    HOLOSCAN_LOG_WARN("GXF graph is already initialized. Skipping initialization.");
    return true;
  }
  is_gxf_graph_initialized_ = true;

  if (graph.is_empty()) {
    HOLOSCAN_LOG_WARN("Operator graph is empty. Skipping execution.");
    return true;
  }

  auto context = context_;

  // Since GXF is not thread-safe, we need to lock the GXF context for execution while
  // multiple threads are setting up the graph.
  static std::mutex gxf_execution_mutex;

  {
    // Lock the GXF context for execution
    std::scoped_lock lock{gxf_execution_mutex};

    // Additional setup for GXF Application
    const std::string utility_entity_name = fmt::format("{}_holoscan_util_entity", entity_prefix_);
    util_entity_ = std::make_shared<nvidia::gxf::GraphEntity>();
    auto maybe = util_entity_->setup(context, utility_entity_name.c_str());
    if (!maybe) {
      throw std::runtime_error(
          fmt::format("Failed to create utility entity: '{}'", utility_entity_name));
    }
    gxf_uid_t eid = util_entity_->eid();

    connections_entity_ = std::make_shared<nvidia::gxf::GraphEntity>();
    const std::string connections_entity_name =
        fmt::format("{}_holoscan_connections_entity", entity_prefix_);
    connections_entity_ = std::make_shared<nvidia::gxf::GraphEntity>();
    maybe = connections_entity_->setup(context_, connections_entity_name.c_str());
    if (!maybe) {
      throw std::runtime_error(
          "Failed to create entity to hold nvidia::gxf::Connection components.");
    }

    bool job_stats_enabled =
        AppDriver::get_bool_env_var("HOLOSCAN_ENABLE_GXF_JOB_STATISTICS", false);
    if (job_stats_enabled) {
      auto clock = util_entity_->add<nvidia::gxf::RealtimeClock>("jobstats_clock", {});
      if (clock) {
        bool codelet_statistics =
            AppDriver::get_bool_env_var("HOLOSCAN_GXF_JOB_STATISTICS_CODELET", false);
        uint32_t event_history_count =
            AppDriver::get_int_env_var("HOLOSCAN_GXF_JOB_STATISTICS_COUNT", 100u);

        // GXF issue 4552622: can't create FilePath Arg, so we call setParameter below instead
        std::vector<nvidia::gxf::Arg> jobstats_args{
            nvidia::gxf::Arg("clock", clock),
            nvidia::gxf::Arg("codelet_statistics", codelet_statistics),
            nvidia::gxf::Arg("event_history_count", event_history_count)};

        std::string json_file_path = "";  // default value is no JSON output
        const char* path_env_value = std::getenv("HOLOSCAN_GXF_JOB_STATISTICS_PATH");
        if (path_env_value && path_env_value[0] != '\0') {
          jobstats_args.push_back(
              nvidia::gxf::Arg("json_file_path", nvidia::gxf::FilePath(path_env_value)));
        }

        HOLOSCAN_LOG_DEBUG("GXF JobStatistics enabled with:");
        HOLOSCAN_LOG_DEBUG("  codelet report: {}", codelet_statistics);
        HOLOSCAN_LOG_DEBUG("  event_history_count: {}", event_history_count);
        HOLOSCAN_LOG_DEBUG("  json_file_path: {}", json_file_path);
        auto stats =
            util_entity_->addComponent("nvidia::gxf::JobStatistics", "jobstats", jobstats_args);
        if (!stats) { HOLOSCAN_LOG_ERROR("Failed to create JobStatistics component."); }
      } else {
        HOLOSCAN_LOG_ERROR(
            "Failed to create clock for job statistics (statistics will not be collected).");
      }
    }

    auto scheduler = std::dynamic_pointer_cast<gxf::GXFScheduler>(fragment_->scheduler());
    scheduler->initialize();  // will call GXFExecutor::initialize_scheduler

    // Initialize the fragment and its operators
    if (!initialize_fragment()) {
      HOLOSCAN_LOG_ERROR("Failed to initialize fragment");
      return false;
    }

    // If DFFT is on, then attach the DFFTCollector EntityMonitor to the main entity
    if (fragment_->data_flow_tracker()) {
      auto dft_tracker_handle = util_entity_->add<holoscan::DFFTCollector>("dft_tracker", {});
      if (dft_tracker_handle.is_null()) {
        throw std::runtime_error(fmt::format("Unable to add holoscan::DFFTCollector component."));
      }

      holoscan::DFFTCollector* dfft_collector_ptr = dft_tracker_handle.get();
      dfft_collector_ptr->data_flow_tracker(fragment_->data_flow_tracker());

      // Identify leaf and root operators and add to the DFFTCollector object
      for (auto& op : graph.get_nodes()) {
        if (op->is_leaf()) {
          dfft_collector_ptr->add_leaf_op(op.get());
        } else if (op->is_root() || op->is_user_defined_root()) {
          dfft_collector_ptr->add_root_op(op.get());
        }
      }
    }

    // network context initialization after connection entities were created (see GXF's program.cpp)
    if (fragment_->network_context()) {
      HOLOSCAN_LOG_DEBUG("GXFExecutor::run: initializing NetworkContext");
      auto network_context =
          std::dynamic_pointer_cast<gxf::GXFNetworkContext>(fragment_->network_context());
      // have to set the application eid before initialize() can be called
      network_context->gxf_eid(eid);
      network_context->initialize();

      auto entity_group_gid = ::holoscan::gxf::add_entity_group(context_, "network_entity_group");

      int32_t gpu_id =
          static_cast<int32_t>(AppDriver::get_int_env_var("HOLOSCAN_UCX_DEVICE_ID", 0));
      std::string device_entity_name = fmt::format("{}gpu_device_entity", entity_prefix_);
      gpu_device_entity_ = std::make_shared<nvidia::gxf::GraphEntity>();
      auto maybe = gpu_device_entity_->setup(context, device_entity_name.c_str());
      if (!maybe) {
        throw std::runtime_error(
            fmt::format("Failed to create GPU device entity: '{}'", device_entity_name));
      }
      // TODO (GXF4): should have an addResource to add to resources_ member instead of components_?
      auto device_handle = gpu_device_entity_->addComponent(
          "nvidia::gxf::GPUDevice", "gpu_device_component", {nvidia::gxf::Arg("dev_id", gpu_id)});
      if (device_handle.is_null()) {
        HOLOSCAN_LOG_ERROR("Failed to create GPU device resource for device {}", gpu_id);
      }

      // Note: GxfUpdateEntityGroup
      //   calls Runtime::GxfUpdateEntityGroup(gid, eid)
      //     which calls  EntityGroups::groupAddEntity(gid, eid); (entity_groups_ in
      //     SharedContext)
      //       which calls EntityGroupItem::addEntity for the EntityGroupItem corresponding to
      //       gid
      //         any eid corresponding to a ResourceBase class like GPUDevice or ThreadPool is
      //             stored in internal resources_ vector
      //         all other eid are stored in the entities vector

      // add GPUDevice resource to the networking entity group
      GXF_ASSERT_SUCCESS(
          GxfUpdateEntityGroup(context_, entity_group_gid, gpu_device_entity_->eid()));

      // add the network context to the entity group
      auto gxf_network_context =
          std::dynamic_pointer_cast<holoscan::gxf::GXFNetworkContext>(fragment_->network_context());
      HOLOSCAN_GXF_CALL_FATAL(
          GxfUpdateEntityGroup(context, entity_group_gid, gxf_network_context->gxf_eid()));

      // Loop through all operators and add any operators with a UCX port to the entity group
      auto& operator_graph = static_cast<OperatorFlowGraph&>(fragment_->graph());
      for (auto& node : operator_graph.get_nodes()) {
        auto op_spec = node->spec();
        bool already_added = false;
        for (const auto& [_, io_spec] : op_spec->inputs()) {
          if (io_spec->connector_type() == IOSpec::ConnectorType::kUCX) {
            add_operator_to_entity_group(context, entity_group_gid, node);
            already_added = true;
            break;
          }
        }
        if (already_added) { continue; }
        for (const auto& [_, io_spec] : op_spec->outputs()) {
          if (io_spec->connector_type() == IOSpec::ConnectorType::kUCX) {
            add_operator_to_entity_group(context, entity_group_gid, node);
            break;
          }
        }
      }

      // Add implicit broadcast entities to the network entity group if they have a UCX connector
      for (auto& broadcast_entity : implicit_broadcast_entities_) {
        // Add the entity to the entity group if it has a UCX connector
        if (has_ucx_connector(broadcast_entity)) {
          auto broadcast_eid = broadcast_entity->eid();
          HOLOSCAN_LOG_DEBUG("Adding implicit broadcast eid '{}' to entity group '{}'",
                             broadcast_eid,
                             entity_group_gid);
          HOLOSCAN_GXF_CALL_FATAL(GxfUpdateEntityGroup(context, entity_group_gid, broadcast_eid));
        }
      }
    } else {
      HOLOSCAN_LOG_DEBUG("GXFExecutor::run: no NetworkContext to initialize");

      const std::string ucx_error_msg{
          "UCX-based connection found, but there is no NetworkContext."};

      // Raise an error if any operator has a UCX connector.
      auto& operator_graph = static_cast<OperatorFlowGraph&>(fragment_->graph());
      for (auto& node : operator_graph.get_nodes()) {
        if (node->has_ucx_connector()) { throw std::runtime_error(ucx_error_msg); }
      }

      // Raise an error if any broadcast entity has a UCX connector
      for (auto& broadcast_entity : implicit_broadcast_entities_) {
        // Add the entity to the entity group if it has a UCX connector
        if (has_ucx_connector(broadcast_entity)) { throw std::runtime_error(ucx_error_msg); }
      }
    }
  }

  return true;
}

void GXFExecutor::activate_gxf_graph() {
  // Activate the graph if it is not already activated.
  // This allows us to activate multiple gxf graphs sequentially in a single process, avoiding
  // segfaults that occur when trying to activate multiple graphs in parallel using multi threading.
  if (!is_gxf_graph_activated_) {
    auto context = context_;
    HOLOSCAN_LOG_INFO("Activating Graph...");
    HOLOSCAN_GXF_CALL_FATAL(GxfGraphActivate(context));
    is_gxf_graph_activated_ = true;
  }
}

void GXFExecutor::run_gxf_graph() {
  auto context = context_;

  // Install signal handler
  auto sig_handler = [](void* context, int signum) {
    (void)signum;
    gxf_result_t code = GxfGraphInterrupt(context);
    if (code != GXF_SUCCESS) {
      HOLOSCAN_LOG_ERROR("GxfGraphInterrupt Error: {}", GxfResultStr(code));
      HOLOSCAN_LOG_ERROR("Send interrupt once more to terminate immediately");
      SignalHandler::unregister_signal_handler(context, signum);
      // Register the global signal handler.
      SignalHandler::register_global_signal_handler(signum, [](int sig) {
        (void)sig;
        HOLOSCAN_LOG_ERROR("Interrupted by user (global signal handler)");
        exit(1);
      });
    }
  };
  SignalHandler::register_signal_handler(context, SIGINT, sig_handler);
  SignalHandler::register_signal_handler(context, SIGTERM, sig_handler);

  // Run the graph
  auto frag_name_display = fragment_->name();
  if (!frag_name_display.empty()) { frag_name_display = "[" + frag_name_display + "] "; }
  activate_gxf_graph();
  HOLOSCAN_LOG_INFO("{}Running Graph...", frag_name_display);
  HOLOSCAN_GXF_CALL_FATAL(GxfGraphRunAsync(context));
  HOLOSCAN_LOG_INFO("{}Waiting for completion...", frag_name_display);
  auto wait_result = HOLOSCAN_GXF_CALL_WARN(GxfGraphWait(context));
  if (wait_result == GXF_SUCCESS) {
    HOLOSCAN_LOG_INFO("{}Deactivating Graph...", frag_name_display);
    // Usually the graph is already deactivated by the GXF framework (program.cpp)
    // when GxfGraphWait() fails.
    HOLOSCAN_GXF_CALL_WARN(GxfGraphDeactivate(context));
  }
  is_gxf_graph_activated_ = false;

  // TODO: do we want to move the log level of these info messages to debug?
  HOLOSCAN_LOG_INFO("{}Graph execution finished.", frag_name_display);

  // clean up any shared pointers to graph entities within operators, scheulder, network context
  fragment_->reset_graph_entities();

  if (wait_result != GXF_SUCCESS) {
    const std::string error_msg =
        fmt::format("{}Graph execution error: {}", frag_name_display, GxfResultStr(wait_result));
    HOLOSCAN_LOG_ERROR(error_msg);
    auto& stored_exception = exception_;
    if (stored_exception) {
      // Rethrow the stored exception if there is one
      std::rethrow_exception(stored_exception);
    }
  }
}

bool GXFExecutor::connection_items(
    std::vector<std::shared_ptr<holoscan::ConnectionItem>>& connection_items) {
  // Clear the existing connection items and add the new ones
  bool is_updated = false;
  if (!connection_items_.empty()) {
    is_updated = true;
    connection_items_.clear();
  }
  connection_items_.insert(
      connection_items_.end(), connection_items.begin(), connection_items.end());
  return is_updated;
}

void GXFExecutor::register_extensions() {
  if (gxf_holoscan_extension_ != nullptr) {
    HOLOSCAN_LOG_WARN("GXF Holoscan extension is already registered");
    return;
  }

  // Register the default GXF extensions
  for (auto& gxf_extension_file_name : kDefaultGXFExtensions) {
    extension_manager_->load_extension(gxf_extension_file_name);
  }

  // Register the default Holoscan GXF extensions
  for (auto& gxf_extension_file_name : kDefaultHoloscanGXFExtensions) {
    extension_manager_->load_extension(gxf_extension_file_name);
  }

  // Register the GXF extension that provides the native operators
  gxf_tid_t gxf_wrapper_tid{0xd4e7c16bcae741f8, 0xa5eb93731de9ccf6};
  auto gxf_extension_manager = std::dynamic_pointer_cast<GXFExtensionManager>(extension_manager_);

  if (gxf_extension_manager && !gxf_extension_manager->is_extension_loaded(gxf_wrapper_tid)) {
    GXFExtensionRegistrar extension_factory(
        context_,
        "HoloscanSdkInternalExtension",
        "A runtime hidden extension used by Holoscan SDK to provide the native operators",
        gxf_wrapper_tid);

    extension_factory.add_component<holoscan::gxf::GXFWrapper, nvidia::gxf::Codelet>(
        "GXF wrapper to support Holoscan SDK native operators");
    extension_factory.add_type<holoscan::Message>("Holoscan message type",
                                                  {0x61510ca06aa9493b, 0x8a777d0bf87476b7});
    extension_factory.add_type<holoscan::Tensor>("Holoscan's Tensor type",
                                                 {0xa5eb0ed57d7f4aa2, 0xb5865ccca0ef955c});
    extension_factory.add_type<holoscan::MetadataDictionary>(
        "Holoscan's MetadataDictionary type", {0x112607eb7b23407c, 0xb93fcd10ad8b2ba7});

    // Add a new type of Double Buffer Receiver and Transmitter
    extension_factory
        .add_component<holoscan::AnnotatedDoubleBufferReceiver, nvidia::gxf::DoubleBufferReceiver>(
            "Holoscan's annotated double buffer receiver",
            {0x218e0c7d4dda480a, 0x90a7ea8f8fb319af});

    extension_factory.add_component<holoscan::AnnotatedDoubleBufferTransmitter,
                                    nvidia::gxf::DoubleBufferTransmitter>(
        "Holoscan's annotated double buffer transmitter", {0x444505a86c014d90, 0xab7503bcd0782877});

    extension_factory.add_type<holoscan::MessageLabel>("Holoscan message Label",
                                                       {0x6e09e888ccfa4a32, 0xbc501cd20c8b4337});

    extension_factory.add_component<holoscan::DFFTCollector, nvidia::gxf::Monitor>(
        "Holoscan's DFFTCollector based on Monitor", {0xe6f50ca5cad74469, 0xad868076daf2c923});

    nvidia::gxf::Extension* extension_ptr = nullptr;
    if (!extension_factory.register_extension(&extension_ptr)) {
      HOLOSCAN_LOG_ERROR("Failed to register Holoscan SDK internal extension");
    }
    // Set the extension pointer so that we can delete the extension object later in ~GXFExecutor()
    gxf_holoscan_extension_ = extension_ptr;
  }
}

bool GXFExecutor::initialize_scheduler(Scheduler* sch) {
  if (!sch->spec()) {
    HOLOSCAN_LOG_ERROR("No component spec for GXFScheduler '{}'", sch->name());
    return false;
  }

  gxf::GXFScheduler* gxf_sch = static_cast<gxf::GXFScheduler*>(sch);
  gxf_sch->gxf_context(context_);

  // op_eid_ and op_cid_ will only be nonzero if OperatorWrapper wraps a codelet created by GXF.
  // (i.e. this executor belongs to a GXF application using a Holoscan operator as a codelet)
  // In that case GXF we do not create a GraphEntity or a Component for the scheduler.
  gxf_uid_t eid = op_eid_;
  gxf_uid_t scheduler_cid = op_cid_;
  if (is_holoscan()) {
    const std::string scheduler_entity_name = fmt::format("{}{}", entity_prefix_, sch->name());
    scheduler_entity_ = std::make_shared<nvidia::gxf::GraphEntity>();
    auto maybe = scheduler_entity_->setup(context_, scheduler_entity_name.c_str());
    if (!maybe) {
      throw std::runtime_error(
          fmt::format("Failed to create entity for scheduler: '{}'", scheduler_entity_name));
    }
    eid = scheduler_entity_->eid();
    // Set the entity id and graph entity shared pointer
    gxf_sch->gxf_graph_entity(scheduler_entity_);
    gxf_sch->gxf_eid(eid);

    // Create Scheduler component
    gxf_sch->gxf_initialize();
    scheduler_cid = gxf_sch->gxf_cid();

    // initialize all GXF resources and assign them to a graph entity
    initialize_gxf_resources(sch->resources(), eid, scheduler_entity_);

    // Set any parameters based on the specified arguments and parameter value defaults.
    add_component_args_to_graph_entity(sch->args(), scheduler_entity_);
    sch->set_parameters();
  }
  // Set GXF Scheduler ID as the ID of the scheduler
  sch->id(scheduler_cid);
  return true;
}

bool GXFExecutor::initialize_network_context(NetworkContext* network_context) {
  if (!network_context->spec()) {
    HOLOSCAN_LOG_ERROR("No component spec for GXFNetworkContext '{}'", network_context->name());
    return false;
  }

  gxf::GXFNetworkContext* gxf_network_context =
      static_cast<gxf::GXFNetworkContext*>(network_context);
  gxf_network_context->gxf_context(context_);

  // op_eid_ and op_cid_ will only be nonzero if OperatorWrapper wraps a codelet created by GXF.
  // (i.e. this executor belongs to a GXF application using a Holoscan operator as a codelet)
  // In that case GXF we do not create a GraphEntity or a Component for the network context.
  gxf_uid_t eid = op_eid_;
  gxf_uid_t network_context_cid = op_cid_;
  if (is_holoscan()) {
    const std::string network_context_entity_name =
        fmt::format("{}{}", entity_prefix_, network_context->name());
    // TODO (GXF4): add way to check error code and throw runtime_error if setup call failed
    network_context_entity_ = std::make_shared<nvidia::gxf::GraphEntity>();
    auto maybe = network_context_entity_->setup(context_, network_context_entity_name.c_str());
    if (!maybe) {
      throw std::runtime_error(fmt::format("Failed to create entity for network context: '{}'",
                                           network_context_entity_name));
    }
    eid = network_context_entity_->eid();
    // Set the entity id and graph entity shared pointer
    gxf_network_context->gxf_graph_entity(network_context_entity_);
    gxf_network_context->gxf_eid(eid);

    // Create NetworkContext component
    gxf_network_context->gxf_initialize();
    network_context_cid = gxf_network_context->gxf_cid();

    // initialize all GXF resources and assign them to a graph entity
    initialize_gxf_resources(network_context->resources(), eid, network_context_entity_);

    // Set any parameters based on the specified arguments and parameter value defaults.
    add_component_args_to_graph_entity(network_context->args(), network_context_entity_);
    network_context->set_parameters();
  }
  // Set GXF NetworkContext ID as the ID of the network_context
  network_context->id(network_context_cid);
  return true;
}

bool GXFExecutor::add_condition_to_graph_entity(
    std::shared_ptr<Condition> condition, std::shared_ptr<nvidia::gxf::GraphEntity> graph_entity) {
  if (condition && graph_entity) {
    add_component_args_to_graph_entity(condition->args(), graph_entity);
    auto gxf_condition = std::dynamic_pointer_cast<gxf::GXFCondition>(condition);
    if (!gxf_condition) {
      // Non-GXF condition isn't supported, so log an error if this unexpected path is reached.
      HOLOSCAN_LOG_ERROR("Failed to cast condition '{}' to holoscan::gxf::GXFCondition",
                         condition->name());
      return false;
    }
    // do not overwrite previous graph entity if this condition is already associated with one
    if (gxf_condition && !gxf_condition->gxf_graph_entity()) {
      HOLOSCAN_LOG_TRACE(
          "Adding Condition '{}' to graph entity '{}'", condition->name(), graph_entity->name());
      gxf_condition->gxf_eid(graph_entity->eid());
      gxf_condition->gxf_graph_entity(graph_entity);
      // Don't have to call initialize() here, ArgumentSetter already calls it later.
      return true;
    }
  }
  return false;
}

bool GXFExecutor::add_resource_to_graph_entity(
    std::shared_ptr<Resource> resource, std::shared_ptr<nvidia::gxf::GraphEntity> graph_entity) {
  if (resource && graph_entity) {
    add_component_args_to_graph_entity(resource->args(), graph_entity);
    // Native Resources will not be added to the GraphEntity
    auto gxf_resource = std::dynamic_pointer_cast<gxf::GXFResource>(resource);
    // don't raise error if the pointer cast failed as that is expected for native Resource types

    // do not overwrite previous graph entity if this resource is already associated with one
    // (e.g. sometimes the same allocator may be used across multiple operators)
    if (gxf_resource && !gxf_resource->gxf_graph_entity()) {
      HOLOSCAN_LOG_TRACE(
          "Adding Resource '{}' to graph entity '{}'", resource->name(), graph_entity->name());
      gxf_resource->gxf_eid(graph_entity->eid());
      gxf_resource->gxf_graph_entity(graph_entity);
      // Don't have to call initialize() here, ArgumentSetter already calls it later.
      return true;
    }
  }
  return false;
}

bool GXFExecutor::add_iospec_to_graph_entity(
    IOSpec* io_spec, std::shared_ptr<nvidia::gxf::GraphEntity> graph_entity) {
  if (!io_spec || !graph_entity) { return false; }
  auto resource = io_spec->connector();
  bool overall_status = false;
  if (!resource) {
    HOLOSCAN_LOG_ERROR("IOSpec: failed to cast io_spec->connector() to GXFResource");
    return overall_status;
  }
  overall_status = add_resource_to_graph_entity(resource, graph_entity);
  if (!overall_status) {
    HOLOSCAN_LOG_ERROR("IOSpec: failed to add connector '{}' to graph entity", resource->name());
  }
  for (auto& [_, condition] : io_spec->conditions()) {
    bool condition_status = add_condition_to_graph_entity(condition, graph_entity);
    if (!condition_status) {
      HOLOSCAN_LOG_ERROR("IOSpec: failed to add connector '{}' to graph entity", condition->name());
    }
    overall_status = overall_status && condition_status;
  }
  return overall_status;
}

void GXFExecutor::add_component_args_to_graph_entity(
    std::vector<Arg>& args, std::shared_ptr<nvidia::gxf::GraphEntity> graph_entity) {
  for (auto& arg : args) {
    auto arg_type = arg.arg_type();
    auto element_type = arg_type.element_type();
    if ((element_type != ArgElementType::kResource) &&
        (element_type != ArgElementType::kCondition) && (element_type != ArgElementType::kIOSpec)) {
      continue;
    }
    auto container_type = arg_type.container_type();
    if ((container_type != ArgContainerType::kNative) &&
        (container_type != ArgContainerType::kVector)) {
      HOLOSCAN_LOG_ERROR(
          "Error setting GXF entity for argument '{}': Operator currently only supports scalar and "
          "vector containers for arguments of Condition, Resource or IOSpec type.",
          arg.name());
      continue;
    }
    if (container_type == ArgContainerType::kNative) {
      if (element_type == ArgElementType::kCondition) {
        auto condition = std::any_cast<std::shared_ptr<Condition>>(arg.value());
        add_condition_to_graph_entity(std::move(condition), graph_entity);
      } else if (element_type == ArgElementType::kResource) {
        auto resource = std::any_cast<std::shared_ptr<Resource>>(arg.value());
        add_resource_to_graph_entity(std::move(resource), graph_entity);
      } else if (element_type == ArgElementType::kIOSpec) {
        auto io_spec = std::any_cast<IOSpec*>(arg.value());
        add_iospec_to_graph_entity(io_spec, graph_entity);
      }
    } else if (container_type == ArgContainerType::kVector) {
      if (element_type == ArgElementType::kCondition) {
        auto conditions = std::any_cast<std::vector<std::shared_ptr<Condition>>>(arg.value());
        for (auto& condition : conditions) {
          add_condition_to_graph_entity(condition, graph_entity);
        }
      } else if (element_type == ArgElementType::kResource) {
        auto resources = std::any_cast<std::vector<std::shared_ptr<Resource>>>(arg.value());
        for (auto& resource : resources) { add_resource_to_graph_entity(resource, graph_entity); }
      } else if (element_type == ArgElementType::kIOSpec) {
        auto io_specs = std::any_cast<std::vector<IOSpec*>>(arg.value());
        for (auto& io_spec : io_specs) { add_iospec_to_graph_entity(io_spec, graph_entity); }
      }
    }
  }
}

}  // namespace holoscan::gxf
