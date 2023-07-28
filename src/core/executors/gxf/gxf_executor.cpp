/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <common/assert.hpp>
#include <common/logger.hpp>

#include "holoscan/core/application.hpp"
#include "holoscan/core/conditions/gxf/downstream_affordable.hpp"
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
#include "holoscan/core/gxf/gxf_tensor.hpp"
#include "holoscan/core/gxf/gxf_utils.hpp"
#include "holoscan/core/gxf/gxf_wrapper.hpp"
#include "holoscan/core/message.hpp"
#include "holoscan/core/messagelabel.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/resources/gxf/annotated_double_buffer_receiver.hpp"
#include "holoscan/core/resources/gxf/annotated_double_buffer_transmitter.hpp"
#include "holoscan/core/resources/gxf/dfft_collector.hpp"
#include "holoscan/core/resources/gxf/double_buffer_receiver.hpp"
#include "holoscan/core/resources/gxf/double_buffer_transmitter.hpp"
#include "holoscan/core/services/common/virtual_operator.hpp"
#include "holoscan/core/signal_handler.hpp"

#include "gxf/std/default_extension.hpp"

#include "gxf/std/extension_factory_helper.hpp"
#include "gxf/std/monitor.hpp"
#include "gxf/test/components/entity_monitor.hpp"

namespace holoscan::gxf {

static const std::vector<std::string> kDefaultGXFExtensions{
    "libgxf_std.so",
    "libgxf_cuda.so",
    "libgxf_multimedia.so",
    "libgxf_serialization.so",
    "libgxf_ucx.so",  // UCXContext, UCXReceiver, UCXTransmitter, etc.
};

static const std::vector<std::string> kDefaultHoloscanGXFExtensions{
    "libgxf_bayer_demosaic.so",
    "libgxf_stream_playback.so",  // keep for use of VideoStreamSerializer
    "libgxf_ucx_holoscan.so",     // serialize holoscan::gxf::GXFTensor and holoscan::Message
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
    GXF_LOG_INFO("Creating context");
    HOLOSCAN_GXF_CALL_FATAL(GxfContextCreate(&context_));
    // }
    own_gxf_context_ = true;
    gxf_extension_manager_ = std::make_shared<GXFExtensionManager>(context_);
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
  // Deinitialize GXF context only if `own_gxf_context_` is true
  if (own_gxf_context_) {
    GXF_LOG_INFO("Destroying context");
    // Unregister signal handlers if any
    SignalHandler::unregister_signal_handler(context_, SIGINT);
    SignalHandler::unregister_signal_handler(context_, SIGTERM);
    HOLOSCAN_GXF_CALL(GxfContextDestroy(context_));
  }

  // Delete GXF Holoscan Extension
  if (gxf_holoscan_extension_) {
    delete gxf_holoscan_extension_;
    gxf_holoscan_extension_ = nullptr;
  }
}

static gxf_uid_t add_entity_group(void* context, std::string name) {
  gxf_uid_t entity_group_gid = kNullUid;
  HOLOSCAN_GXF_CALL_FATAL(GxfCreateEntityGroup(context, name.c_str(), &entity_group_gid));
  return entity_group_gid;
}

static std::pair<gxf_tid_t, gxf_uid_t> create_gpu_device_entity(void* context,
                                                                std::string entity_name) {
  // Get GPU device type id
  gxf_tid_t device_tid = GxfTidNull();
  HOLOSCAN_GXF_CALL_FATAL(GxfComponentTypeId(context, "nvidia::gxf::GPUDevice", &device_tid));

  // Create a GPUDevice entity
  gxf_uid_t device_eid = kNullUid;
  GxfEntityCreateInfo entity_create_info = {entity_name.c_str(), GXF_ENTITY_CREATE_PROGRAM_BIT};
  HOLOSCAN_GXF_CALL_FATAL(GxfCreateEntity(context, &entity_create_info, &device_eid));
  GXF_ASSERT_NE(device_eid, kNullUid);
  return std::make_pair(device_tid, device_eid);
}

static gxf_uid_t create_gpu_device_component(void* context, gxf_tid_t device_tid,
                                             gxf_uid_t device_eid, std::string component_name,
                                             int32_t dev_id = 0) {
  // Create the GPU device component
  gxf_uid_t device_cid = kNullUid;
  HOLOSCAN_GXF_CALL_FATAL(
      GxfComponentAdd(context, device_eid, device_tid, component_name.c_str(), &device_cid));
  GXF_ASSERT_NE(device_cid, kNullUid);

  // set the device ID parameter
  HOLOSCAN_GXF_CALL_FATAL(GxfParameterSetInt32(context, device_cid, "dev_id", dev_id));
  return device_cid;
}

void GXFExecutor::add_operator_to_entity_group(gxf_context_t context, gxf_uid_t entity_group_gid,
                                               std::shared_ptr<Operator> op) {
  gxf_uid_t op_eid = kNullUid;
  if (op->operator_type() == Operator::OperatorType::kGXF) {
    op_eid = std::dynamic_pointer_cast<holoscan::ops::GXFOperator>(op)->gxf_eid();
  } else {
    // get the GXF entity ID corresponding to the native operator's GXF Codelet
    const std::string op_entity_name = fmt::format("{}{}", entity_prefix_, op->name());
    HOLOSCAN_GXF_CALL_FATAL(GxfEntityFind(context, op_entity_name.c_str(), &op_eid));
  }
  HOLOSCAN_GXF_CALL_FATAL(GxfUpdateEntityGroup(context, entity_group_gid, op_eid));
}

void GXFExecutor::run(OperatorGraph& graph) {
  if (!initialize_gxf_graph(graph)) {
    HOLOSCAN_LOG_ERROR("Failed to initialize GXF graph");
    return;
  }

  if (!run_gxf_graph()) {
    HOLOSCAN_LOG_ERROR("Failed to run GXF graph");
    return;
  }
}

std::future<void> GXFExecutor::run_async(OperatorGraph& graph) {
  if (!is_gxf_graph_initialized_) { initialize_gxf_graph(graph); }

  return std::async(std::launch::async, [this, &graph]() {
    try {
      this->run_gxf_graph();
    } catch (const RuntimeError& e) {
      // Do not propagate the exception to the caller because the failure is already logged
      // by the GXF and the failure on GxfGraphWait() is expected when the graph is interrupted.
      // (Normal execution with the distributed application.)
      HOLOSCAN_LOG_DEBUG("Exception in GXFExecutor::run_gxf_graph - {}", e.what());
    }
    HOLOSCAN_LOG_INFO("Fragment '{}' Terminated", this->fragment()->name());
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
  gxf_extension_manager_ = std::make_shared<GXFExtensionManager>(context_);
}

std::shared_ptr<ExtensionManager> GXFExecutor::extension_manager() {
  return gxf_extension_manager_;
}

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

  // If this executor is used by OperatorWrapper (bind_port == true) to wrap Native Operator,
  // then we need to call `io_spec->connector(...)` to set the existing GXF Receiver for this
  // input.
  if (bind_port) {
    if (rx_type != IOSpec::ConnectorType::kDefault) {
      throw std::runtime_error(
          "TODO: update bind_port code path for types other than ConnectorType::kDefault");
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
      HOLOSCAN_LOG_ERROR("Unsupported GXF receiver type for the handle: '{}' in '{}' entity",
                         rx_name,
                         entity_name);
    }
    return;
  }

  auto connector = std::dynamic_pointer_cast<Receiver>(io_spec->connector());

  if (!connector || (connector->gxf_cptr() == nullptr)) {
    // Create Receiver component for this input
    std::shared_ptr<Receiver> rx_resource;
    switch (rx_type) {
      case IOSpec::ConnectorType::kDefault:
        HOLOSCAN_LOG_DEBUG("creating input port using DoubleBufferReceiver");
        rx_resource = std::make_shared<DoubleBufferReceiver>();
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

    rx_resource->gxf_eid(eid);
    rx_resource->initialize();

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
    io_spec->condition(ConditionType::kMessageAvailable,
                       Arg("receiver") = io_spec->connector(),
                       Arg("min_size") = 1UL);
  }

  // Initialize conditions for this input
  int condition_index = 0;
  for (const auto& [condition_type, condition] : io_spec->conditions()) {
    ++condition_index;
    switch (condition_type) {
      case ConditionType::kMessageAvailable: {
        std::shared_ptr<MessageAvailableCondition> message_available_condition =
            std::dynamic_pointer_cast<MessageAvailableCondition>(condition);

        message_available_condition->receiver(connector);
        message_available_condition->name(
            ::holoscan::gxf::create_name("__condition_input_", condition_index).c_str());
        message_available_condition->fragment(fragment);
        auto rx_condition_spec = std::make_shared<ComponentSpec>(fragment);
        message_available_condition->setup(*rx_condition_spec);
        message_available_condition->spec(std::move(rx_condition_spec));

        message_available_condition->gxf_eid(eid);
        message_available_condition->initialize();
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
  // If this executor is used by OperatorWrapper (bind_port == true) to wrap Native Operator,
  // then we need to call `io_spec->connector(...)` to set the existing GXF Transmitter for this
  // output.
  if (bind_port) {
    if (tx_type != IOSpec::ConnectorType::kDefault) {
      throw std::runtime_error(
          "TODO: update bind_port code path for types other than ConnectorType::kDefault");
    }
    const char* entity_name = "";
    HOLOSCAN_GXF_CALL_FATAL(GxfComponentName(gxf_context, eid, &entity_name));

    gxf_tid_t transmitter_find_tid{};
    HOLOSCAN_GXF_CALL_FATAL(
        GxfComponentTypeId(gxf_context, "nvidia::gxf::Transmitter", &transmitter_find_tid));

    gxf_uid_t transmitter_cid = 0;
    HOLOSCAN_GXF_CALL_FATAL(GxfComponentFind(
        gxf_context, eid, transmitter_find_tid, tx_name, nullptr, &transmitter_cid));

    gxf_tid_t transmitter_tid{};
    HOLOSCAN_GXF_CALL_FATAL(GxfComponentType(gxf_context, transmitter_cid, &transmitter_tid));

    gxf_tid_t double_buffer_transmitter_tid{};

    if (fragment->data_flow_tracker()) {
      HOLOSCAN_GXF_CALL_FATAL(GxfComponentTypeId(gxf_context,
                                                 "holoscan::AnnotatedDoubleBufferTransmitter",
                                                 &double_buffer_transmitter_tid));
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
    return;
  }

  auto connector = std::dynamic_pointer_cast<Transmitter>(io_spec->connector());

  if (!connector || (connector->gxf_cptr() == nullptr)) {
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

    tx_resource->gxf_eid(eid);
    tx_resource->initialize();

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

        downstream_msg_affordable_condition->transmitter(connector);
        downstream_msg_affordable_condition->name(
            ::holoscan::gxf::create_name("__condition_output_", condition_index).c_str());
        downstream_msg_affordable_condition->fragment(fragment);
        auto tx_condition_spec = std::make_shared<ComponentSpec>(fragment);
        downstream_msg_affordable_condition->setup(*tx_condition_spec);
        downstream_msg_affordable_condition->spec(std::move(tx_condition_spec));

        downstream_msg_affordable_condition->gxf_eid(eid);
        downstream_msg_affordable_condition->initialize();
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
 */
void create_virtual_operators_and_connections(
    Fragment* fragment, const ConnectionMapType& connection_map,
    std::vector<std::shared_ptr<ops::VirtualOperator>>& virtual_ops) {
  for (auto& [op, port_map] : connection_map) {
    for (auto& [port_name, connections] : port_map) {
      int connection_index = 0;
      for (auto& connection : connections) {
        auto io_type = connection->io_type;

        std::shared_ptr<ops::VirtualOperator> virtual_op;
        if (io_type == IOSpec::IOType::kOutput) {
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
          // connect op.port_name to virtual_op.port_name
          fragment->add_flow(op, virtual_op, {{port_name, port_name}});
        } else {
          // connect virtual_op.port_name to op.port_name
          fragment->add_flow(virtual_op, op, {{port_name, port_name}});

          // If we cannot find the port_name in the op's input, it means that
          // the port name is the parameter name and the parameter type is
          // 'std::vector<holoscan::IOSpec*>'.
          // In this case, we need to correct VirtualReceiverOp's port name to
          // the parameter name ('<parameter name>:<index>').
          auto op_spec = op->spec();
          auto& op_spec_inputs = op_spec->inputs();
          if (op_spec_inputs.find(port_name) == op_spec_inputs.end()) {
            auto& op_params = op_spec->params();
            const std::any& any_value = op_params[port_name].value();
            auto& param = *std::any_cast<Parameter<std::vector<holoscan::IOSpec*>>*>(any_value);
            std::vector<holoscan::IOSpec*>& iospec_vector = param.get();
            int param_index = iospec_vector.size() - 1;
            // Set the port name to <parameter name>:<index>
            virtual_op->port_name(fmt::format("{}:{}", port_name, param_index));
          }
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
        // including virtual operators. This is used to determine if direct UCXTransmitter should be
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
        // It should have only one successor.
        auto first_receiver_op = graph.get_next_nodes(virtual_op)[0];
        auto& port_name = virtual_op->port_name();

        auto& in_spec = first_receiver_op->spec()->inputs()[port_name];
        // Create the connector for in_spec from the virtual_op
        in_spec->connector(virtual_op->connector_type(), virtual_op->arg_list());
      } break;
    }
  }
}

using BroadcastEidMapType = std::unordered_map<holoscan::OperatorGraph::NodeType,
                                               std::unordered_map<std::string, gxf_uid_t>>;

/**
 * @brief Add connection between the prior Broadcast component and the current operator's input
 * port(s).
 *
 * Creates a transmitter on the broadcast component and connects it to the input port of `op`.
 *
 * Any connected ports of the operator are removed from port_map_val
 */
void connect_broadcast_to_previous_op(gxf_context_t context,  // context_
                                      Fragment* fragment_,    // fragment_
                                      const BroadcastEidMapType& broadcast_eids,
                                      holoscan::OperatorGraph::NodeType op,
                                      holoscan::OperatorGraph::NodeType prev_op,
                                      holoscan::OperatorGraph::EdgeDataType port_map_val) {
  auto op_type = op->operator_type();

  // A Broadcast component was added for prev_op
  for (const auto& [port_name, broadcast_eid] : broadcast_eids.at(prev_op)) {
    // Find the Broadcast component's source port name in the port-map.
    if (port_map_val->find(port_name) != port_map_val->end()) {
      // There is an output port of the prev_op that is associated with a Broadcast component.
      // Create a transmitter for the current Operator's input port in the Broadcast
      // entity and add a connection from the transmitter to the current Operator's input
      // port.
      auto target_ports = port_map_val->at(port_name);
      for (const auto& target_port : target_ports) {
        // Create a Transmitter in the Broadcast entity.
        gxf_uid_t tx_cid;
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
        uint64_t prev_connector_policy = 2;  // fault

        // Create a transmitter based on the prev_connector_type.
        // TODO(gbae): create a special resource for the broadcast codelet and use it.
        switch (prev_connector_type) {
          case IOSpec::ConnectorType::kDefault:
          case IOSpec::ConnectorType::kDoubleBuffer: {
            // We don't create a AnnotatedDoubleBufferTransmitter even if DFFT is on because
            // we don't want to annotate a message at the Broadcast component.
            auto prev_double_buffer_connector =
                std::dynamic_pointer_cast<DoubleBufferTransmitter>(prev_connector);
            prev_connector_capacity = prev_double_buffer_connector->capacity_;
            prev_connector_policy = prev_double_buffer_connector->policy_;

            create_gxf_component(
                context, "nvidia::gxf::DoubleBufferTransmitter", "", broadcast_eid, &tx_cid);
            HOLOSCAN_GXF_CALL_FATAL(
                GxfParameterSetUInt64(context, tx_cid, "capacity", prev_connector_capacity));
            HOLOSCAN_GXF_CALL_FATAL(
                GxfParameterSetUInt64(context, tx_cid, "policy", prev_connector_policy));

            // Clone the condition of the prev_op's output port and set it as the
            // transmitter's condition for the broadcast entity.

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
            if (prev_condition) {
              auto prev_downstream_condition =
                  std::dynamic_pointer_cast<DownstreamMessageAffordableCondition>(prev_condition);
              auto min_size = prev_downstream_condition->min_size();

              gxf_uid_t tx_term_cid;
              create_gxf_component(context,
                                   "nvidia::gxf::DownstreamReceptiveSchedulingTerm",
                                   "",
                                   broadcast_eid,
                                   &tx_term_cid);
              HOLOSCAN_GXF_CALL_FATAL(
                  GxfParameterSetHandle(context, tx_term_cid, "transmitter", tx_cid));
              HOLOSCAN_GXF_CALL_FATAL(
                  GxfParameterSetUInt64(context, tx_term_cid, "min_size", min_size));
            }
            // Get the current Operator's input port
            auto target_gxf_resource = std::dynamic_pointer_cast<GXFResource>(
                op->spec()->inputs()[target_port]->connector());
            gxf_uid_t target_cid = target_gxf_resource->gxf_cid();

            // Connect the newly created Transmitter with current operator's input port
            ::holoscan::gxf::add_connection(context, tx_cid, target_cid);
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
              prev_connector_capacity = prev_ucx_connector->capacity_;
              prev_connector_policy = prev_ucx_connector->policy_;

              auto prev_connector_receiver_address = prev_ucx_connector->receiver_address();
              auto prev_connector_port = prev_ucx_connector->port();
              transmitter = std::make_shared<UcxTransmitter>(
                  Arg("capacity", prev_connector_capacity),
                  Arg("policy", prev_connector_policy),
                  Arg("receiver_address", prev_connector_receiver_address),
                  Arg("port", prev_connector_port));
            }
            auto broadcast_out_port_name = fmt::format("{}_{}", op->name(), port_name);
            transmitter->name(broadcast_out_port_name);
            transmitter->fragment(fragment_);
            auto spec = std::make_shared<ComponentSpec>(fragment_);
            transmitter->setup(*spec.get());
            transmitter->spec(spec);
            // Set eid to the broadcast entity's eid so that this component is bound to the
            // broadcast entity.
            transmitter->gxf_eid(broadcast_eid);
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

// Map of connections indexed by source port uid and stores a pair of the target operator name
// and target port name
using TargetPort = std::pair<holoscan::OperatorGraph::NodeType, std::string>;
using TargetsInfo = std::pair<IOSpec::ConnectorType, std::set<TargetPort>>;
using TargetConnectionsMapType = std::unordered_map<gxf_uid_t, TargetsInfo>;

/**
 * @brief Create Broadcast components and add their IDs to broadcast_eids.
 *
 * Creates broadcast components for any output ports of `op` that connect to more than one
 * input port.
 *
 * Does not add any transmitter to the Broadcast entity. The transmitters will be added later
 * when the incoming edges to the respective operators are processed.
 *
 * Any connected ports of the operator are removed from port_map_val
 */
void create_broadcast_components(gxf_context_t context,             // context_
                                 const std::string& entity_prefix,  // entity_prefix_
                                 holoscan::OperatorGraph::NodeType op,
                                 BroadcastEidMapType& broadcast_eids,
                                 const TargetConnectionsMapType& connections) {
  gxf_tid_t broadcast_tid = GxfTidNull();
  gxf_tid_t rx_term_tid = GxfTidNull();

  gxf_tid_t rx_double_buffer_tid = GxfTidNull();

  auto& op_name = op->name();

  for (const auto& [source_cid, target_info] : connections) {
    auto& [connector_type, target_ports] = target_info;
    if (target_ports.empty()) {
      HOLOSCAN_LOG_ERROR("No target component found for source_id: {}", source_cid);
      continue;
    }

    // Insert GXF's Broadcast component if source port is connected to multiple targets
    if (target_ports.size() > 1) {
      gxf_tid_t rx_tid = GxfTidNull();

      const char* source_cname = "";
      HOLOSCAN_GXF_CALL_FATAL(GxfComponentName(context, source_cid, &source_cname));

      uint64_t curr_min_size = 1;
      uint64_t curr_connector_capacity = 1;
      uint64_t curr_connector_policy = 2;  // fault

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

      gxf_uid_t broadcast_eid;
      auto broadcast_entity_name =
          fmt::format("{}_broadcast_{}_{}", entity_prefix, op_name, source_cname);
      // TODO(gbae): create an operator class for the broadcast codelet and use it here
      //             instead of using GXF API directly.
      const GxfEntityCreateInfo broadcast_entity_create_info = {broadcast_entity_name.c_str(),
                                                                GXF_ENTITY_CREATE_PROGRAM_BIT};
      HOLOSCAN_GXF_CALL_FATAL(
          GxfCreateEntity(context, &broadcast_entity_create_info, &broadcast_eid));

      // Add the broadcast_eid for the current operator and the source port name
      broadcast_eids[op][source_cname] = broadcast_eid;

      switch (connector_type) {
        case IOSpec::ConnectorType::kDefault:
        case IOSpec::ConnectorType::kDoubleBuffer:
        case IOSpec::ConnectorType::kUCX:  // In any case, need to add doubleBufferReceiver.
        {
          // We don't create a holoscan::AnnotatedDoubleBufferReceiver even if data flow
          // tracking is on because we don't want to mark annotations for the Broadcast
          // component.
          if (rx_double_buffer_tid == GxfTidNull()) {
            HOLOSCAN_GXF_CALL_FATAL(GxfComponentTypeId(
                context, "nvidia::gxf::DoubleBufferReceiver", &rx_double_buffer_tid));
          }
          rx_tid = rx_double_buffer_tid;

          // Get the connector capacity and policy of the current operator's output port.
          nvidia::gxf::DoubleBufferReceiver* curr_double_buffer_connector = nullptr;

          HOLOSCAN_GXF_CALL_FATAL(
              GxfComponentPointer(context,
                                  source_cid,
                                  rx_tid,
                                  reinterpret_cast<void**>(&curr_double_buffer_connector)));

          curr_connector_capacity = curr_double_buffer_connector->capacity_;
          curr_connector_policy = curr_double_buffer_connector->policy_;
        } break;
        default:
          HOLOSCAN_LOG_ERROR("Unrecognized connector_type '{}' for source name '{}'",
                             static_cast<int>(connector_type),
                             source_cname);
      }
      gxf_uid_t rx_cid;
      HOLOSCAN_GXF_CALL_FATAL(GxfComponentAdd(context, broadcast_eid, rx_tid, "", &rx_cid));
      // Set capacity and policy of the receiver component.
      HOLOSCAN_GXF_CALL_FATAL(
          GxfParameterSetUInt64(context, rx_cid, "capacity", curr_connector_capacity));
      HOLOSCAN_GXF_CALL_FATAL(
          GxfParameterSetUInt64(context, rx_cid, "policy", curr_connector_policy));

      if (rx_term_tid == GxfTidNull()) {
        HOLOSCAN_GXF_CALL_FATAL(GxfComponentTypeId(
            context, "nvidia::gxf::MessageAvailableSchedulingTerm", &rx_term_tid));
      }
      gxf_uid_t rx_term_cid;
      HOLOSCAN_GXF_CALL_FATAL(
          GxfComponentAdd(context, broadcast_eid, rx_term_tid, "", &rx_term_cid));
      HOLOSCAN_GXF_CALL_FATAL(GxfParameterSetHandle(context, rx_term_cid, "receiver", rx_cid));
      HOLOSCAN_GXF_CALL_FATAL(
          GxfParameterSetUInt64(context, rx_term_cid, "min_size", curr_min_size));

      if (broadcast_tid == GxfTidNull()) {
        HOLOSCAN_GXF_CALL_FATAL(
            GxfComponentTypeId(context, "nvidia::gxf::Broadcast", &broadcast_tid));
      }
      gxf_uid_t broadcast_cid;
      auto broadcast_component_name =
          fmt::format("{}_broadcast_component_{}_{}", entity_prefix, op_name, source_cname);
      HOLOSCAN_GXF_CALL_FATAL(GxfComponentAdd(
          context, broadcast_eid, broadcast_tid, broadcast_component_name.c_str(), &broadcast_cid));
      HOLOSCAN_GXF_CALL_FATAL(GxfParameterSetHandle(context, broadcast_cid, "source", rx_cid));

      HOLOSCAN_LOG_DEBUG(
          "Connected MessageAvailableSchedulingTerm to receiver for Broadcast entity : {}",
          broadcast_entity_name);

      // Connect Broadcast entity's receiver with the transmitter of the current operator
      ::holoscan::gxf::add_connection(context, source_cid, rx_cid);
    }
  }
}

}  // unnamed namespace

bool GXFExecutor::initialize_fragment() {
  HOLOSCAN_LOG_DEBUG("Initializing Fragment.");

  auto context = context_;

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

  // Keep a list of all the broadcast entity ids, if an operator's output port is connected to
  // multiple inputs. The map is indexed by the operators. Each value in the map is indexed by the
  // source port name
  BroadcastEidMapType broadcast_eids;

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
        HOLOSCAN_LOG_ERROR(
            "Worklist is empty, but not all nodes have been visited. There is a cycle.");
        HOLOSCAN_LOG_ERROR("Application is being aborted.");
        // Holoscan supports DAG for fragments. Cycles will be supported in a future release.
        return false;
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
    op->initialize();
    auto op_type = op->operator_type();

    HOLOSCAN_LOG_DEBUG("Connecting earlier operators of Op: {}", op_name);
    // Add the connections from the previous operator to the current operator, for both direct and
    // Broadcast connections.
    auto prev_operators = graph.get_previous_nodes(op);

    for (auto& prev_op : prev_operators) {
      auto prev_gxf_op = std::dynamic_pointer_cast<ops::GXFOperator>(prev_op);

      auto port_map = graph.get_port_map(prev_op, op);
      if (!port_map.has_value()) {
        HOLOSCAN_LOG_ERROR("Could not find port map for {} -> {}", prev_op->name(), op->name());
        return false;
      }

      const auto& port_map_val = port_map.value();

      // If the previous operator is found to be one that is connected to the current operator via
      // the Broadcast component, then add the connection between the Broadcast component and the
      // current operator's input port.
      if (broadcast_eids.find(prev_op) != broadcast_eids.end()) {
        // Add transmitter to the prev_op's broadcast component and connect it to op's input port.
        // Any connected ports are removed from port_map_val.
        connect_broadcast_to_previous_op(
            context_, fragment_, broadcast_eids, op, prev_op, port_map_val);
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
          auto source_gxf_resource = std::dynamic_pointer_cast<GXFResource>(
              prev_op->spec()->outputs()[source_port]->connector());
          gxf_uid_t source_cid = source_gxf_resource->gxf_cid();
          if (target_ports.size() > 1) {
            HOLOSCAN_LOG_ERROR(
                "Source port is connected to multiple target ports without Broadcast component. "
                "Op: {} source name: {}",
                op->name(),
                source_port);
            return false;
          }
          // GXF Connection component should not be added for types using a NetworkContext
          auto connector_type = prev_op->spec()->outputs()[source_port]->connector_type();
          if (connector_type != IOSpec::ConnectorType::kUCX) {
            const auto& target_port = target_ports.begin();
            auto target_gxf_resource = std::dynamic_pointer_cast<GXFResource>(
                op->spec()->inputs()[*target_port]->connector());
            gxf_uid_t target_cid = target_gxf_resource->gxf_cid();
            ::holoscan::gxf::add_connection(context, source_cid, target_cid);
            HOLOSCAN_LOG_DEBUG(
                "Connected directly source : {} -> target : {}", source_port, *target_port);
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

            auto connector_type = op_spec->outputs()[source_port]->connector_type();
            if (connections.find(source_cid) == connections.end()) {
              connections[source_cid] = TargetsInfo{connector_type, std::set<TargetPort>{}};
            }
            // For the source port, add a target in the tuple form (next operator, receiving port
            // name)
            connections[source_cid].second.insert(std::make_pair(next_op, target_port));
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

    // Create the Broadcast components and add their IDs to broadcast_eids, but do not add any
    // transmitter to the Broadcast entity. The transmitters will be added later when the incoming
    // edges to the respective operators are processed.
    create_broadcast_components(context, entity_prefix_, op, broadcast_eids, connections);
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
  HOLOSCAN_LOG_DEBUG("Initializing Operator '{}'", op->name());

  if (!op->spec()) {
    HOLOSCAN_LOG_ERROR("No operator spec for GXFOperator '{}'", op->name());
    return false;
  }

  // If the type name is not set, the operator is assumed to be a Holoscan native operator and use
  // `holoscan::gxf::GXFWrapper` as the GXF Codelet.
  const bool is_native_operator = (op->operator_type() == Operator::OperatorType::kNative);
  ops::GXFOperator* gxf_op = static_cast<ops::GXFOperator*>(op);

  const char* codelet_typename = nullptr;
  if (is_native_operator) {
    codelet_typename = "holoscan::gxf::GXFWrapper";
  } else {
    codelet_typename = gxf_op->gxf_typename();
  }

  auto& spec = *(op->spec());

  gxf_uid_t eid = 0;

  // Create Entity for the operator if `op_eid_` is 0
  if (op_eid_ == 0) {
    const std::string op_entity_name = fmt::format("{}{}", entity_prefix_, op->name());
    const GxfEntityCreateInfo entity_create_info = {op_entity_name.c_str(),
                                                    GXF_ENTITY_CREATE_PROGRAM_BIT};
    HOLOSCAN_GXF_CALL_MSG_FATAL(
        GxfCreateEntity(context_, &entity_create_info, &eid),
        "Unable to create GXF entity for operator '{}'. Please check if the "
        "operator name is unique.",
        op->name());
  } else {
    eid = op_eid_;
  }

  gxf_uid_t codelet_cid;
  // Create Codelet component if `op_cid_` is 0
  if (op_cid_ == 0) {
    gxf_tid_t codelet_tid;
    HOLOSCAN_GXF_CALL(GxfComponentTypeId(context_, codelet_typename, &codelet_tid));
    HOLOSCAN_GXF_CALL_FATAL(
        GxfComponentAdd(context_, eid, codelet_tid, op->name().c_str(), &codelet_cid));

    // Set the operator to the GXFWrapper if it is a native operator
    if (is_native_operator) {
      holoscan::gxf::GXFWrapper* gxf_wrapper = nullptr;
      HOLOSCAN_GXF_CALL_FATAL(GxfComponentPointer(
          context_, codelet_cid, codelet_tid, reinterpret_cast<void**>(&gxf_wrapper)));
      if (gxf_wrapper) {
        gxf_wrapper->set_operator(op);
      } else {
        HOLOSCAN_LOG_ERROR("Unable to get GXFWrapper for Operator '{}'", op->name());
      }
    } else {
      // Set the entity id
      gxf_op->gxf_eid(eid);
      // Set the codelet component id
      gxf_op->gxf_cid(codelet_cid);
    }
  } else {
    codelet_cid = op_cid_;
  }

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

  // Create Components for condition
  for (const auto& [name, condition] : op->conditions()) {
    auto gxf_condition = std::dynamic_pointer_cast<gxf::GXFCondition>(condition);
    // Initialize GXF component if it is not already initialized.
    if (gxf_condition != nullptr && gxf_condition->gxf_context() == nullptr) {
      gxf_condition->fragment(fragment());

      gxf_condition->gxf_eid(eid);  // set GXF entity id
    }
    // Initialize condition
    gxf_condition->initialize();
  }

  // Create Components for resource
  for (const auto& [name, resource] : op->resources()) {
    auto gxf_resource = std::dynamic_pointer_cast<gxf::GXFResource>(resource);
    // Initialize GXF component if it is not already initialized.
    if (gxf_resource != nullptr && gxf_resource->gxf_context() == nullptr) {
      gxf_resource->fragment(fragment());

      gxf_resource->gxf_eid(eid);  // set GXF entity id
    }
    // Initialize resource
    resource->initialize();
  }

  // Set arguments
  auto& params = spec.params();
  for (auto& arg : op->args()) {
    // Find if arg.name() is in spec_->params()
    if (params.find(arg.name()) == params.end()) {
      HOLOSCAN_LOG_WARN("Argument '{}' is not defined in spec", arg.name());
      continue;
    }

    // Set arg.value() to spec_->params()[arg.name()]
    auto& param_wrap = params[arg.name()];

    HOLOSCAN_LOG_TRACE("GXFOperator '{}':: setting argument '{}'", op->name(), arg.name());

    ArgumentSetter::set_param(param_wrap, arg);
  }

  // Set Handler parameters if it is an operator that wraps an existing GXF Codelet.
  if (!is_native_operator) {
    // Set Handler parameters
    for (auto& [key, param_wrap] : params) {
      HOLOSCAN_GXF_CALL_WARN_MSG(::holoscan::gxf::GXFParameterAdaptor::set_param(
                                     context_, codelet_cid, key.c_str(), param_wrap),
                                 "GXFOperator '{}':: failed to set GXF parameter '{}'",
                                 op->name(),
                                 key);
      HOLOSCAN_LOG_TRACE("GXFOperator '{}':: setting GXF parameter '{}'", op->name(), key);
    }
  } else {
    // Set only default parameter values
    for (auto& [key, param_wrap] : params) {
      // If no value is specified, the default value will be used by setting an empty argument.
      Arg empty_arg("");
      ArgumentSetter::set_param(param_wrap, empty_arg);
    }
  }
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
  // TODO: Currently, there is no convenient API to set the condition of the receivers (input
  // ports)
  //       from the setup() method of the operator. We need to add a new API to set the condition
  //       of the receivers (input ports) from the setup() method of the operator.

  // Add the new input port to the vector.
  iospec_vector.push_back(&input_port);

  // IOSpec vector is added, the parameters will be initialized in the initialize_operator()
  // function, when all the parameters are initialized.

  // Add new label to the label vector so that the port map of the graph edge can be updated.
  new_input_labels.push_back(new_input_label);

  return true;
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

  // Load extensions from config file only if not already loaded,
  // to avoid unnecessary loading on multiple run() calls.
  if (!is_extensions_loaded_) {
    HOLOSCAN_LOG_INFO("Loading extensions from configs...");
    // Load extensions from config file if exists.
    for (const auto& yaml_node : fragment_->config().yaml_nodes()) {
      gxf_extension_manager_->load_extensions_from_yaml(yaml_node);
    }
    is_extensions_loaded_ = true;
  }

  {
    // Lock the GXF context for execution
    std::scoped_lock lock{gxf_execution_mutex};

    // Additional setup for GXF Application
    gxf_uid_t eid;
    const std::string utility_entity_name = fmt::format("{}_holoscan_util_entity", entity_prefix_);
    const GxfEntityCreateInfo entity_create_info = {utility_entity_name.c_str(),
                                                    GXF_ENTITY_CREATE_PROGRAM_BIT};
    HOLOSCAN_GXF_CALL_FATAL(GxfCreateEntity(context, &entity_create_info, &eid));

    auto scheduler = std::dynamic_pointer_cast<gxf::GXFScheduler>(fragment_->scheduler());
    // have to set the application eid before initialize() can be called
    scheduler->gxf_eid(eid);
    scheduler->initialize();

    // Initialize the fragment and its operators
    if (!initialize_fragment()) {
      HOLOSCAN_LOG_ERROR("Failed to initialize fragment");
      return false;
    }

    // If DFFT is on, then attach the DFFTCollector EntityMonitor to the main entity
    if (fragment_->data_flow_tracker()) {
      gxf_tid_t monitor_tid;
      HOLOSCAN_GXF_CALL_FATAL(GxfComponentTypeId(context, "holoscan::DFFTCollector", &monitor_tid));

      gxf_uid_t monitor_cid;
      HOLOSCAN_GXF_CALL_FATAL(
          GxfComponentAdd(context, eid, monitor_tid, "dft_tracker", &monitor_cid));

      holoscan::DFFTCollector* dfft_collector_ptr = nullptr;
      HOLOSCAN_GXF_CALL_FATAL(GxfComponentPointer(
          context, monitor_cid, monitor_tid, reinterpret_cast<void**>(&dfft_collector_ptr)));
      if (!dfft_collector_ptr) {
        throw std::runtime_error(
            fmt::format("Unable to retrieve holoscan::DFFTCollector pointer."));
      }

      dfft_collector_ptr->data_flow_tracker(fragment_->data_flow_tracker());

      // Identify leaf and root operators and add to the DFFTCollector object
      for (auto op : graph.get_nodes()) {
        if (op->is_leaf()) {
          dfft_collector_ptr->add_leaf_op(op.get());
        } else if (op->is_root()) {
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

      std::string entity_group_name = "network_entity_group";
      auto entity_group_gid = add_entity_group(context_, entity_group_name);

      std::string device_entity_name = fmt::format("{}gpu_device_entity", entity_prefix_);
      std::string device_component_name = "gpu_device_component";
      auto [gpu_device_tid, gpu_device_eid] = create_gpu_device_entity(context, device_entity_name);
      create_gpu_device_component(
          context, gpu_device_tid, gpu_device_eid, device_component_name, 0);

      // Note: GxfUpdateEntityGroup
      //   calls Runtime::GxfUpdateEntityGroup(gid, eid)
      //     which calls  EntityGroups::groupAddEntity(gid, eid); (entity_groups_ in SharedContext)
      //       which calls EntityGroupItem::addEntity for the EntityGroupItem corresponding to gid
      //         any eid corresponding to a ResourceBase class like GPUDevice or ThreadPool is
      //             stored in internal resources_ vector
      //         all other eid are stored in the entities vector

      // add GPUDevice resource to the networking entity group
      GXF_ASSERT_SUCCESS(GxfUpdateEntityGroup(context_, entity_group_gid, gpu_device_eid));

      // add the network context to the entity group
      auto gxf_network_context =
          std::dynamic_pointer_cast<holoscan::gxf::GXFNetworkContext>(fragment_->network_context());
      HOLOSCAN_GXF_CALL_FATAL(
          GxfUpdateEntityGroup(context, entity_group_gid, gxf_network_context->gxf_eid()));

      // Loop through all operators and add any operators with a UCX port to the entity group
      auto operator_graph = static_cast<OperatorFlowGraph&>(fragment_->graph());
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

    } else {
      HOLOSCAN_LOG_DEBUG("GXFExecutor::run: no NetworkContext to initialize");

      // Loop through all operator ports and raise an error if any are UCX-based.
      // (UCX-based connections require a UcxContext).
      auto operator_graph = static_cast<OperatorFlowGraph&>(fragment_->graph());
      for (auto& node : operator_graph.get_nodes()) {
        auto op_spec = node->spec();
        for (const auto& [_, io_spec] : op_spec->inputs()) {
          if (io_spec->connector_type() == IOSpec::ConnectorType::kUCX) {
            throw std::runtime_error("UCX-based connection found, but there is no NetworkContext.");
          }
        }
        for (const auto& [_, io_spec] : op_spec->outputs()) {
          if (io_spec->connector_type() == IOSpec::ConnectorType::kUCX) {
            throw std::runtime_error("UCX-based connection found, but there is no NetworkContext.");
          }
        }
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

bool GXFExecutor::run_gxf_graph() {
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
      SignalHandler::register_global_signal_handler(signum, [](int signum) {
        (void)signum;
        HOLOSCAN_LOG_ERROR("Interrupted by user (global signal handler)");
        exit(1);
      });
    }
  };
  SignalHandler::register_signal_handler(context, SIGINT, sig_handler);
  SignalHandler::register_signal_handler(context, SIGTERM, sig_handler);

  // Run the graph
  activate_gxf_graph();
  HOLOSCAN_LOG_INFO("Running Graph...");
  HOLOSCAN_GXF_CALL_FATAL(GxfGraphRunAsync(context));
  HOLOSCAN_LOG_INFO("Waiting for completion...");
  HOLOSCAN_LOG_INFO("Graph execution waiting. Fragment: {}", fragment_->name());
  auto wait_result = HOLOSCAN_GXF_CALL_WARN(GxfGraphWait(context));
  if (wait_result != GXF_SUCCESS) {
    // Usually the graph is already deactivated when GxfGraphWait() fails.
    is_gxf_graph_activated_ = false;
    HOLOSCAN_LOG_ERROR("GxfGraphWait Error: {}", GxfResultStr(wait_result));
    throw RuntimeError(ErrorCode::kFailure, "Failed to wait for graph to complete");
  }

  HOLOSCAN_LOG_INFO("Graph execution deactivating. Fragment: {}", fragment_->name());
  HOLOSCAN_LOG_INFO("Deactivating Graph...");
  HOLOSCAN_GXF_CALL_WARN(GxfGraphDeactivate(context));
  is_gxf_graph_activated_ = false;
  HOLOSCAN_LOG_INFO("Graph execution finished. Fragment: {}", fragment_->name());
  return true;
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
    gxf_extension_manager_->load_extension(gxf_extension_file_name);
  }

  // Register the default Holoscan GXF extensions
  for (auto& gxf_extension_file_name : kDefaultHoloscanGXFExtensions) {
    gxf_extension_manager_->load_extension(gxf_extension_file_name);
  }

  // Register the GXF extension that provides the native operators
  gxf_tid_t gxf_wrapper_tid{0xd4e7c16bcae741f8, 0xa5eb93731de9ccf6};

  if (!gxf_extension_manager_->is_extension_loaded(gxf_wrapper_tid)) {
    GXFExtensionRegistrar extension_factory(
        context_,
        "HoloscanSdkInternalExtension",
        "A runtime hidden extension used by Holoscan SDK to provide the native operators",
        gxf_wrapper_tid);

    extension_factory.add_component<holoscan::gxf::GXFWrapper, nvidia::gxf::Codelet>(
        "GXF wrapper to support Holoscan SDK native operators");
    extension_factory.add_type<holoscan::Message>("Holoscan message type",
                                                  {0x61510ca06aa9493b, 0x8a777d0bf87476b7});

    extension_factory.add_component<holoscan::gxf::GXFTensor, nvidia::gxf::Tensor>(
        "Holoscan's GXF Tensor type", {0xa02945eaf20e418c, 0x8e6992b68672ce40});
    extension_factory.add_type<holoscan::Tensor>("Holoscan's Tensor type",
                                                 {0xa5eb0ed57d7f4aa2, 0xb5865ccca0ef955c});

    // Add a new type of Double Buffer Receiver and Tramsmitter
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

  auto& spec = *(sch->spec());

  gxf_uid_t eid = 0;
  // Create Entity for the scheduler if `op_eid_` is 0
  if (op_eid_ == 0) {
    const std::string scheduler_entity_name = fmt::format("{}{}", entity_prefix_, sch->name());
    const GxfEntityCreateInfo entity_create_info = {scheduler_entity_name.c_str(),
                                                    GXF_ENTITY_CREATE_PROGRAM_BIT};
    HOLOSCAN_GXF_CALL_MSG_FATAL(
        GxfCreateEntity(context_, &entity_create_info, &eid),
        "Unable to create GXF entity for scheduler '{}'. Please check if the "
        "scheduler name is unique.",
        sch->name());
  } else {
    eid = op_eid_;
  }

  gxf_uid_t scheduler_cid;
  // Create Codelet component if `op_cid_` is 0
  if (op_cid_ == 0) {
    gxf_tid_t scheduler_tid;
    HOLOSCAN_GXF_CALL_FATAL(GxfComponentTypeId(context_, gxf_sch->gxf_typename(), &scheduler_tid));
    HOLOSCAN_GXF_CALL_FATAL(
        GxfComponentAdd(context_, eid, scheduler_tid, sch->name().c_str(), &scheduler_cid));

    // Set the entity id
    gxf_sch->gxf_eid(eid);
    // Set the scheduler component id
    gxf_sch->gxf_cid(scheduler_cid);
  } else {
    scheduler_cid = op_cid_;
  }

  // Set GXF Scheduler ID as the ID of the scheduler
  sch->id(scheduler_cid);

  // Create Components for resource
  for (const auto& [name, resource] : sch->resources()) {
    auto gxf_resource = std::dynamic_pointer_cast<gxf::GXFResource>(resource);
    // Initialize GXF component if it is not already initialized.
    if (gxf_resource->gxf_context() == nullptr) {
      gxf_resource->fragment(fragment());

      gxf_resource->gxf_eid(eid);  // set GXF entity id
      gxf_resource->initialize();
    }
  }

  // Set arguments
  auto& params = spec.params();
  for (auto& arg : sch->args()) {
    // Find if arg.name() is in spec_->params()
    if (params.find(arg.name()) == params.end()) {
      HOLOSCAN_LOG_WARN("Argument '{}' is not defined in spec", arg.name());
      continue;
    }

    // Set arg.value() to spec_->params()[arg.name()]
    auto& param_wrap = params[arg.name()];

    HOLOSCAN_LOG_TRACE("GXFScheduler '{}':: setting argument '{}'", sch->name(), arg.name());

    ArgumentSetter::set_param(param_wrap, arg);
  }

  // Set Handler parameters
  for (auto& [key, param_wrap] : params) {
    HOLOSCAN_LOG_TRACE("GXFScheduler '{}':: setting GXF parameter '{}'", sch->name(), key);
    HOLOSCAN_GXF_CALL_WARN_MSG(::holoscan::gxf::GXFParameterAdaptor::set_param(
                                   context_, scheduler_cid, key.c_str(), param_wrap),
                               "GXFScheduler '{}':: failed to set GXF parameter '{}'",
                               sch->name(),
                               key);
    // TODO: handle error
  }

  return true;
}

bool GXFExecutor::initialize_network_context(NetworkContext* network_context) {
  if (!network_context->spec()) {
    HOLOSCAN_LOG_ERROR("No component spec for GXFNetworkContext '{}'", network_context->name());
    return false;
  }

  gxf::GXFNetworkContext* gxf_network_context =
      static_cast<gxf::GXFNetworkContext*>(network_context);

  auto& spec = *(network_context->spec());

  gxf_uid_t eid = 0;

  // Create Entity for the network_context if `op_eid_` is 0
  if (op_eid_ == 0) {
    const std::string network_context_entity_name =
        fmt::format("{}{}", entity_prefix_, network_context->name());
    const GxfEntityCreateInfo entity_create_info = {network_context_entity_name.c_str(),
                                                    GXF_ENTITY_CREATE_PROGRAM_BIT};
    HOLOSCAN_GXF_CALL_MSG_FATAL(
        GxfCreateEntity(context_, &entity_create_info, &eid),
        "Unable to create GXF entity for scheduler '{}'. Please check if the "
        "scheduler name is unique.",
        network_context->name());
  } else {
    eid = op_eid_;
  }

  gxf_uid_t network_context_cid;
  // Create Codelet component if `op_cid_` is 0
  if (op_cid_ == 0) {
    gxf_tid_t network_context_tid;
    HOLOSCAN_GXF_CALL_FATAL(
        GxfComponentTypeId(context_, gxf_network_context->gxf_typename(), &network_context_tid));
    HOLOSCAN_GXF_CALL_FATAL(GxfComponentAdd(
        context_, eid, network_context_tid, network_context->name().c_str(), &network_context_cid));

    // Set the entity id
    gxf_network_context->gxf_eid(eid);
    // Set the network_context component id
    gxf_network_context->gxf_cid(network_context_cid);
  } else {
    network_context_cid = op_cid_;
  }

  // Set GXF NetworkContext ID as the ID of the network_context
  network_context->id(network_context_cid);

  // Create Components for resource
  for (const auto& [name, resource] : network_context->resources()) {
    auto gxf_resource = std::dynamic_pointer_cast<gxf::GXFResource>(resource);
    // Initialize GXF component if it is not already initialized.
    if (gxf_resource->gxf_context() == nullptr) {
      gxf_resource->fragment(fragment());

      gxf_resource->gxf_eid(eid);  // set GXF entity id
      gxf_resource->initialize();
    }
  }

  // Set arguments
  auto& params = spec.params();
  for (auto& arg : network_context->args()) {
    // Find if arg.name() is in spec_->params()
    if (params.find(arg.name()) == params.end()) {
      HOLOSCAN_LOG_WARN("Argument '{}' is not defined in spec", arg.name());
      continue;
    }

    // Set arg.value() to spec_->params()[arg.name()]
    auto& param_wrap = params[arg.name()];

    HOLOSCAN_LOG_TRACE(
        "GXFNetworkContext '{}':: setting argument '{}'", network_context->name(), arg.name());

    ArgumentSetter::set_param(param_wrap, arg);
  }

  // Set Handler parameters
  for (auto& [key, param_wrap] : params) {
    HOLOSCAN_LOG_TRACE(
        "GXFNetworkContext '{}':: setting GXF parameter '{}'", network_context->name(), key);
    HOLOSCAN_GXF_CALL_WARN_MSG(::holoscan::gxf::GXFParameterAdaptor::set_param(
                                   context_, network_context_cid, key.c_str(), param_wrap),
                               "GXFNetworkContext '{}':: failed to set GXF parameter '{}'",
                               network_context->name(),
                               key);
  }

  return true;
}

}  // namespace holoscan::gxf
