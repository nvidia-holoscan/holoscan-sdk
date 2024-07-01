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

#ifndef HOLOSCAN_CORE_IO_SPEC_HPP
#define HOLOSCAN_CORE_IO_SPEC_HPP

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "./condition.hpp"
#include "./conditions/gxf/asynchronous.hpp"
#include "./conditions/gxf/boolean.hpp"
#include "./conditions/gxf/count.hpp"
#include "./conditions/gxf/downstream_affordable.hpp"
#include "./conditions/gxf/periodic.hpp"
#include "./conditions/gxf/message_available.hpp"
#include "./conditions/gxf/expiring_message.hpp"
#include "./resources/gxf/double_buffer_receiver.hpp"
#include "./resources/gxf/double_buffer_transmitter.hpp"
#include "./resources/gxf/ucx_receiver.hpp"
#include "./resources/gxf/ucx_transmitter.hpp"
#include "./resource.hpp"
#include "./gxf/entity.hpp"
#include "./common.hpp"

namespace holoscan {

/**
 * @brief Class to define the specification of an input/output port of an Operator.
 *
 * An interaction point between two operators. Operators ingest data at Input ports and publish data
 * at Output ports. Receiver, Transmitter, and MessageRouter in GXF would be replaced with the
 * concept of Input/Output Port of the Operator and the Flow (Edge) of the Application Workflow
 * in the Framework.
 */
class IOSpec {
 public:
  /**
   * @brief Input/Output type.
   */
  enum class IOType { kInput, kOutput };

  /**
   * @brief Connector type. Determines the type of Receiver (when IOType is kInput) or Transmitter
   *        (when IOType is kOutput) class used.
   */
  enum class ConnectorType { kDefault, kDoubleBuffer, kUCX };

  /**
   * @brief Construct a new IOSpec object.
   *
   * @param op_spec The pointer to the operator specification that contains this input/output.
   * @param name The name of this input/output.
   * @param io_type The type of this input/output.
   */
  IOSpec(OperatorSpec* op_spec, const std::string& name, IOType io_type)
      : op_spec_(op_spec),
        name_(name),
        io_type_(io_type),
        typeinfo_(&typeid(holoscan::gxf::Entity)) {
    // Operator::parse_port_name requires that "." is not allowed in the IOSPec name
    if (name.find(".") != std::string::npos) {
      throw std::invalid_argument(fmt::format(
          "The . character is reserved and cannot be used in the port (IOSpec) name ('{}').",
          name));
    }
    name_ = name;
  }

  /**
   * @brief Construct a new IOSpec object.
   *
   * @param op_spec The pointer to the operator specification that contains this input/output.
   * @param name The name of this input/output.
   * @param io_type The type of this input/output.
   * @param typeinfo The type info of the data of this input/output.
   */
  IOSpec(OperatorSpec* op_spec, const std::string& name, IOType io_type,
         const std::type_info* typeinfo)
      : op_spec_(op_spec), io_type_(io_type), typeinfo_(typeinfo) {
    // Operator::parse_port_name requires that "." is not allowed in the IOSPec name
    if (name.find(".") != std::string::npos) {
      throw std::invalid_argument(fmt::format(
          "The . character is reserved and cannot be used in the port (IOSpec) name ('{}').",
          name));
    }
    name_ = name;
  }

  /**
   * @brief Get the operator specification that contains this input/output.
   *
   * @return The pointer to the operator specification that contains this input/output.
   */
  OperatorSpec* op_spec() const { return op_spec_; }

  /**
   * @brief Get the name of this input/output.
   *
   * @return The name of this input/output.
   */
  const std::string& name() const { return name_; }

  /**
   * @brief Get the input/output type.
   *
   * @return The input/output type.
   */
  IOType io_type() const { return io_type_; }

  /**
   * @brief Get the receiver/transmitter type.
   *
   * @return The receiver type (for inputs) or transmitter type (for outputs)
   */
  ConnectorType connector_type() const { return connector_type_; }

  /**
   * @brief Get the type info of the data of this input/output.
   *
   * @return The type info of the data of this input/output.
   */
  const std::type_info* typeinfo() const { return typeinfo_; }

  /**
   * @brief Get the conditions of this input/output.
   *
   * @return The reference to the conditions of this input/output.
   */
  std::vector<std::pair<ConditionType, std::shared_ptr<Condition>>>& conditions() {
    return conditions_;
  }

  /**
   * @brief Add a condition to this input/output.
   *
   * The following ConditionTypes are supported:
   *
   * - ConditionType::kMessageAvailable
   * - ConditionType::kDownstreamAffordable
   * - ConditionType::kNone
   *
   * @param type The type of the condition.
   * @param args The arguments of the condition.
   *
   * @return The reference to this IOSpec.
   */
  template <typename... ArgsT>
  IOSpec& condition(ConditionType type, ArgsT&&... args) {
    switch (type) {
      case ConditionType::kMessageAvailable:
        conditions_.emplace_back(
            type, std::make_shared<MessageAvailableCondition>(std::forward<ArgsT>(args)...));
        break;
      case ConditionType::kExpiringMessageAvailable:
        conditions_.emplace_back(
            type,
            std::make_shared<ExpiringMessageAvailableCondition>(std::forward<ArgsT>(args)...));
        break;
      case ConditionType::kDownstreamMessageAffordable:
        conditions_.emplace_back(
            type,
            std::make_shared<DownstreamMessageAffordableCondition>(std::forward<ArgsT>(args)...));
        break;
      case ConditionType::kNone:
        conditions_.emplace_back(type, nullptr);
        break;
      default:
        HOLOSCAN_LOG_ERROR("Unsupported condition type for IOSpec: {}", static_cast<int>(type));
        break;
    }
    return *this;
  }

  /**
   * @brief Get the connector (transmitter or receiver) of this input/output.
   *
   * @return The connector (transmitter or receiver) of this input/output.
   */
  std::shared_ptr<Resource> connector() const { return connector_; }

  /**
   * @brief Set the connector (transmitter or receiver) of this input/output.
   *
   * @param connector The connector (transmitter or receiver) of this input/output.
   */
  void connector(std::shared_ptr<Resource> connector) { connector_ = std::move(connector); }

  /**
   * @brief Add a connector (receiver/transmitter) to this input/output.
   *
   * The following ConnectorTypes are supported:
   *
   * - ConnectorType::kDefault
   * - ConnectorType::kDoubleBuffer
   * - ConnectorType::kUCX
   *
   * @param type The type of the connector (receiver/transmitter).
   * @param args The arguments of the connector (receiver/transmitter).
   *
   * @return The reference to this IOSpec.
   */
  template <typename... ArgsT>
  IOSpec& connector(ConnectorType type, ArgsT&&... args) {
    connector_type_ = type;
    switch (type) {
      case ConnectorType::kDefault:
        // default receiver or transmitter will be created in GXFExecutor::run instead
        break;
      case ConnectorType::kDoubleBuffer:
        if (io_type_ == IOType::kInput) {
          connector_ = std::make_shared<DoubleBufferReceiver>(std::forward<ArgsT>(args)...);
        } else {
          connector_ = std::make_shared<DoubleBufferTransmitter>(std::forward<ArgsT>(args)...);
        }
        break;
      case ConnectorType::kUCX:
        if (io_type_ == IOType::kInput) {
          connector_ = std::make_shared<UcxReceiver>(std::forward<ArgsT>(args)...);
        } else {
          connector_ = std::make_shared<UcxTransmitter>(std::forward<ArgsT>(args)...);
        }
        break;
      default:
        HOLOSCAN_LOG_ERROR("Unknown connector type {}", static_cast<int>(type));
        break;
    }
    return *this;
  }

  /**
   * @brief Get a YAML representation of the IOSpec.
   *
   * @return YAML node including the parameters of this component.
   */
  virtual YAML::Node to_yaml_node() const;

  /**
   * @brief Get a description of the IOSpec.
   *
   * @see to_yaml_node()
   * @return YAML string.
   */
  std::string description() const;

 private:
  OperatorSpec* op_spec_ = nullptr;
  std::string name_;
  IOType io_type_;
  const std::type_info* typeinfo_ = nullptr;
  std::shared_ptr<Resource> connector_;
  std::vector<std::pair<ConditionType, std::shared_ptr<Condition>>> conditions_;
  ConnectorType connector_type_ = ConnectorType::kDefault;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_IO_SPEC_HPP */
