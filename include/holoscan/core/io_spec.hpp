/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <optional>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "./common.hpp"
#include "./condition.hpp"
#include "./conditions/gxf/asynchronous.hpp"
#include "./conditions/gxf/boolean.hpp"
#include "./conditions/gxf/count.hpp"
#include "./conditions/gxf/downstream_affordable.hpp"
#include "./conditions/gxf/expiring_message.hpp"
#include "./conditions/gxf/message_available.hpp"
#include "./conditions/gxf/multi_message_available_timeout.hpp"
#include "./conditions/gxf/periodic.hpp"
#include "./resource.hpp"
#include "./resources/gxf/double_buffer_receiver.hpp"
#include "./resources/gxf/double_buffer_transmitter.hpp"
#include "./resources/gxf/ucx_receiver.hpp"
#include "./resources/gxf/ucx_transmitter.hpp"

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
  virtual ~IOSpec() = default;

  /**
   * @brief Input/Output type.
   */
  enum class IOType { kInput, kOutput };

  /**
   * @brief Input/Output size.
   */
  class IOSize {
   public:
    /**
     * @brief Construct a new IOSize object.
     *
     * @param size The size of the input/output.
     */
    explicit IOSize(int64_t size = 0) : size_(size) {}

    /**
     * @brief Set the size of the input/output.
     *
     * @param size The new size of the input/output.
     */
    void size(int64_t size) { size_ = size; }

    /**
     * @brief Get the size of the input/output.
     *
     * @return The size of the input/output.
     */
    int64_t size() const { return size_; }

    /**
     * @brief Cast the IOSize to int64_t.
     *
     * @return The size of the input/output.
     */
    operator int64_t() const { return size_; }

   private:
    int64_t size_;
  };

  // Define the static constants for the IOSize class.
  inline static const IOSize kAnySize = IOSize{-1};        ///< Any size
  inline static const IOSize kPrecedingCount = IOSize{0};  ///< # of preceding connections
  inline static const IOSize kSizeOne = IOSize{1};         ///< Size one

  /**
   * @brief Connector type. Determines the type of Receiver (when IOType is kInput) or Transmitter
   *        (when IOType is kOutput) class used.
   */
  enum class ConnectorType { kDefault, kDoubleBuffer, kUCX };

  /**
   * @enum QueuePolicy
   * @brief Enum class representing the policy for handling queue operations.
   *
   * This enum class defines the different policies that can be applied to queue operations when
   * a queue is full.
   *
   * @var QueuePolicy::kPop
   * Policy to pop the oldest item in the queue so the new item can be added.
   *
   * @var QueuePolicy::kReject
   * Policy to reject the incoming item.
   *
   * @var QueuePolicy::kFault
   * Policy to log a warning and reject the new item if the queue is full.
   */
  enum class QueuePolicy : uint8_t {
    kPop = 0,
    kReject = 1,
    kFault = 2,
  };

  /**
   * @brief Construct a new IOSpec object.
   *
   * @param op_spec The pointer to the operator specification that contains this input/output.
   * @param name The name of this input/output.
   * @param io_type The type of this input/output.
   * @param typeinfo The type info of the data of this input/output.
   * @param size The size of the input/output queue.
   */
  IOSpec(OperatorSpec* op_spec, const std::string& name, IOType io_type,
         const std::type_info* typeinfo = &typeid(void*), IOSpec::IOSize size = IOSpec::kSizeOne,
         std::optional<IOSpec::QueuePolicy> policy = std::nullopt)
      : op_spec_(op_spec),
        io_type_(io_type),
        typeinfo_(typeinfo),
        queue_size_(size),
        queue_policy_(policy) {
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
   * - ConditionType::kExpiringMessageAvailable
   * - ConditionType::kDownstreamAffordable
   * - ConditionType::kMultiMessageAvailableTimeout
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
      case ConditionType::kMultiMessageAvailableTimeout:
        // May want to use this multi-message condition even with a single port as a way to have
        // a timeout on the condition. Unlike ExpiringMessageAvailableCondition, this one does not
        // require a timestamp to be emitted by the upstream operator.
        conditions_.emplace_back(
            type,
            std::make_shared<MultiMessageAvailableTimeoutCondition>(std::forward<ArgsT>(args)...));
        break;
      case ConditionType::kNone:
        conditions_.emplace_back(type, nullptr);
        break;
      default:
        HOLOSCAN_LOG_ERROR("Unsupported condition type for IOSpec: {}", static_cast<int>(type));
        break;
    }

    if (queue_size_ == kAnySize) {
      HOLOSCAN_LOG_WARN(
          "The queue size is currently set to 'any size' (IOSpec::kAnySize in C++ or "
          "IOSpec.ANY_SIZE in Python) "
          "for receivers that don't support condition changes. Please set the queue size "
          "explicitly when calling the input() method in setup() if you want to use the ordinary "
          "input port with the condition (input port: {}).",
          name_);
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
   * Note: Typically the application author does not need to call this method. The SDK will assign
   *       an appropriate transmitter or receiver type automatically (e.g. DoubleBufferReceiver or
   *       DoubleBufferTransmitter for with-fragment connections, but UcxReceiver or UcxTransmitter
   *       for intra-fragment connections in distributed applications). Similarly, annotated
   *       variants of these are used when data flow tracking is enabled.
   *
   * Note: If you just want to keep the default transmitter or receiver class, but override the
   *       queue capacity or policy, it is easier to specify the `capacity` and/or `policy`
   *       arguments to `IOSpec::input` or `IOSpec::output` instead of using this method.
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
        connector_.reset();
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

    if (queue_size_ == kAnySize) {
      HOLOSCAN_LOG_WARN(
          "The queue size is currently set to 'any size' (IOSpec::kAnySize in C++ or "
          "IOSpec.ANY_SIZE in Python) "
          "for receivers that don't support connector changes. Please set the queue size "
          "explicitly when calling the input() method in setup() if you want to use the ordinary "
          "input port with the condition (input port: {}).",
          name_);
    }
    return *this;
  }

  /**
   * @brief Get the queue size of the input/output port.
   *
   * Note: This value is only used for initializing input ports. 'queue_size_' is set by the
   *       'OperatorSpec::input()' method or the 'IOSpec::queue_size(int64_t)' method. If the queue
   *       size is set to 'any size' (IOSpec::kAnySize in C++ or IOSpec.ANY_SIZE in Python),
   *       the connector/condition settings will be ignored.
   *       If the queue size is set to other values, the default connector
   *       (DoubleBufferReceiver/UcxReceiver) and condition (MessageAvailableCondition) will use
   *       the queue size for initialization ('capacity' for the connector and 'min_size' for
   *       the condition) if they are not set.
   *
   * @return The queue size of the input/output port.
   */
  int64_t queue_size() const { return queue_size_.size(); }

  /**
   * @brief Set the queue size of the input/output port.
   *
   * Note: This value is only used for initializing input ports. 'queue_size_' is set by the
   *       'OperatorSpec::input()' method or this method. If the queue
   *       size is set to 'any size' (IOSpec::kAnySize in C++ or IOSpec.ANY_SIZE in Python),
   *       the connector/condition settings will be ignored.
   *       If the queue size is set to other values, the default connector
   *       (DoubleBufferReceiver/UcxReceiver) and condition (MessageAvailableCondition) will use
   *       the queue size for initialization ('capacity' for the connector and 'min_size' for
   *       the condition) if they are not set.
   *
   * @param size The queue size of the input/output port.
   * @return The reference to this IOSpec.
   */
  IOSpec& queue_size(int64_t size) {
    queue_size_.size(size);
    return *this;
  }

  /**
   * @brief Get the queue policy of the input/output port.
   *
   * Note: This value is only used for initializing input and output ports. 'queue_policy_' is set
   *       by the 'OperatorSpec::input()', 'OperatorSpec::output()' or 'IOSpec::queue_policy'
   *       method.
   *
   * @return The queue policy of the input/output port.
   */
  std::optional<IOSpec::QueuePolicy> queue_policy() const { return queue_policy_; }

  /**
   * @brief Set the queue policy of the input/output port.
   *
   * Note: This value is only used for initializing input and output ports. 'queue_policy_' is set
   *       by the 'OperatorSpec::input()', 'OperatorSpec::output()' or 'IOSpec::queue_policy'
   *       method.
   *
   * The following IOSpec::QueuePolicy values are supported:
   *
   * - QueuePolicy::kPop    - If the queue is full, pop the oldest item, then add the new one.
   * - QueuePolicy::kReject - If the queue is full, reject (discard) the new item.
   * - QueuePolicy::kFault  - If the queue is full, log a warning and reject the new item.
   *
   * @param policy The queue policy of the input/output port.
   * @return The reference to this IOSpec.
   */
  IOSpec& queue_policy(IOSpec::QueuePolicy policy) {
    queue_policy_ = policy;
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
  IOSize queue_size_ = kSizeOne;
  std::optional<QueuePolicy> queue_policy_ = std::nullopt;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_IO_SPEC_HPP */
