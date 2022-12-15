/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <iostream>
#include <memory>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

#include "./conditions/gxf/boolean.hpp"
#include "./conditions/gxf/count.hpp"
#include "./conditions/gxf/downstream_affordable.hpp"
#include "./conditions/gxf/message_available.hpp"
#include "./gxf/entity.hpp"
#include "./common.hpp"
namespace holoscan {

/**
 * @brief Class to define the specification of an input/output port of an Operator.
 *
 * An interaction point between two operators. Operators ingest data at Input ports and publish data
 * at Output ports. Receiver, Transmitter, and MessageRouter in GXF would be replaced with the
 * concept of Input/Output Port of the Operator and the Flow (Edge) of the Application Workflow
 * (DAG) in the Framework.
 */
class IOSpec {
 public:
  /**
   * @brief Input/Output type.
   */
  enum class IOType { kInput, kOutput };

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
        typeinfo_(&typeid(holoscan::gxf::Entity)) {}

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
      : op_spec_(op_spec), name_(name), io_type_(io_type), typeinfo_(typeinfo) {}

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
   * @brief Get the type info of the data of this input/output.
   *
   * @return The type info of the data of this input/output.
   */
  const std::type_info* typeinfo() const { return typeinfo_; }

  /**
   * @brief Get the resource of this input/output.
   *
   * @return The resource of this input/output.
   */
  std::shared_ptr<Resource> resource() const { return resource_; }
  /**
   * @brief Set the resource of this input/output.
   *
   * @param resource The resource of this input/output.
   */
  void resource(std::shared_ptr<Resource> resource) { resource_ = resource; }

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
   * - ConditionType::kCount
   * - ConditionType::kBoolean
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
      case ConditionType::kDownstreamMessageAffordable:
        conditions_.emplace_back(
            type,
            std::make_shared<DownstreamMessageAffordableCondition>(std::forward<ArgsT>(args)...));
        break;
      case ConditionType::kCount:
        conditions_.emplace_back(type,
                                 std::make_shared<CountCondition>(std::forward<ArgsT>(args)...));
        break;
      case ConditionType::kBoolean:
        conditions_.emplace_back(type,
                                 std::make_shared<BooleanCondition>(std::forward<ArgsT>(args)...));
        break;
      case ConditionType::kNone:
        conditions_.emplace_back(type, nullptr);
        break;
    }
    return *this;
  }

 private:
  OperatorSpec* op_spec_ = nullptr;
  std::string name_;
  IOType io_type_;
  const std::type_info* typeinfo_ = nullptr;
  std::shared_ptr<Resource> resource_;
  std::vector<std::pair<ConditionType, std::shared_ptr<Condition>>> conditions_;
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_IO_SPEC_HPP */
