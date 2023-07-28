/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_SERVICES_COMMON_VIRTUAL_OPERATOR_HPP
#define HOLOSCAN_CORE_SERVICES_COMMON_VIRTUAL_OPERATOR_HPP

#include <string>
#include <utility>

#include "../../operator.hpp"

namespace holoscan::ops {

/**
 * @brief Virtual operator.
 *
 * This class is used when connecting fragments with transmitters and receivers (such as
 * UCXTransmitter and UCXReceiver) that communicate with other fragments.
 *
 * The input/output port of an operator in the fragment can be connected to this virtual operator so
 * that the internal graph initialization mechanism (such as implicit broadcasting or
 * multi-receivers) can be applied to the port of the operator connected to this operator in the
 * same fragment.
 */
class VirtualOperator : public holoscan::Operator {
 public:
  // HOLOSCAN_OPERATOR_FORWARD_ARGS(VirtualOperator)
  template <typename StringT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT>>>
  VirtualOperator(StringT port_name, IOSpec::ConnectorType connector_type, ArgList arg_list)
      : port_name_(port_name), connector_type_(connector_type), arg_list_(arg_list) {
    operator_type_ = OperatorType::kVirtual;
  }

  VirtualOperator() : Operator() { operator_type_ = OperatorType::kVirtual; }

  /**
   * @brief Initialize the Virtual operator.
   *
   * This function does not call the Operator::initialize() method that is called when the fragment
   * is initialized by Executor::initialize_fragment().
   *
   * Instead, it just sets the operator type to `holoscan::Operator::OperatorType::kVirtual`.
   */
  void initialize() override;

  /**
   * @brief Get the name of the port of the operator connected to this operator in the same
   * fragment.
   *
   * @return The name of the port of the operator connected to this operator in the same fragment.
   */
  const std::string& port_name() const { return port_name_; }

  /**
   * @brief Set the name of the port of the operator connected to this operator in the same
   * fragment.
   *
   * @param port_name The name of the port of the operator connected to this operator in the same
   * fragment.
   */
  void port_name(const std::string& port_name) { port_name_ = port_name; }

  /**
   * @brief Get the connector type of this operator.
   *
   * @return The connector type of this operator.
   */
  IOSpec::ConnectorType connector_type() const { return connector_type_; }

  /**
   * @brief Get the argument list of this operator.
   *
   * @return The argument list of this operator.
   */
  const ArgList& arg_list() const { return arg_list_; }

  /**
   * @brief Get the input specification for this operator.
   *
   * @return The pointer to the input specification for this operator.
   */
  IOSpec* input_spec();

  /**
   * @brief Get the output specification for this operator.
   *
   * @return The pointer to the output specification for this operator.
   */
  IOSpec* output_spec();

  /**
   * @brief Get the IO type of this operator.
   *
   * @return The IO type of this operator.
   */
  IOSpec::IOType io_type() const { return io_type_; }

 protected:
  /// The name of the port of the operator connected to this operator in the same fragment.
  std::string port_name_;
  /// The connector type of this operator.
  IOSpec::ConnectorType connector_type_;
  /// The argument list of this operator.
  ArgList arg_list_;

  /// The pointer to the input specification for this operator.
  IOSpec* input_spec_ = nullptr;
  /// The pointer to the output specification for this operator.
  IOSpec* output_spec_ = nullptr;

  IOSpec::IOType io_type_ = IOSpec::IOType::kInput;
};

/**
 * @brief Virtual Transmitter operator.
 *
 * This operator represents a transmitter that is connected to a receiver in another fragment.
 * Although this operator has an input port, the input port (and its name) of this operator
 * represents the output port (and its name) of an operator that is connected to this operator in
 * the same fragment.
 */
class VirtualTransmitterOp : public VirtualOperator {
 public:
  template <typename StringT, typename ArgListT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT> &&
                                        std::is_same_v<ArgList, std::decay_t<ArgListT>>>>
  explicit VirtualTransmitterOp(StringT&& output_port_name, IOSpec::ConnectorType connector_type,
                                ArgListT&& arg_list)
      : VirtualOperator(std::forward<StringT>(output_port_name), connector_type,
                        std::forward<ArgListT>(arg_list)) {
    io_type_ = IOSpec::IOType::kOutput;
  }

  void setup(OperatorSpec& spec) override;
};

/**
 * @brief Virtual Receiver operator.
 *
 * This operator represents a receiver that is connected to a transmitter in another fragment.
 * Although this operator has an output port, the output port (and its name) of this operator
 * represents the input port (and its name) of an operator that is connected to this operator in
 * the same fragment.
 */
class VirtualReceiverOp : public VirtualOperator {
 public:
  template <typename StringT, typename ArgListT,
            typename = std::enable_if_t<std::is_constructible_v<std::string, StringT> &&
                                        std::is_same_v<ArgList, std::decay_t<ArgListT>>>>
  explicit VirtualReceiverOp(StringT&& input_port, IOSpec::ConnectorType connector_type,
                             ArgListT&& arg_list)
      : VirtualOperator(std::forward<StringT>(input_port), connector_type,
                        std::forward<ArgListT>(arg_list)) {
    io_type_ = IOSpec::IOType::kInput;
  }

  void setup(OperatorSpec& spec) override;
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_CORE_SERVICES_COMMON_VIRTUAL_OPERATOR_HPP */
