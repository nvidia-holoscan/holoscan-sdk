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

#ifndef HOLOSCAN_CORE_IO_CONTEXT_HPP
#define HOLOSCAN_CORE_IO_CONTEXT_HPP

#include <any>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gxf/core/expected.hpp>

#include "./common.hpp"
#include "./gxf/entity.hpp"
#include "./message.hpp"
#include "./operator.hpp"
#include "./type_traits.hpp"

namespace holoscan {

/**
 * @brief Class to hold the input context.
 *
 * This class provides the interface to receive the input data from the operator.
 */
class InputContext {
 public:
  /**
   * @brief Construct a new InputContext object.
   *
   * @param op The pointer to the operator that this context is associated with.
   * @param inputs The references to the map of the input specs.
   */
  InputContext(Operator* op, std::unordered_map<std::string, std::unique_ptr<IOSpec>>& inputs)
      : op_(op), inputs_(inputs) {}

  /**
   * @brief Construct a new InputContext object.
   *
   * inputs for the InputContext will be set to op->spec()->inputs()
   *
   * @param op The pointer to the operator that this context is associated with.
   */
  explicit InputContext(Operator* op) : op_(op), inputs_(op->spec()->inputs()) {}

  /**
   * @brief Return the operator that this context is associated with.
   * @return The pointer to the operator.
   */
  Operator* op() const { return op_; }

  /**
   * @brief Return the reference to the map of the input specs.
   * @return The reference to the map of the input specs.
   */
  std::unordered_map<std::string, std::unique_ptr<IOSpec>>& inputs() const { return inputs_; }

  /**
   * @brief Receive a shared pointer to the message data from the input port with the given name.
   *
   * If the operator has a single input port, the name of the input port can be omitted.
   *
   * If the input port with the given name and type (`DataT`) is available,
   * it will return a shared pointer to the data (`std::shared_ptr<DataT>`) from the input port.
   * Otherwise, it will return a null shared pointer.
   *
   * Example:
   *
   * ```cpp
   * class PingRxOp : public holoscan::ops::GXFOperator {
   *  public:
   *   HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(PingRxOp, holoscan::ops::GXFOperator)
   *
   *   PingRxOp() = default;
   *
   *   void setup(OperatorSpec& spec) override {
   *     spec.input<ValueData>("in");
   *   }
   *
   *   void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
   *     // The type of `value` is `std::shared_ptr<ValueData>`
   *     auto value = op_input.receive<ValueData>("in");
   *     if (value) {
   *       HOLOSCAN_LOG_INFO("Message received (value: {})", value->data());
   *     }
   *   }
   * };
   * ```
   *
   * @tparam DataT The type of the data to receive.
   * @param name The name of the input port to receive the data from.
   * @return The shared pointer to the data.
   */
  template <typename DataT, typename = std::enable_if_t<
                                !holoscan::is_vector_v<DataT> &&
                                !holoscan::is_one_of_v<DataT, holoscan::gxf::Entity, std::any>>>
  std::shared_ptr<DataT> receive(const char* name = nullptr) {
    auto value = receive_impl(name);

    // If the received data is nullptr, return a null shared pointer.
    if (value.type() == typeid(nullptr_t)) { return nullptr; }

    try {
      return std::any_cast<std::shared_ptr<DataT>>(value);
    } catch (const std::bad_any_cast& e) {
      HOLOSCAN_LOG_ERROR(
          "Unable to cast the received data to the specified type (std::shared_ptr<"
          "DataT>): {}",
          e.what());
      return nullptr;
    }
  }

  /**
   * @brief Receive message data (GXF Entity) from the input port with the given name.
   *
   * This method is for interoperability with the GXF Codelet.
   *
   * If the operator has a single input port, the name of the input port can be omitted.
   *
   * If the input port with the given name and the type (`nvidia::gxf::Entity`) is available,
   * it will return the message data (`holoscan::gxf::Entity`) from the input
   * port. Otherwise, it will return an empty Entity or throw an exception if it fails to parse
   * the message data.
   *
   * Example:
   *
   * ```cpp
   * class PingRxOp : public holoscan::ops::GXFOperator {
   *  public:
   *   HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(PingRxOp, holoscan::ops::GXFOperator)
   *
   *   PingRxOp() = default;
   *
   *   void setup(OperatorSpec& spec) override {
   *     spec.input<holoscan::gxf::Entity>("in");
   *   }
   *
   *   void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
   *     // The type of `in_entity` is 'holoscan::gxf::Entity'.
   *     auto in_entity = op_input.receive<holoscan::gxf::Entity>("in");
   *     if (in_entity) {
   *       // Process with `in_entity`.
   *       // ...
   *     }
   *   }
   * };
   * ```
   *
   * @tparam DataT The type of the data to receive. It should be `holoscan::gxf::Entity`.
   * @param name The name of the input port to receive the data from.
   * @return The entity object (`holoscan::gxf::Entity`).
   * @throws std::runtime_error if it fails to parse the message data from the input port with the
   * given type (`DataT`).
   */
  template <typename DataT,
            typename = std::enable_if_t<holoscan::is_one_of_v<DataT, holoscan::gxf::Entity>>>
  DataT receive(const char* name) {
    auto value = receive_impl(name);

    // If the received data is nullptr, return an empty entity
    if (value.type() == typeid(nullptr_t)) { return {}; }

    try {
      return std::any_cast<holoscan::gxf::Entity>(value);
    } catch (const std::bad_any_cast& e) {
      throw std::runtime_error(
          fmt::format("Unable to cast the received data to the specified type (holoscan::gxf::"
                      "Entity): {}",
                      e.what()));
    }
  }

  /**
   * @brief Receive message data (std::any) from the input port with the given name.
   *
   * This method is for interoperability with arbitrary data types.
   *
   * If the operator has a single input port, the name of the input port can be omitted.
   *
   * If the input port with the given name is available,
   * it will return the message data (either `holoscan::gxf::Entity` or `std::shared_ptr<T>`) from
   * the input port. Otherwise, it will return a `std::any` object with the `std::nullptr`.
   *
   * Example:
   *
   * ```cpp
   * class PingRxOp : public holoscan::ops::GXFOperator {
   *  public:
   *   HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(PingRxOp, holoscan::ops::GXFOperator)
   *
   *   PingRxOp() = default;
   *
   *   void setup(OperatorSpec& spec) override {
   *     spec.input<std::any>("in");
   *   }
   *
   *   void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
   *     // The type of `in_any` is 'std::any'.
   *     auto in_any = op_input.receive<std::any>("in");
   *     auto& in_any_type = in_any.type();
   *
   *     if (in_any_type == typeid(holoscan::gxf::Entity)) {
   *       auto in_entity = std::any_cast<holoscan::gxf::Entity>(in_any);
   *       // Process with `in_entity`.
   *       // ...
   *     } else if (in_any_type == typeid(std::shared_ptr<ValueData>)) {
   *       auto in_message = std::any_cast<std::shared_ptr<ValueData>>(in_any);
   *       // Process with `in_message`.
   *       // ...
   *     } else if (in_any_type == typeid(nullptr_t)) {
   *       // No message is available.
   *     } else {
   *       HOLOSCAN_LOG_ERROR("Invalid message type: {}", in_any_type.name());
   *       return;
   *     }
   *   }
   * };
   * ```
   *
   * @tparam DataT The type of the data to receive. It should be `holoscan::gxf::Entity`.
   * @param name The name of the input port to receive the data from.
   * @return The std::any object (`holoscan::gxf::Entity`, `std::shared_ptr<T>`, or `std::nullptr`).
   */
  template <typename DataT, typename = std::enable_if_t<holoscan::is_one_of_v<DataT, std::any>>>
  std::any receive(const char* name) {
    auto value = receive_impl(name);
    return value;
  }

  /**
   * @brief Receive a vector of the shared pointers to the message data from the receivers with the
   * given name.
   *
   * If the parameter (of type `std::vector<IOSpec*>`) with the given name is available,
   * it will return a vector of the shared pointers to the data
   * (`std::vector<std::shared_ptr<DataT>>`) from the input port. Otherwise, it will return an
   * empty vector. The vector can have empty shared pointers if the data of the corresponding input
   * port is not available.
   *
   * Example:
   *
   * ```cpp
   * class ProcessTensorOp : public holoscan::ops::GXFOperator {
   *  public:
   *   HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(ProcessTensorOp, holoscan::ops::GXFOperator)
   *
   *   ProcessTensorOp() = default;
   *
   *   void setup(OperatorSpec& spec) override {
   *     spec.param(receivers_, "receivers", "Input Receivers", "List of input receivers.", {});
   *   }
   *
   *   void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override
   *   {
   *     auto in_messages = op_input.receive<std::vector<ValueData>("receivers");
   *     HOLOSCAN_LOG_INFO("Received {} messages", in_messages.size());
   *     // The type of `in_message` is 'std::vector<std::shared_ptr<ValueData>>'.
   *     auto& in_message = in_messages[0];
   *     if (in_message) {
   *       // Process with `in_message`.
   *       // ...
   *     }
   *   }
   *
   *  private:
   *   Parameter<std::vector<IOSpec*>> receivers_;
   * };
   * ```
   *
   * @tparam DataT The type of the data to receive.
   * @param name The name of the receivers whose parameter type is 'std::vector<IOSpec*>'.
   * @return The vector of the shared pointers to the data.
   */
  template <typename DataT,
            typename = std::enable_if_t<
                holoscan::is_vector_v<DataT> &&
                !holoscan::is_one_of_v<typename holoscan::type_info<DataT>::element_type,
                                       holoscan::gxf::Entity, std::any>>>
  std::vector<std::shared_ptr<typename holoscan::type_info<DataT>::element_type>> receive(
      const char* name) {
    using DataT_ElementT = typename holoscan::type_info<DataT>::element_type;

    std::vector<std::shared_ptr<DataT_ElementT>> input_vector;
    auto& params = op_->spec()->params();

    auto it = params.find(std::string(name));

    if (it == params.end()) {
      HOLOSCAN_LOG_ERROR(
          "Unable to find input parameter with name '{}'", name);
      return input_vector;
    }
    auto& param_wrapper = it->second;
    auto& arg_type = param_wrapper.arg_type();
    if ((arg_type.element_type() != ArgElementType::kIOSpec) ||
        (arg_type.container_type() != ArgContainerType::kVector)) {
      HOLOSCAN_LOG_ERROR("Input parameter with name '{}' is not of type 'std::vector<IOSpec*>'",
                         name);
      return input_vector;
    }
    std::any& any_param = param_wrapper.value();
    // Note that the type of any_param is Parameter<typeT>*, not Parameter<typeT>.
    auto& param = *std::any_cast<Parameter<std::vector<IOSpec*>>*>(any_param);

    int num_inputs = param.get().size();
    input_vector.reserve(num_inputs);

    for (int index = 0; index < num_inputs; ++index) {
      // Check if the input name points to the parameter name of the operator,
      // and the parameter type is 'std::vector<holoscan::IOSpec*>'.
      // In other words, find if there is a receiver with a specific label
      // ('<parameter name>:<index>'. e.g, 'receivers:0') to return an object with
      // 'std::vector<std::shared_ptr<DataT_ElementT>' type.
      auto value = receive_impl(fmt::format("{}:{}", name, index).c_str(), true);

      // If the received data is nullptr, add a null shared pointer.
      if (value.type() == typeid(nullptr_t)) {
        input_vector.emplace_back(nullptr);
        continue;
      }

      try {
        auto casted_value = std::any_cast<std::shared_ptr<DataT_ElementT>>(value);
        input_vector.push_back(std::move(casted_value));
      } catch (const std::bad_any_cast& e) {
        HOLOSCAN_LOG_ERROR(
            "Unable to receive input (std::vector<std::shared_ptr<DataT>>) with name "
            "'{}:{}' ({}). Skipping adding to the vector.",
            name,
            index,
            e.what());
      }
    }
    return input_vector;
  }

  /**
   * @brief Receive a vector of entities from the receivers with the given name.
   *
   * This method is for interoperability with the GXF Codelet and is only available when the
   * `DataT` is `std::vector<holoscan::gxf::Entity>`.
   *
   * If the parameter (of type `std::vector<IOSpec*>`) with
   * the given name is available, it will return a vector of entities
   * (`std::vector<holoscan::gxf::Entity>`). Otherwise, it will return an
   * empty vector. The vector can have empty entities if the data of the corresponding
   * input port is not available.
   *
   * Example:
   *
   * ```cpp
   * class ProcessTensorOp : public holoscan::ops::GXFOperator {
   *  public:
   *   HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(ProcessTensorOp, holoscan::ops::GXFOperator)
   *
   *   ProcessTensorOp() = default;
   *
   *   void setup(OperatorSpec& spec) override {
   *     spec.param(receivers_, "receivers", "Input Receivers", "List of input receivers.", {});
   *   }
   *
   *   void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override
   *   {
   *     auto in_messages = op_input.receive<std::vector<holoscan::gxf::Entity>>("receivers");
   *     HOLOSCAN_LOG_INFO("Received {} messages", in_messages.size());
   *     // in_message's type is 'holoscan::gxf::Entity'.
   *     auto& in_message = in_messages[0];
   *     if (in_message)
   *     {
   *       // Process with 'in_message' here.
   *     }
   *   }
   *
   *  private:
   *   Parameter<std::vector<IOSpec*>> receivers_;
   * };
   * ```
   *
   * @tparam DataT The type of the data to receive.
   * @param name The name of the receivers whose parameter type is 'std::vector<IOSpec*>'.
   * @return The vector of entities (`std::vector<holoscan::gxf::Entity>`).
   */
  template <typename DataT,
            typename = std::enable_if_t<
                holoscan::is_vector_v<DataT> &&
                holoscan::is_one_of_v<typename holoscan::type_info<DataT>::element_type,
                                      holoscan::gxf::Entity>>>
  std::vector<holoscan::gxf::Entity> receive(const char* name) {
    std::vector<holoscan::gxf::Entity> input_vector;
    auto& params = op_->spec()->params();

    auto it = params.find(std::string(name));

    if (it == params.end()) {
      HOLOSCAN_LOG_ERROR(
          "Unable to find input parameter with name '{}'", name);
      return input_vector;
    }
    auto& param_wrapper = it->second;
    auto& arg_type = param_wrapper.arg_type();
    if ((arg_type.element_type() != ArgElementType::kIOSpec) ||
        (arg_type.container_type() != ArgContainerType::kVector)) {
      HOLOSCAN_LOG_ERROR("Input parameter with name '{}' is not of type 'std::vector<IOSpec*>'",
                         name);
      return input_vector;
    }
    std::any& any_param = param_wrapper.value();
    // Note that the type of any_param is Parameter<typeT>*, not Parameter<typeT>.
    auto& param = *std::any_cast<Parameter<std::vector<IOSpec*>>*>(any_param);

    int num_inputs = param.get().size();
    input_vector.reserve(num_inputs);

    for (int index = 0; index < num_inputs; ++index) {
      // Check if the input name points to the parameter name of the operator,
      // and the parameter type is 'std::vector<holoscan::IOSpec*>'.
      // In other words, find if there is a receiver with a specific label
      // ('<parameter name>:<index>'. e.g, 'receivers:0') to return an object with
      // 'std::vector<holoscan::gxf::Entity>' type.
      auto value = receive_impl(fmt::format("{}:{}", name, index).c_str(), true);

      // If the received data is nullptr, add an empty entity.
      if (value.type() == typeid(nullptr_t)) {
        input_vector.emplace_back();
        continue;
      }

      try {
        auto casted_value = std::any_cast<holoscan::gxf::Entity>(value);
        input_vector.push_back(std::move(casted_value));
      } catch (const std::bad_any_cast& e) {
        HOLOSCAN_LOG_ERROR(
            "Unable to receive input (std::vector<holoscan::gxf::Entity>) "
            "with name "
            "'{}:{}' ({}). Skipping adding to the vector.",
            name,
            index,
            e.what());
      }
    }
    return input_vector;
  }

  /**
   * @brief Receive a vector of entities from the receivers with the given name.
   *
   * If the parameter (of type `std::vector<IOSpec*>`) with
   * the given name is available, it will return a vector of entities
   * (`std::vector<std::any>`) from the receivers. Otherwise, it will return an
   * empty vector.
   *
   * Example:
   *
   * ```cpp
   * class ProcessTensorOp : public holoscan::ops::GXFOperator {
   *  public:
   *   HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(ProcessTensorOp, holoscan::ops::GXFOperator)
   *
   *   ProcessTensorOp() = default;
   *
   *   void setup(OperatorSpec& spec) override {
   *     spec.param(receivers_, "receivers", "Input Receivers", "List of input receivers.", {});
   *   }
   *
   *   void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override
   *   {
   *     auto in_any_vector = op_input.receive<std::vector<std::any>>("receivers");
   *     HOLOSCAN_LOG_INFO("Received {} messages", in_any_vector.size());
   *
   *     if (in_any_vector.empty()) {
   *         HOLOSCAN_LOG_ERROR("No input messages received.");
   *         return;
   *     }
   *
   *     // in_any's type is 'std::any'.
   *     auto& in_any = in_any_vector[0];
   *     auto& in_any_type = in_any.type();
   *
   *     if (in_any_type == typeid(holoscan::gxf::Entity)) {
   *       auto in_entity = std::any_cast<holoscan::gxf::Entity>(in_any);
   *       // Process with `in_entity`.
   *       // ...
   *     } else if (in_any_type == typeid(std::shared_ptr<ValueData>)) {
   *       auto in_message = std::any_cast<std::shared_ptr<ValueData>>(in_any);
   *       // Process with `in_message`.
   *       // ...
   *     } else if (in_any_type == typeid(nullptr_t)) {
   *       // No message is available.
   *     } else {
   *       HOLOSCAN_LOG_ERROR("Invalid message type: {}", in_any_type.name());
   *       return;
   *     }
   *   }
   *
   *  private:
   *   Parameter<std::vector<IOSpec*>> receivers_;
   * };
   * ```
   *
   * @tparam DataT The type of the data to receive.
   * @param name The name of the receivers whose parameter type is 'std::vector<IOSpec*>'.
   * @return The vector of entities (`std::vector<std::any>`).
   */
  template <typename DataT,
            typename = std::enable_if_t<
                holoscan::is_vector_v<DataT> &&
                holoscan::is_one_of_v<typename holoscan::type_info<DataT>::element_type, std::any>>>
  std::vector<std::any> receive(const char* name) {
    std::vector<std::any> input_vector;
    auto& params = op_->spec()->params();

    auto it = params.find(std::string(name));

    if (it == params.end()) {
      HOLOSCAN_LOG_ERROR(
          "Unable to find input parameter with name '{}'", name);
      return input_vector;
    }
    auto& param_wrapper = it->second;
    auto& arg_type = param_wrapper.arg_type();
    if ((arg_type.element_type() != ArgElementType::kIOSpec) ||
        (arg_type.container_type() != ArgContainerType::kVector)) {
      HOLOSCAN_LOG_ERROR("Input parameter with name '{}' is not of type 'std::vector<IOSpec*>'",
                         name);
      return input_vector;
    }
    std::any& any_param = param_wrapper.value();
    // Note that the type of any_param is Parameter<typeT>*, not Parameter<typeT>.
    auto& param = *std::any_cast<Parameter<std::vector<IOSpec*>>*>(any_param);

    int num_inputs = param.get().size();
    input_vector.reserve(num_inputs);

    for (int index = 0; index < num_inputs; ++index) {
      // Check if the input name points to the parameter name of the operator,
      // and the parameter type is 'std::vector<holoscan::IOSpec*>'.
      // In other words, find if there is a receiver with a specific label
      // ('<parameter name>:<index>'. e.g, 'receivers:0') to return an object with
      // 'std::vector<std::any>' type.
      auto value = receive_impl(fmt::format("{}:{}", name, index).c_str(), true);
      input_vector.push_back(std::move(value));
    }
    return input_vector;
  }

 protected:
  /**
   * @brief The implementation of the `receive` method.
   *
   * Depending on the type of the data, this method receives a message from the input port
   * with the given name.
   *
   * @param name The name of the input port.
   * @param no_error_message Whether to print an error message when the input port is not
   * found.
   * @return The data received from the input port.
   */
  virtual std::any receive_impl(const char* name = nullptr, bool no_error_message = false) {
    (void)name;
    (void)no_error_message;
    return nullptr;
  }

  Operator* op_ = nullptr;  ///< The operator that this context is associated with.
  std::unordered_map<std::string, std::unique_ptr<IOSpec>>& inputs_;  ///< The inputs.
};

/**
 * @brief Class to hold the output context.
 *
 * This class provides the interface to send data to the output ports of the operator.
 */
class OutputContext {
 public:
  /**
   * @brief Construct a new OutputContext object.
   *
   * @param op The pointer to the operator that this context is associated with.
   * @param outputs The references to the map of the output specs.
   */
  OutputContext(Operator* op, std::unordered_map<std::string, std::unique_ptr<IOSpec>>& outputs)
      : op_(op), outputs_(outputs) {}

  /**
   * @brief Return the operator that this context is associated with.
   * @return The pointer to the operator.
   */
  Operator* op() const { return op_; }

  /**
   * @brief Return the reference to the map of the output specs.
   * @return The reference to the map of the output specs.
   */
  std::unordered_map<std::string, std::unique_ptr<IOSpec>>& outputs() const { return outputs_; }

  /**
   * @brief Construct a new OutputContext object.
   *
   * outputs for the OutputContext will be set to op->spec()->outputs()
   *
   * @param op The pointer to the operator that this context is associated with.
   */
  explicit OutputContext(Operator* op) : op_(op), outputs_(op->spec()->outputs()) {}

  /**
   * @brief The output data type.
   */
  enum class OutputType {
    kSharedPointer,  ///< The message data to send is a shared pointer.
    kGXFEntity,      ///< The message data to send is a GXF entity.
  };

  /**
   * @brief Send a shared pointer of the message data to the output port with the given name.
   *
   * The object to be sent must be a shared pointer of the message data and the output port with
   * the given name must exist.
   *
   * If the operator has a single output port, the output port name can be omitted.
   *
   * Example:
   *
   * ```cpp
   * class PingTxOp : public holoscan::ops::GXFOperator {
   *  public:
   *   HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(PingTxOp, holoscan::ops::GXFOperator)
   *
   *   PingTxOp() = default;
   *
   *   void setup(OperatorSpec& spec) override {
   *     spec.output<ValueData>("out");
   *   }
   *
   *   void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
   *     auto value = std::make_shared<ValueData>(7);
   *     op_output.emit(value, "out");
   *   }
   * };
   * ```
   *
   * @tparam DataT The type of the data to send.
   * @param data The shared pointer to the data.
   * @param name The name of the output port.
   */
  template <typename DataT>
  void emit(std::shared_ptr<DataT>& data, const char* name = nullptr) {
    emit_impl(data, name);
  }

  /**
   * @brief Send message data (GXF Entity) to the output port with the given name.
   *
   * This method is for interoperability with the GXF Codelet.
   *
   * The object to be sent must be an object with `holoscan::gxf::Entity` type and the output port
   * with the given name must exist.
   *
   * If the operator has a single output port, the output port name can be omitted.
   *
   * Example:
   *
   * ```cpp
   * class PingTxOp : public holoscan::ops::GXFOperator {
   *  public:
   *   HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(PingTxOp, holoscan::ops::GXFOperator)
   *
   *   PingTxOp() = default;
   *
   *   void setup(OperatorSpec& spec) override {
   *     spec.input<holoscan::gxf::Entity>("in");
   *     spec.output<holoscan::gxf::Entity>("out");
   *   }
   *
   *   void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override
   *   {
   *     // The type of `in_message` is 'holoscan::gxf::Entity'.
   *     auto in_message = op_input.receive<holoscan::gxf::Entity>("in");
   *     // The type of `tensor` is 'std::shared_ptr<holoscan::Tensor>'.
   *     auto tensor = in_message.get<Tensor>();
   *
   *     // Process with 'tensor' here.
   *     // ...
   *
   *     // Create a new message (Entity)
   *     auto out_message = holoscan::gxf::Entity::New(&context);
   *     out_message.add(tensor, "tensor");
   *
   *     // Send the processed message.
   *     op_output.emit(out_message, "out");
   *   }
   * };
   * ```
   *
   * @tparam DataT The type of the data to send. It should be `holoscan::gxf::Entity`.
   * @param data The entity object to send (`holoscan::gxf::Entity`).
   * @param name The name of the output port.
   */
  template <typename DataT,
            typename = std::enable_if_t<std::is_same_v<holoscan::gxf::Entity, std::decay_t<DataT>>>>
  void emit(DataT& data, const char* name = nullptr) {
    emit_impl(data, name, OutputType::kGXFEntity);
  }

 protected:
  /**
   * @brief The implementation of the `emit` method.
   *
   * Depending on the type of the data, this method wraps the data with a message and sends it to
   * the output port with the given name.
   *
   * @param data The data to send.
   * @param name The name of the output port.
   * @param out_type The type of the message data.
   */
  virtual void emit_impl(std::any data, const char* name = nullptr,
                         OutputType out_type = OutputType::kSharedPointer) {
    (void)data;
    (void)name;
    (void)out_type;
  }

  Operator* op_ = nullptr;  ///< The operator that this context is associated with.
  std::unordered_map<std::string, std::unique_ptr<IOSpec>>& outputs_;  ///< The outputs.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_IO_CONTEXT_HPP */
