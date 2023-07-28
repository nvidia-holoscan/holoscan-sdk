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

#ifndef HOLOSCAN_CORE_IO_CONTEXT_HPP
#define HOLOSCAN_CORE_IO_CONTEXT_HPP

#include <any>
#include <map>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "./common.hpp"
#include "./domain/tensor_map.hpp"
#include "./errors.hpp"
#include "./expected.hpp"
#include "./gxf/entity.hpp"
#include "./message.hpp"
#include "./operator.hpp"
#include "./type_traits.hpp"

namespace holoscan {

static inline std::string get_well_formed_name(
    const char* name, const std::unordered_map<std::string, std::unique_ptr<IOSpec>>& io_list) {
  std::string well_formed_name;
  if (name == nullptr || name[0] == '\0') {
    if (io_list.size() == 1) {
      well_formed_name = io_list.begin()->first;
    } else {
      well_formed_name = "";
    }
  } else {
    well_formed_name = name;
  }
  return well_formed_name;
}

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
   * @param execution_context The pointer to the execution context.
   * @param op The pointer to the operator that this context is associated with.
   * @param inputs The references to the map of the input specs.
   */
  InputContext(ExecutionContext* execution_context, Operator* op,
               std::unordered_map<std::string, std::unique_ptr<IOSpec>>& inputs)
      : execution_context_(execution_context), op_(op), inputs_(inputs) {}

  /**
   * @brief Construct a new InputContext object.
   *
   * inputs for the InputContext will be set to op->spec()->inputs()
   *
   * @param execution_context The pointer to GXF execution runtime
   * @param op The pointer to the operator that this context is associated with.
   */
  InputContext(ExecutionContext* execution_context, Operator* op)
      : execution_context_(execution_context), op_(op), inputs_(op->spec()->inputs()) {}

  /**
   * @brief Get pointer to the execution context.
   *
   * @return The pointer to the execution context.
   */
  ExecutionContext* execution_context() const { return execution_context_; }

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
   * @brief Return whether the input port has any data.
   *
   * For parameters with std::vector<IOSpec*> type, if all the inputs are empty, it will return
   * true. Otherwise, it will return false.
   *
   * @param name The name of the input port to check.
   * @return True, if it has no data, otherwise false.
   */
  bool empty(const char* name = nullptr) {
    // First see if the name could be found in the inputs
    auto& inputs = op_->spec()->inputs();
    auto it = inputs.find(std::string(name));
    if (it != inputs.end()) { return empty_impl(name); }

    // Then see if it is in the parameters
    auto& params = op_->spec()->params();
    auto it2 = params.find(std::string(name));
    if (it2 != params.end()) {
      auto& param_wrapper = it2->second;
      auto& arg_type = param_wrapper.arg_type();
      if ((arg_type.element_type() != ArgElementType::kIOSpec) ||
          (arg_type.container_type() != ArgContainerType::kVector)) {
        HOLOSCAN_LOG_ERROR("Input parameter with name '{}' is not of type 'std::vector<IOSpec*>'",
                           name);
        return true;
      }
      std::any& any_param = param_wrapper.value();
      // Note that the type of any_param is Parameter<typeT>*, not Parameter<typeT>.
      auto& param = *std::any_cast<Parameter<std::vector<IOSpec*>>*>(any_param);
      int num_inputs = param.get().size();
      for (int i = 0; i < num_inputs; ++i) {
        // if any of them is not empty return false
        if (!empty_impl(fmt::format("{}:{}", name, i).c_str())) { return false; }
      }
      return true;  // all of them are empty, so return true.
    }

    HOLOSCAN_LOG_ERROR("Input port '{}' not found", name);
    return true;
  }

  /**
   * @brief Receive a message from the input port with the given name.
   *
   * If the operator has a single input port, the name of the input port can be omitted.
   *
   * If the input port with the given name and type (`DataT`) is available, it will return the data
   * from the input port. Otherwise, it will return an object of the holoscan::unexpected class
   * which will contain the error message. The error message can be access by calling the `what()`
   * method of the holoscan::unexpected object.
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
   *     spec.input<std::shared_ptr<ValueData>>("in");
   *   }
   *
   *   void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
   *     auto value = op_input.receive<std::shared_ptr<ValueData>>("in");
   *     if (value.has_value()) {
   *       HOLOSCAN_LOG_INFO("Message received (value: {})", value->data());
   *     }
   *   }
   * };
   * ```
   *
   * @tparam DataT The type of the data to receive.
   * @param name The name of the input port to receive the data from.
   * @return The received data.
   */
  template <typename DataT>
  holoscan::expected<DataT, holoscan::RuntimeError> receive(const char* name = nullptr) {
    if constexpr (holoscan::is_vector_v<DataT>) {
      // It could either be a parameter which is trying to receive from a vector
      // or a vector of values from the inputs
      // First check, if it is trying to receive from a parameter

      auto& params = op_->spec()->params();
      auto it = params.find(std::string(name));

      if (it == params.end()) {
        // the name is not a parameter, so it must be an input
        auto& inputs = op_->spec()->inputs();
        if (inputs.find(std::string(name)) == inputs.end()) {
          auto error_message =
              fmt::format("Unable to find input parameter or input port with name '{}'", name);
          // Keep the debugging info on for development purposes
          HOLOSCAN_LOG_DEBUG(error_message);
          return make_unexpected<holoscan::RuntimeError>(
              holoscan::RuntimeError(holoscan::ErrorCode::kReceiveError, error_message.c_str()));
        }

        auto value = receive_impl(name);
        if (value.type() == typeid(nullptr_t)) {
          auto error_message =
              fmt::format("No data is received from the input port with name '{}'", name);
          HOLOSCAN_LOG_DEBUG(error_message);
          return make_unexpected<holoscan::RuntimeError>(
              holoscan::RuntimeError(holoscan::ErrorCode::kReceiveError, error_message.c_str()));
        }
        try {
          DataT result = std::any_cast<DataT>(value);
          return result;
        } catch (const std::bad_any_cast& e) {
          auto error_message = fmt::format(
              "Unable to cast input (DataT of type std::vector) with input name '{}' ({}).",
              name,
              e.what());
          HOLOSCAN_LOG_DEBUG(error_message);
          return make_unexpected<holoscan::RuntimeError>(
              holoscan::RuntimeError(holoscan::ErrorCode::kReceiveError, error_message.c_str()));
        }
      }

      auto& param_wrapper = it->second;
      auto& arg_type = param_wrapper.arg_type();
      if ((arg_type.element_type() != ArgElementType::kIOSpec) ||
          (arg_type.container_type() != ArgContainerType::kVector)) {
        auto error_message = fmt::format(
            "Input parameter with name '{}' is not of type 'std::vector<IOSpec*>'", name);
        HOLOSCAN_LOG_ERROR(error_message);
        return make_unexpected<holoscan::RuntimeError>(
            holoscan::RuntimeError(holoscan::ErrorCode::kReceiveError, error_message.c_str()));
      }
      std::any& any_param = param_wrapper.value();
      // Note that the type of any_param is Parameter<typeT>*, not Parameter<typeT>.
      auto& param = *std::any_cast<Parameter<std::vector<IOSpec*>>*>(any_param);

      std::vector<typename DataT::value_type> input_vector;
      int num_inputs = param.get().size();
      input_vector.reserve(num_inputs);

      for (int index = 0; index < num_inputs; ++index) {
        // Check if the input name points to the parameter name of the operator,
        // and the parameter type is 'std::vector<holoscan::IOSpec*>'.
        // In other words, find if there is a receiver with a specific label
        // ('<parameter name>:<index>'. e.g, 'receivers:0') to return an object with
        // 'std::vector<std::shared_ptr<DataT_ElementT>' type.
        auto value = receive_impl(fmt::format("{}:{}", name, index).c_str(), true);

        try {
          // If the received data is nullptr, any_cast will try to cast to appropriate pointer
          // type. Otherwise it will register an error.
          if constexpr (std::is_same_v<typename DataT::value_type, std::any>) {
            input_vector.push_back(std::move(value));
          } else {
            auto casted_value = std::move(std::any_cast<typename DataT::value_type>(value));
            input_vector.push_back(std::move(casted_value));
          }
        } catch (const std::bad_any_cast& e) {
          auto error_message =
              fmt::format("Unable to cast input (DataT::value_type) with name '{}:{}' ({}).",
                          name,
                          index,
                          e.what());
          try {
            // An empty holoscan::gxf::Entity will be added to the vector.
            typename DataT::value_type placeholder;
            input_vector.push_back(std::move(placeholder));
            error_message =
                fmt::format("{}\tA placeholder value is added to the vector for input '{}:{}'.",
                            error_message,
                            name,
                            index);
            HOLOSCAN_LOG_WARN(error_message);
          } catch (std::exception& e) {
            error_message = fmt::format(
                "{}\tUnable to add a placeholder value to the vector for input '{}:{}' :{}. "
                "Skipping adding a value to the vector.",
                error_message,
                name,
                index,
                e.what());
            HOLOSCAN_LOG_ERROR(error_message);
            continue;
          }
        }
      }
      return std::any_cast<DataT>(input_vector);
    } else {
      // If it is not a vector then try to get the input directly and convert for respective data
      // type for an input
      auto value = receive_impl(name);
      // If the received data is nullptr, then check whether nullptr or empty holoscan::gxf::Entity
      // can be sent
      if (value.type() == typeid(nullptr_t)) {
        HOLOSCAN_LOG_DEBUG("nullptr is received from the input port with name '{}'", name);
        // If it is a shared pointer, or raw pointer then return nullptr because it might be a valid
        // nullptr
        if constexpr (holoscan::is_shared_ptr_v<DataT>) {
          return nullptr;
        } else if constexpr (std::is_pointer_v<DataT>) {
          return nullptr;
        }
        // If it's holoscan::gxf::Entity then return an error message
        if constexpr (is_one_of_derived_v<DataT, nvidia::gxf::Entity>) {
          auto error_message = fmt::format(
              "Null received in place of nvidia::gxf::Entity or derived type for input {}", name);
          return make_unexpected<holoscan::RuntimeError>(
              holoscan::RuntimeError(holoscan::ErrorCode::kReceiveError, error_message.c_str()));
        } else if constexpr (is_one_of_derived_v<DataT, holoscan::TensorMap>) {
          auto error_message = fmt::format(
              "Null received in place of holoscan::TensorMap or derived type for input {}", name);
          return make_unexpected<holoscan::RuntimeError>(
              holoscan::RuntimeError(holoscan::ErrorCode::kReceiveError, error_message.c_str()));
        }
      }

      try {
        // Check if the types of value and DataT are the same or not
        if constexpr (std::is_same_v<DataT, std::any>) { return value; }
        DataT return_value = std::any_cast<DataT>(value);
        return return_value;
      } catch (const std::bad_any_cast& e) {
        // If it is of the type of holoscan::gxf::Entity then show a specific error message
        if constexpr (is_one_of_derived_v<DataT, nvidia::gxf::Entity>) {
          auto error_message = fmt::format(
              "Unable to cast the received data to the specified type (holoscan::gxf::"
              "Entity) for input {}: {}",
              name,
              e.what());
          HOLOSCAN_LOG_DEBUG(error_message);
          return make_unexpected<holoscan::RuntimeError>(
              holoscan::RuntimeError(holoscan::ErrorCode::kReceiveError, error_message.c_str()));
        } else if constexpr (is_one_of_derived_v<DataT, holoscan::TensorMap>) {
          TensorMap tensor_map;

          try {
            auto gxf_entity = std::any_cast<holoscan::gxf::Entity>(value);

            auto components_expected = gxf_entity.findAll();
            auto components = components_expected.value();
            for (size_t i = 0; i < components.size(); i++) {
              const auto component = components[i];
              const auto component_name = component->name();

              std::shared_ptr<holoscan::Tensor> holoscan_tensor =
                  gxf_entity.get<holoscan::Tensor>(component_name);
              if (holoscan_tensor) { tensor_map.insert({component_name, holoscan_tensor}); }
            }
          } catch (const std::bad_any_cast& e) {
            auto error_message = fmt::format(
                "Unable to cast the received data to the specified type (holoscan::TensorMap) for "
                "input {}: {}",
                name,
                e.what());
            HOLOSCAN_LOG_DEBUG(error_message);
            return make_unexpected<holoscan::RuntimeError>(
                holoscan::RuntimeError(holoscan::ErrorCode::kReceiveError, error_message.c_str()));
          }
          return tensor_map;
        }
        auto error_message = fmt::format(
            "Unable to cast the received data to the specified type (DataT) for input {}: {}",
            name,
            e.what());
        HOLOSCAN_LOG_DEBUG(error_message);
        return make_unexpected<holoscan::RuntimeError>(
            holoscan::RuntimeError(holoscan::ErrorCode::kReceiveError, error_message.c_str()));
      }
    }
  }

 protected:
  /**
   * @brief The implementation of the `empty` method.
   *
   * @param name The name of the input port
   * @return True if the input port is empty or by default. Otherwise, false.
   */
  virtual bool empty_impl(const char* name = nullptr) {
    (void)name;
    return true;
  }
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

  ExecutionContext* execution_context_ =
      nullptr;              ///< The execution context that is associated with.
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
   * outputs for the OutputContext will be set to op->spec()->outputs()
   *
   * @param execution_context The pointer to the execution context.
   * @param op The pointer to the operator that this context is associated with.
   */
  OutputContext(ExecutionContext* execution_context, Operator* op)
      : execution_context_(execution_context), op_(op), outputs_(op->spec()->outputs()) {}

  /**
   * @brief Construct a new OutputContext object.
   *
   * @param execution_context The pointer to the execution context.
   * @param op The pointer to the operator that this context is associated with.
   * @param outputs The references to the map of the output specs.
   */
  OutputContext(ExecutionContext* execution_context, Operator* op,
                std::unordered_map<std::string, std::unique_ptr<IOSpec>>& outputs)
      : execution_context_(execution_context), op_(op), outputs_(outputs) {}

  /**
   * @brief Get pointer to the execution context.
   *
   * @return The pointer to the execution context.
   */
  ExecutionContext* execution_context() const { return execution_context_; }

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
   * @brief The output data type.
   */
  enum class OutputType {
    kSharedPointer,  ///< The message data to send is a shared pointer.
    kGXFEntity,      ///< The message data to send is a GXF entity.
    kAny,            ///< The message data to send is a std::any.
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
  template <typename DataT, typename = std::enable_if_t<!holoscan::is_one_of_derived_v<
                                DataT, nvidia::gxf::Entity, std::any>>>
  void emit(std::shared_ptr<DataT>& data, const char* name = nullptr) {
    emit_impl(data, name);
  }

  /**
   * @brief Send message data (GXF Entity) to the output port with the given name.
   *
   * This method is for interoperability with the GXF Codelet.
   *
   * The object to be sent must be an object with `holoscan::gxf::Entity` type and the output
   * port with the given name must exist.
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
   *   void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&)
   * override
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
            typename = std::enable_if_t<holoscan::is_one_of_derived_v<DataT, nvidia::gxf::Entity>>>
  void emit(DataT& data, const char* name = nullptr) {
    // if it is the same as nvidia::gxf::Entity then just pass it to emit_impl
    if constexpr (holoscan::is_one_of_v<DataT, nvidia::gxf::Entity>) {
      emit_impl(data, name, OutputType::kGXFEntity);
    } else {
      // Convert it to nvidia::gxf::Entity and then pass it to emit_impl
      // Otherwise, we will lose the type information and cannot cast appropriately in emit_impl
      emit_impl(nvidia::gxf::Entity(data), name, OutputType::kGXFEntity);
    }
  }

  /**
   * @brief Send the message data (std::any) to the output port with the given name.
   *
   * This method is for interoperability with arbitrary data types.
   *
   * The object to be sent can be any type except the shared pointer (std::shared_ptr<T>) or the
   * GXF Entity (holoscan::gxf::Entity) type, and the output port with the given name must exist.
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
   * @tparam DataT The type of the data to send. It can be any type except the shared pointer
   * (std::shared_ptr<T>) or the GXF Entity (holoscan::gxf::Entity) type.
   * @param data The entity object to send (as `std::any`).
   * @param name The name of the output port.
   */
  template <typename DataT,
            typename = std::enable_if_t<!holoscan::is_one_of_derived_v<DataT, nvidia::gxf::Entity>>>
  void emit(DataT data, const char* name = nullptr) {
    emit_impl(data, name, OutputType::kAny);
  }

  void emit(holoscan::TensorMap& data, const char* name = nullptr) {
    auto out_message = holoscan::gxf::Entity::New(execution_context_);
    for (auto& [key, tensor] : data) { out_message.add(tensor, key.c_str()); }
    emit(out_message, name);
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

  ExecutionContext* execution_context_ =
      nullptr;              ///< The execution context that is associated with.
  Operator* op_ = nullptr;  ///< The operator that this context is associated with.
  std::unordered_map<std::string, std::unique_ptr<IOSpec>>& outputs_;  ///< The outputs.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_IO_CONTEXT_HPP */
