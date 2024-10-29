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

#ifndef HOLOSCAN_CORE_IO_CONTEXT_HPP
#define HOLOSCAN_CORE_IO_CONTEXT_HPP

#include <any>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include <common/type_name.hpp>
#include "./common.hpp"
#include "./domain/tensor_map.hpp"
#include "./errors.hpp"
#include "./expected.hpp"
#include "./gxf/entity.hpp"
#include "./message.hpp"
#include "./operator.hpp"
#include "./type_traits.hpp"

namespace holoscan {

// To indicate that data is not available for the input port
struct NoMessageType {};
constexpr NoMessageType kNoReceivedMessage;

// To indicate that input port is not accessible
struct NoAccessibleMessageType : public std::string {
  NoAccessibleMessageType() : std::string("Port is not accessible") {}
  explicit NoAccessibleMessageType(const std::string& message) : std::string(message) {}
  explicit NoAccessibleMessageType(const char* message) : std::string(message) {}
  explicit NoAccessibleMessageType(std::string&& message) : std::string(std::move(message)) {}
};

static inline std::string get_well_formed_name(
    const char* name, const std::unordered_map<std::string, std::shared_ptr<IOSpec>>& io_list) {
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
               std::unordered_map<std::string, std::shared_ptr<IOSpec>>& inputs)
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
  std::unordered_map<std::string, std::shared_ptr<IOSpec>>& inputs() const { return inputs_; }

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
   * It throws an invalid argument exception if the operator attempts to receive non-vector data
   * (`op_input.receive<T>()`) from an input port with a queue size of `IOSpec::kAnySize`.
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
   *   void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
   *                [[maybe_unused]] ExecutionContext& context) override {
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
    auto& params = op_->spec()->params();
    auto param_it = params.find(std::string(name));

    if constexpr (holoscan::is_vector_v<DataT>) {
      DataT input_vector;
      std::string error_message;

      if (param_it != params.end()) {
        auto& param_wrapper = param_it->second;
        if (!is_valid_param_type(param_wrapper.arg_type())) {
          return make_unexpected<holoscan::RuntimeError>(
              create_receive_error(name, "Input parameter is not of type 'std::vector<IOSpec*>'"));
        }
        if (!fill_input_vector_from_params(param_wrapper, name, input_vector, error_message)) {
          return make_unexpected<holoscan::RuntimeError>(
              create_receive_error(name, error_message.c_str()));
        }
      } else {
        if (!fill_input_vector_from_inputs(name, input_vector, error_message)) {
          return make_unexpected<holoscan::RuntimeError>(
              create_receive_error(name, error_message.c_str()));
        }
      }
      return input_vector;
    } else {
      return receive_single_value<DataT>(name);
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

  // --------------- Start of helper functions for the receive method ---------------
  inline bool is_valid_param_type(const ArgType& arg_type) {
    return (arg_type.element_type() == ArgElementType::kIOSpec) &&
           (arg_type.container_type() == ArgContainerType::kVector);
  }

  template <typename DataT>
  inline bool fill_input_vector_from_params(ParameterWrapper& param_wrapper, const char* name,
                                            DataT& input_vector, std::string& error_message) {
    auto& param = *std::any_cast<Parameter<std::vector<IOSpec*>>*>(param_wrapper.value());
    int num_inputs = param.get().size();
    input_vector.reserve(num_inputs);

    for (int index = 0; index < num_inputs; ++index) {
      std::string port_name = fmt::format("{}:{}", name, index);
      auto value = receive_impl(port_name.c_str(), true);
      const std::type_info& value_type = value.type();

      if (value_type == typeid(kNoReceivedMessage)) {
        error_message =
            fmt::format("No data is received from the input port with name '{}'", port_name);
        return false;
      }

      if (!process_received_value(value, value_type, name, index, input_vector, error_message)) {
        return false;
      }
    }
    return true;
  }

  template <typename DataT>
  inline bool fill_input_vector_from_inputs(const char* name, DataT& input_vector,
                                            std::string& error_message) {
    const auto& inputs = op_->spec()->inputs();
    const auto input_it = inputs.find(std::string(name));

    if (input_it == inputs.end()) { return false; }

    int index = 0;
    while (true) {
      auto value = receive_impl(name);
      const std::type_info& value_type = value.type();

      if (value_type == typeid(kNoReceivedMessage)) {
        if (index == 0) {
          error_message =
              fmt::format("No data is received from the input port with name '{}'", name);
          return false;
        }
        break;
      }
      if (index == 0 && value_type == typeid(DataT)) {
        // If the first input is of type DataT (such as `std::vector<bool>`), then return the value
        // directly
        input_vector = std::move(std::any_cast<DataT>(value));
        return true;
      }

      if (!process_received_value(value, value_type, name, index++, input_vector, error_message)) {
        return false;
      }
    }
    return true;
  }

  inline bool populate_tensor_map(const holoscan::gxf::Entity& gxf_entity,
                                  holoscan::TensorMap& tensor_map) {
    auto tensor_components_expected = gxf_entity.findAllHeap<nvidia::gxf::Tensor>();
    for (const auto& gxf_tensor : tensor_components_expected.value()) {
      // Do zero-copy conversion to holoscan::Tensor (as in gxf_entity.get<holoscan::Tensor>())
      auto maybe_dl_ctx = (*gxf_tensor->get()).toDLManagedTensorContext();
      if (!maybe_dl_ctx) {
        HOLOSCAN_LOG_ERROR(
            "Failed to get std::shared_ptr<DLManagedTensorContext> from nvidia::gxf::Tensor");
        return false;
      }
      auto holoscan_tensor = std::make_shared<Tensor>(maybe_dl_ctx.value());
      tensor_map.insert({gxf_tensor->name(), holoscan_tensor});
    }
    return true;
  }

  template <typename DataT>
  inline bool process_received_value(std::any& value, const std::type_info& value_type,
                                     const char* name, int index, DataT& input_vector,
                                     std::string& error_message) {
    bool is_bad_any_cast = false;

    // Assume that the received data is not of type NoMessageType
    // (this case should be handled by the caller)

    if (value_type == typeid(NoAccessibleMessageType)) {
      auto casted_value = std::any_cast<NoAccessibleMessageType>(value);
      HOLOSCAN_LOG_ERROR(static_cast<std::string>(casted_value));
      error_message = std::move(static_cast<std::string>(casted_value));
      return false;
    }

    if constexpr (std::is_same_v<typename DataT::value_type, std::any>) {
      input_vector.push_back(std::move(value));
    } else if (value_type == typeid(std::nullptr_t)) {
      handle_null_value<DataT>(input_vector);
    } else {
      try {
        auto casted_value = std::any_cast<typename DataT::value_type>(value);
        input_vector.push_back(casted_value);
      } catch (const std::bad_any_cast& e) {
        is_bad_any_cast = true;
      } catch (const std::exception& e) {
        error_message = fmt::format(
            "Unable to cast the received data to the specified type for input '{}:{}' of "
            "type {}: {}",
            name,
            index,
            value_type.name(),
            e.what());
        return false;
      }
    }

    if (is_bad_any_cast) {
      return handle_bad_any_cast<DataT>(value, name, index, input_vector, error_message);
    }

    return true;
  }

  template <typename DataT>
  inline void handle_null_value(DataT& input_vector) {
    if constexpr (holoscan::is_shared_ptr_v<typename DataT::value_type> ||
                  std::is_pointer_v<typename DataT::value_type>) {
      input_vector.push_back(typename DataT::value_type{nullptr});
    }
  }

  template <typename DataT>
  inline bool handle_bad_any_cast(std::any& value, const char* name, int index, DataT& input_vector,
                                  std::string& error_message) {
    if constexpr (is_one_of_derived_v<typename DataT::value_type, nvidia::gxf::Entity>) {
      error_message = fmt::format(
          "Unable to cast the received data to the specified type (holoscan::gxf::Entity) for "
          "input "
          "'{}:{}'",
          name,
          index);
      HOLOSCAN_LOG_DEBUG(error_message);
      return false;
    } else if constexpr (is_one_of_derived_v<typename DataT::value_type, holoscan::TensorMap>) {
      TensorMap tensor_map;
      try {
        auto gxf_entity = std::any_cast<holoscan::gxf::Entity>(value);
        bool is_tensor_map_populated = populate_tensor_map(gxf_entity, tensor_map);
        if (!is_tensor_map_populated) {
          error_message = fmt::format(
              "Unable to populate the TensorMap from the received GXF Entity for input '{}:{}'",
              name,
              index);
          HOLOSCAN_LOG_DEBUG(error_message);
          return false;
        }
      } catch (const std::bad_any_cast& e) {
        error_message = fmt::format(
            "Unable to cast the received data to the specified type (holoscan::TensorMap) for "
            "input "
            "'{}:{}'",
            name,
            index);
        HOLOSCAN_LOG_DEBUG(error_message);
        return false;
      }
      input_vector.push_back(std::move(tensor_map));
    } else {
      error_message = fmt::format(
          "Unable to cast the received data to the specified type for input '{}:{}' of type {}: {}",
          name,
          index,
          value.type().name(),
          error_message);
      HOLOSCAN_LOG_DEBUG(error_message);
      return false;
    }
    return true;
  }

  template <typename DataT>
  inline holoscan::expected<DataT, holoscan::RuntimeError> receive_single_value(const char* name) {
    auto value = receive_impl(name);
    const std::type_info& value_type = value.type();

    if (value_type == typeid(NoMessageType)) {
      return make_unexpected<holoscan::RuntimeError>(
          create_receive_error(name, "No message received from the input port"));
    } else if (value_type == typeid(NoAccessibleMessageType)) {
      auto casted_value = std::any_cast<NoAccessibleMessageType>(value);
      HOLOSCAN_LOG_ERROR(static_cast<std::string>(casted_value));
      auto error_message = std::move(static_cast<std::string>(casted_value));
      return make_unexpected<holoscan::RuntimeError>(
          create_receive_error(name, error_message.c_str()));
    }

    try {
      if constexpr (std::is_same_v<DataT, std::any>) {
        return value;
      } else if (value_type == typeid(std::nullptr_t)) {
        return handle_null_value<DataT>();
      } else if constexpr (is_one_of_derived_v<DataT, nvidia::gxf::Entity>) {
        // Handle nvidia::gxf::Entity
        return std::any_cast<DataT>(value);
      } else if constexpr (is_one_of_derived_v<DataT, holoscan::TensorMap>) {
        // Handle holoscan::TensorMap
        TensorMap tensor_map;
        bool is_tensor_map_populated =
            populate_tensor_map(std::any_cast<holoscan::gxf::Entity>(value), tensor_map);
        if (!is_tensor_map_populated) {
          auto error_message = fmt::format(
              "Unable to populate the TensorMap from the received GXF Entity for input '{}'", name);
          HOLOSCAN_LOG_DEBUG(error_message);
          return make_unexpected<holoscan::RuntimeError>(
              create_receive_error(name, error_message.c_str()));
        }
        return tensor_map;
      } else {
        return std::any_cast<DataT>(value);
      }
    } catch (const std::bad_any_cast& e) {
      auto error_message = fmt::format(
          "Unable to cast the received data to the specified type for input '{}' of type {}",
          name,
          value.type().name());
      HOLOSCAN_LOG_DEBUG(error_message);

      return make_unexpected<holoscan::RuntimeError>(
          create_receive_error(name, error_message.c_str()));
    }
  }

  inline holoscan::RuntimeError create_receive_error(const char* name, const char* message) {
    auto error_message = fmt::format("ReceiveError on input port '{}': {}", name, message);
    HOLOSCAN_LOG_DEBUG(error_message);
    return holoscan::RuntimeError(holoscan::ErrorCode::kReceiveError, error_message.c_str());
  }

  template <typename DataT>
  inline holoscan::expected<DataT, holoscan::RuntimeError> handle_null_value() {
    if constexpr (holoscan::is_shared_ptr_v<DataT> || std::is_pointer_v<DataT>) {
      return DataT{nullptr};
    } else {
      auto error_message = "Received nullptr for a non-pointer type";
      return make_unexpected<holoscan::RuntimeError>(create_receive_error("input", error_message));
    }
  }
  // --------------- End of helper functions for the receive method ---------------

  ExecutionContext* execution_context_ =
      nullptr;              ///< The execution context that is associated with.
  Operator* op_ = nullptr;  ///< The operator that this context is associated with.
  std::unordered_map<std::string, std::shared_ptr<IOSpec>>& inputs_;  ///< The inputs.
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
                std::unordered_map<std::string, std::shared_ptr<IOSpec>>& outputs)
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
  std::unordered_map<std::string, std::shared_ptr<IOSpec>>& outputs() const { return outputs_; }

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
   *   void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
   *                [[maybe_unused]] ExecutionContext& context) override {
   *     auto value = std::make_shared<ValueData>(7);
   *     op_output.emit(value, "out");
   *   }
   * };
   * ```
   *
   * @tparam DataT The type of the data to send.
   * @param data The shared pointer to the data.
   * @param name The name of the output port.
   * @param acq_timestamp The time when the message is acquired. For instance, this would generally
   *                      be the timestamp of the camera when it captures an image.
   */
  template <typename DataT, typename = std::enable_if_t<!holoscan::is_one_of_derived_v<
                                DataT, nvidia::gxf::Entity, std::any>>>
  void emit(std::shared_ptr<DataT>& data, const char* name = nullptr,
            const int64_t acq_timestamp = -1) {
    emit_impl(data, name, OutputType::kSharedPointer, acq_timestamp);
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
   *   void compute(InputContext& op_input, OutputContext& op_output,
   *                [[maybe_unused]] ExecutionContext& context) override
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
   * @param acq_timestamp The time when the message is acquired. For instance, this would generally
   *                      be the timestamp of the camera when it captures an image.
   */
  template <typename DataT,
            typename = std::enable_if_t<holoscan::is_one_of_derived_v<DataT, nvidia::gxf::Entity>>>
  void emit(DataT& data, const char* name = nullptr, const int64_t acq_timestamp = -1) {
    // if it is the same as nvidia::gxf::Entity then just pass it to emit_impl
    if constexpr (holoscan::is_one_of_v<DataT, nvidia::gxf::Entity>) {
      emit_impl(data, name, OutputType::kGXFEntity, acq_timestamp);
    } else {
      // Convert it to nvidia::gxf::Entity and then pass it to emit_impl
      // Otherwise, we will lose the type information and cannot cast appropriately in emit_impl
      emit_impl(nvidia::gxf::Entity(data), name, OutputType::kGXFEntity, acq_timestamp);
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
   *   void compute(InputContext& op_input, OutputContext& op_output,
   *                [[maybe_unused]] ExecutionContext& context) override
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
   * @param acq_timestamp The time when the message is acquired. For instance, this would generally
   *                      be the timestamp of the camera when it captures an image.
   */
  template <typename DataT,
            typename = std::enable_if_t<!holoscan::is_one_of_derived_v<DataT, nvidia::gxf::Entity>>>
  void emit(DataT data, const char* name = nullptr, const int64_t acq_timestamp = -1) {
    emit_impl(std::move(data), name, OutputType::kAny, acq_timestamp);
  }

  void emit(holoscan::TensorMap& data, const char* name = nullptr,
            const int64_t acq_timestamp = -1) {
    auto out_message = holoscan::gxf::Entity::New(execution_context_);
    for (auto& [key, tensor] : data) { out_message.add(tensor, key.c_str()); }
    emit(out_message, name, acq_timestamp);
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
   * @param acq_timestamp The timestamp to publish in the output message. The default value of -1
   * does not publish a timestamp.
   */
  virtual void emit_impl([[maybe_unused]] std::any data,
                         [[maybe_unused]] const char* name = nullptr,
                         [[maybe_unused]] OutputType out_type = OutputType::kSharedPointer,
                         [[maybe_unused]] const int64_t acq_timestamp = -1) {}

  ExecutionContext* execution_context_ =
      nullptr;              ///< The execution context that is associated with.
  Operator* op_ = nullptr;  ///< The operator that this context is associated with.
  std::unordered_map<std::string, std::shared_ptr<IOSpec>>& outputs_;  ///< The outputs.
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_IO_CONTEXT_HPP */
