/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_UTILS_OPERATOR_RUNNER_HPP
#define HOLOSCAN_UTILS_OPERATOR_RUNNER_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "holoscan/core/conditions/gxf/asynchronous.hpp"
#include "holoscan/core/domain/tensor_map.hpp"
#include "holoscan/core/gxf/gxf_wrapper.hpp"
#include "holoscan/core/operator.hpp"

namespace holoscan::ops {

/**
 * @brief A class to run an operator independently from the workflow.
 *
 * This class provides methods to push entities to the operator's input ports and pop entities from
 * the operator's output ports. It also provides a method to run the operator separately from the
 * workflow.
 *
 * This class uses the GXFWrapper to set up and run the operator.
 * The `start()` and `stop()` methods are called automatically when the operator is started or
 * stopped. It uses AsynchronousCondition (with `AsynchronousEventState::WAIT`) to block the
 * operator execution (calling the `compute()` method) from the holoscan executor, allowing the
 * operator to run independently from the workflow via the `run()` method.
 *
 * Input data can be pushed to the operator's input ports using the `push_input()` method before
 * calling `run()`. Output data can be retrieved from the operator's output ports using the
 * `pop_output()` method after calling `run()`.
 *
 * Both `push_input()` and `pop_output()` methods support the templated `DataT` type, which can be:
 * - Primitive types (int, float, etc.)
 * - Custom classes/structs (must be copyable/movable)
 * - Smart pointers (std::shared_ptr, std::unique_ptr)
 * - Standard library containers (std::vector, std::string, etc.)
 * - std::any
 * - holoscan::gxf::Entity
 * - holoscan::TensorMap
 *
 * Example types:
 * ```cpp
 * // Primitive types
 * runner.pop_output<int>("port");
 * runner.pop_output<float>("port");
 *
 * // Custom types
 * runner.pop_output<MyCustomClass>("port");
 *
 * // Smart pointers
 * runner.pop_output<std::shared_ptr<int>>("port");
 * runner.pop_output<std::shared_ptr<MyCustomClass>>("port");
 *
 * // Standard containers
 * runner.pop_output<std::vector<float>>("port");
 * runner.pop_output<std::string>("port");
 *
 * // std::any
 * runner.pop_output<std::any>("port");
 *
 * // holoscan::gxf::Entity
 * runner.pop_output<holoscan::gxf::Entity>("port");
 *
 * // holoscan::TensorMap
 * runner.pop_output<holoscan::TensorMap>("port");
 * ```
 *
 * If the template type is not specified for `pop_output()` method,
 * the method will return a holoscan::gxf::Entity.
 *
 * @note This class is not thread-safe.
 * @note This is an experimental feature. The API may change in future releases.
 *
 * Example usage:
 * ```cpp
 * // In the operator's initialize() method (after holoscan::Operator::initialize() is called):
 *
 * // Create internal operators using the fragment
 * auto frag = fragment();  // fragment() is a method to get the fragment object in the operator
 *
 * auto receiver_operator = frag->make_operator<FirstOp>("receiver",
 *     holoscan::Arg("parameter_1", param_value_1),
 *     holoscan::Arg("parameter_2", param_value_2));
 * // Wrap the operator with OperatorRunner
 * auto op_first_operator = std::make_shared<holoscan::ops::OperatorRunner>(receiver_operator);
 *
 * // Create the next internal operator
 * auto next_operator = ...; // Create the next internal operator
 * // Wrap the next operator with OperatorRunner
 * auto op_next_operator_ = std::make_shared<holoscan::ops::OperatorRunner>(next_operator);
 *
 * // In the operator's compute() method:
 * void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
 *     holoscan::ExecutionContext& context)
 * {
 *     // Run the first internal operator
 *     op_first_operator->run();
 *
 *     // Get output from the first internal operator
 *     auto output = op_first_operator->pop_output("output");
 *     if (!output) {
 *         HOLOSCAN_LOG_ERROR("Failed to pop output from operator {} - {}",
 *                            op_first_operator->op()->name(),
 *                            output.error().what());
 *         throw std::runtime_error(output.error().what());
 *     }
 *
 *     // Push output to the next internal operator in the chain
 *     auto result = op_next_operator_->push_input("input", output.value());
 *     if (!result) {
 *         HOLOSCAN_LOG_ERROR("Failed to push input to operator {} - {}",
 *                            op_next_operator_->op()->name(),
 *                            result.error().what());
 *         throw std::runtime_error(result.error().what());
 *     }
 *     op_next_operator_->run();
 *
 *     // Get output from the next internal operator
 *     auto next_output = op_next_operator_->pop_output("output");
 *     if (!next_output) {
 *         HOLOSCAN_LOG_ERROR("Failed to pop output from operator {} - {}",
 *                            op_next_operator_->op()->name(),
 *                            next_output.error().what());
 *         throw std::runtime_error(next_output.error().what());
 *     }
 *
 *     // Emit the output from this operator
 *     op_output.emit(next_output.value(), "output");
 * }
 * ```
 */
class OperatorRunner {
 public:
  /**
   * @brief Construct a new Operator Runner object.
   *
   * @param op The operator to run.
   */
  explicit OperatorRunner(const std::shared_ptr<holoscan::Operator>& op);

  /**
   * @brief Get the operator.
   *
   * @return The shared pointer to the operator.
   */
  const std::shared_ptr<holoscan::Operator>& op() const;

  /**
   * @brief Push data to the specified input port of the operator.
   *
   * This method takes a GXF entity and pushes it to the input port identified by port_name.
   * The entity will be available to the operator during its next compute cycle.
   *
   * @param port_name The name of the input port.
   * @param entity The entity to push to the input port.
   * @return A holoscan::expected containing either:
   *         - void if successful
   *         - A holoscan::RuntimeError if the input port is not found
   */
  holoscan::expected<void, holoscan::RuntimeError> push_input(const std::string& port_name,
                                                              nvidia::gxf::Entity& entity);

  /**
   * @brief Push data to the specified input port of the operator.
   *
   * This method takes a GXF entity and pushes it to the input port identified by port_name.
   * The entity will be available to the operator during its next compute cycle.
   *
   * @param port_name The name of the input port.
   * @param entity The entity to push to the input port.
   * @return A holoscan::expected containing either:
   *         - void if successful
   *         - A holoscan::RuntimeError if the input port is not found
   */
  holoscan::expected<void, holoscan::RuntimeError> push_input(const std::string& port_name,
                                                              nvidia::gxf::Entity&& entity);

  /**
   * @brief Push data to the specified input port of the operator.
   *
   * This method takes a data object and pushes it to the input port identified by port_name.
   * The data will be wrapped in a GXF entity and available to the operator during its next compute
   * cycle.
   *
   * @tparam DataT The type of data to push, must not be derived from nvidia::gxf::Entity
   * @param port_name The name of the input port
   * @param data The data to push to the input port
   * @return A holoscan::expected containing either:
   *         - void if successful
   *         - A holoscan::RuntimeError if the input port is not found
   */
  template <typename DataT,
            typename = std::enable_if_t<!holoscan::is_one_of_derived_v<DataT, nvidia::gxf::Entity>>>
  holoscan::expected<void, holoscan::RuntimeError> push_input(const std::string& port_name,
                                                              DataT data) {
    // Create an Entity object and add a Message object to it.
    auto gxf_entity = nvidia::gxf::Entity::New(gxf_context_);
    auto buffer = gxf_entity.value().add<Message>();

    // Set the data to the value of the Message object.
    buffer.value()->set_value(std::move(data));
    auto entity = gxf_entity.value();

    return push_input(port_name, entity);
  }

  /**
   * @brief Push a TensorMap to the specified input port of the operator.
   *
   * This method takes a TensorMap and pushes it to the input port identified by port_name.
   * The TensorMap will be available to the operator during its next compute cycle.
   *
   * @param port_name The name of the input port.
   * @param data The TensorMap to push to the input port.
   * @return A holoscan::expected containing either:
   *         - void if successful
   *         - A holoscan::RuntimeError if the input port is not found
   */
  holoscan::expected<void, holoscan::RuntimeError> push_input(const std::string& port_name,
                                                              const holoscan::TensorMap& data);

  /**
   * @brief Executes one compute cycle of the operator.
   *
   * Internally, it calls the `compute()` method of the operator via the GXFWrapper's `tick()`
   * method.
   */
  void run();

  /**
   * @brief Retrieves and removes an entity from the specified output port.
   *
   * This method retrieves and removes an entity from the output port identified by port_name.
   * The entity is removed from the output port and returned.
   *
   * @param port_name The name of the output port.
   * @return The entity popped from the output port.
   */
  holoscan::expected<holoscan::gxf::Entity, holoscan::RuntimeError> pop_output(
      const std::string& port_name);

  /**
   * @brief Pop data of a specified type from the output port.
   *
   * This templated method retrieves and removes data of type DataT from the output port identified
   * by port_name. The data is removed from the output port and returned.
   *
   * The method handles several data types differently:
   * - For std::any: Returns the raw message value
   * - For nvidia::gxf::Entity derived types: Returns the entity directly
   * - For holoscan::TensorMap: Populates and returns a TensorMap from the entity
   * - For other types: Attempts to cast the message value to the requested type
   *
   * @tparam DataT The type of data to retrieve from the output port.
   * @param port_name The name of the output port.
   * @return A holoscan::expected containing either:
   *         - The data of type DataT if successful
   *         - A holoscan::RuntimeError if:
   *           - The output port is not found
   *           - The message cannot be popped from the transmitter
   *           - The data cannot be cast to the requested type
   *           - The TensorMap cannot be populated (for TensorMap type)
   */
  template <typename DataT>
  holoscan::expected<DataT, holoscan::RuntimeError> pop_output(const std::string& port_name) {
    // Find the transmitter for the given port
    auto it = double_buffer_transmitters_.find(port_name);
    if (it == double_buffer_transmitters_.end()) {
      return create_error(
          holoscan::ErrorCode::kNotFound,
          "The output port ('{}') with DoubleBufferTransmitter not found in the operator '{}'",
          port_name,
          op_->name());
    }

    // Sync the output port
    auto result = it->second->sync();
    if (!result) {
      return create_error(
          holoscan::ErrorCode::kFailure,
          "Failed to sync output port {} with DoubleBufferTransmitter in operator {}",
          port_name,
          op_->name(),
          result.get_error_message());
    }

    // Pop the message from the transmitter
    auto maybe_message = it->second->pop();
    if (!maybe_message) {
      return create_error(holoscan::ErrorCode::kReceiveError,
                          "Failed to pop message from the output port ('{}') with "
                          "DoubleBufferTransmitter in operator '{}'",
                          port_name,
                          op_->name());
    }

    auto& entity = maybe_message.value();

    // Handle std::any type
    if constexpr (std::is_same_v<DataT, std::any>) {
      auto message = entity.get<holoscan::Message>();
      if (!message) {
        return holoscan::gxf::Entity(entity);  // handle gxf::Entity as is
      }
      return message.value()->value();
    }

    // Handle GXF Entity types
    if constexpr (is_one_of_derived_v<DataT, nvidia::gxf::Entity>) { return DataT(entity); }

    // Handle TensorMap type
    if constexpr (is_one_of_derived_v<DataT, holoscan::TensorMap>) {
      return handle_tensor_map_output(holoscan::gxf::Entity(entity), port_name);
    }

    // Handle all other types
    return handle_message_output<DataT>(entity, port_name);
  }

 protected:
  bool populate_tensor_map(const holoscan::gxf::Entity& gxf_entity,
                           holoscan::TensorMap& tensor_map);

  std::shared_ptr<holoscan::Operator> op_;  ///< The operator to run.
  void* gxf_context_ = nullptr;             ///< The GXF context to interact with the operator.
  holoscan::gxf::GXFWrapper* gxf_wrapper_ =
      nullptr;  ///< The underlying GXFWrapper to run the operator.
  std::shared_ptr<holoscan::AsynchronousCondition>
      async_condition_;  ///< The asynchronous condition to block the operator execution.

  /// Double buffer receivers corresponding to the operator's input ports (input port name ->
  /// receiver).
  std::unordered_map<std::string, nvidia::gxf::DoubleBufferReceiver*> double_buffer_receivers_;
  /// Double buffer transmitters corresponding to the operator's output ports (output port name ->
  /// transmitter).
  std::unordered_map<std::string, nvidia::gxf::DoubleBufferTransmitter*>
      double_buffer_transmitters_;

 private:
  template <typename DataT>
  holoscan::expected<DataT, holoscan::RuntimeError> handle_message_output(
      const nvidia::gxf::Entity& entity, const std::string& port_name) {
    auto message = entity.get<holoscan::Message>();
    if (!message) {
      return create_error(
          holoscan::ErrorCode::kReceiveError,
          "Unable to get the holoscan::Message from the received GXF Entity for output '{}'",
          port_name);
    }

    try {
      return std::any_cast<DataT>(message.value()->value());
    } catch (const std::bad_any_cast& e) {
      return create_error(
          holoscan::ErrorCode::kReceiveError,
          "Unable to cast the received data to the specified type for output '{}' of type {}",
          port_name,
          message.value()->value().type().name());
    }
  }

  holoscan::expected<holoscan::TensorMap, holoscan::RuntimeError> handle_tensor_map_output(
      const holoscan::gxf::Entity& entity, const std::string& port_name);

  template <typename... Args>
  holoscan::unexpected<holoscan::RuntimeError> create_error(holoscan::ErrorCode code,
                                                            const char* format, Args&&... args) {
    auto message = fmt::format(format, std::forward<Args>(args)...);
    HOLOSCAN_LOG_DEBUG(message);
    return holoscan::unexpected<holoscan::RuntimeError>(
        holoscan::RuntimeError(code, message.c_str()));
  }
};

}  // namespace holoscan::ops

#endif /* HOLOSCAN_UTILS_OPERATOR_RUNNER_HPP */
