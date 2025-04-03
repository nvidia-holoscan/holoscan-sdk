/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_CORE_EMITTER_RECEIVER_REGISTRY_HPP
#define HOLOSCAN_CORE_EMITTER_RECEIVER_REGISTRY_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // needed for py::cast to work with STL container types

#include <any>
#include <functional>
#include <memory>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "holoscan/core/errors.hpp"
#include "holoscan/core/expected.hpp"
#include "holoscan/logger/logger.hpp"
#include "io_context.hpp"

namespace py = pybind11;

namespace holoscan {

/* Emit and receive of any type T that pybind11 can cast via pybind11::object.cast<T>() and
 * py::cast().
 *
 * For example emitter_receiver<std::string> could be used to convert between C++ and Python
 * strings.
 */
template <typename T>
struct emitter_receiver {
  static void emit(py::object& data, const std::string& name, PyOutputContext& op_output,
                   const int64_t acq_timestamp = -1) {
    auto cpp_type = data.cast<T>();
    py::gil_scoped_release release;
    op_output.emit<T>(std::move(cpp_type), name.c_str(), acq_timestamp);
    return;
  }

  static py::object receive(std::any result, const std::string& name, PyInputContext& op_input) {
    auto cpp_obj = std::any_cast<T>(result);
    return py::cast(cpp_obj);
  }
};

/* Implements a receiver for the array<float, 16> camera pose type accepted by HolovizOp.
 */
template <typename T>
struct emitter_receiver<std::shared_ptr<T>> {
  static void emit(py::object& data, const std::string& name, PyOutputContext& op_output,
                   const int64_t acq_timestamp = -1) {
    auto cpp_obj = std::make_shared<T>(data.cast<T>());
    py::gil_scoped_release release;
    op_output.emit<std::shared_ptr<T>>(std::move(cpp_obj), name.c_str(), acq_timestamp);
    return;
  }
  static py::object receive(std::any result, const std::string& name, PyInputContext& op_input) {
    auto camera_pose = std::any_cast<std::shared_ptr<T>>(result);
    py::object py_camera_pose = py::cast(*camera_pose);
    return py_camera_pose;
  }
};

/**
 * @brief Class to set emitter/receivers for data types.
 *
 * This class is used to set emitter/receivers (emitter + receiver) for data types.
 */
class PYBIND11_EXPORT EmitterReceiverRegistry {
 public:
  /**
   * @brief Function type for emitting a data type
   */
  using EmitFunc = std::function<void(py::object&, const std::string&, PyOutputContext& op_output,
                                      const int64_t acq_timestamp)>;

  /**
   * @brief Function type for receiving a data type
   */
  using ReceiveFunc =
      std::function<py::object(std::any, const std::string&, PyInputContext& op_input)>;

  /**
   * @brief Function tuple type for emitting and receiving a data type
   */
  using EmitterReceiver = std::pair<EmitFunc, ReceiveFunc>;

  inline static EmitFunc none_emit = []([[maybe_unused]] py::object& data,
                                        [[maybe_unused]] const std::string& name,
                                        [[maybe_unused]] PyOutputContext& op_output,
                                        [[maybe_unused]] const int64_t acq_timestamp = -1) -> void {
    HOLOSCAN_LOG_ERROR(
        "Unable to emit message (op: '{}', port: '{}')", op_output.op()->name(), name);
    return;
  };

  inline static ReceiveFunc none_receive =
      []([[maybe_unused]] std::any result, [[maybe_unused]] const std::string& name,
         [[maybe_unused]] PyInputContext& op_input) -> py::object {
    HOLOSCAN_LOG_ERROR(
        "Unable to receive message (op: '{}', port: '{}')", op_input.op()->name(), name);
    return py::none();
  };

  /**
   * @brief Default @ref EmitterReceiver for Arg.
   */
  inline static EmitterReceiver none_emitter_receiver = std::make_pair(none_emit, none_receive);

  /**
   * @brief Get the instance object.
   *
   * @return The reference to the EmitterReceiverRegistry instance.
   */
  static EmitterReceiverRegistry& get_instance();

  /**
   * @brief Emit the message object.
   *
   * @tparam typeT The data type within the message.
   * @param data The Python object corresponding to the data of typeT.
   * @param name The name of the entity emitted.
   * @param op_output The PyOutputContext used to emit the data.
   * @param acq_timestamp The acquisition timestamp of the data.
   */
  template <typename typeT>
  static void emit(py::object& data, const std::string& name, PyOutputContext& op_output,
                   const int64_t acq_timestamp = -1) {
    auto& instance = get_instance();
    const std::type_index index = std::type_index(typeid(typeT));
    const EmitFunc& func = instance.get_emitter(index);
    return func(data, name, op_output, acq_timestamp);
  }

  /**
   * @brief Receive the message object.
   *
   * @tparam typeT The data type within the message.
   * @param message The message to serialize.
   * @param endpoint The serialization endpoint (buffer).
   */
  template <typename typeT>
  static py::object receive(std::any result, const std::string& name, PyInputContext& op_input) {
    auto& instance = get_instance();
    const std::type_index index = std::type_index(typeid(typeT));
    const ReceiveFunc& func = instance.get_receiver(index);
    return func(result, name, op_input);
  }

  /**
   * @brief Get the emitter/receiver function tuple.
   *
   * @param index The type index of the parameter.
   * @return The reference to the EmitterReceiver object.
   */
  const EmitterReceiver& get_emitter_receiver(const std::type_index& index) const;

  /**
   * @brief Check if a given emitter/receiver exists based on the type index.
   *
   * @param index The type index of the parameter.
   * @return boolean indicating if an emitter/receiver exists for the given index.
   */
  bool has_emitter_receiver(const std::type_index& index) const;

  /**
   * @brief Get the emitter/receiver function tuple.
   *
   * @param name The name of the emitter/receiver.
   * @return The reference to the EmitterReceiver object.
   */
  const EmitterReceiver& get_emitter_receiver(const std::string& name) const;

  /**
   * @brief Get the emitter function.
   *
   * @param name The name of the emitter/receiver.
   * @return The reference to the emitter function.
   */
  const EmitFunc& get_emitter(const std::string& name) const;

  /**
   * @brief Get the emitter function.
   *
   * @param index The type index of the parameter.
   * @return The reference to the emitter function.
   */
  const EmitFunc& get_emitter(const std::type_index& index) const;

  /**
   * @brief Get the receiver function.
   *
   * @param name The name of the emitter/receiver.
   * @return The reference to the receiver function.
   */
  const ReceiveFunc& get_receiver(const std::string& name) const;

  /**
   * @brief Get the receiver function.
   *
   * @param index The type index of the parameter.
   * @return The reference to the receiver function.
   */
  const ReceiveFunc& get_receiver(const std::type_index& index) const;

  /**
   * @brief Get the std::type_index corresponding to a emitter/receiver name
   *
   * @param name The name of the emitter/receiver.
   * @return The std::type_index corresponding to the name.
   */
  expected<std::type_index, RuntimeError> name_to_index(const std::string& name) const;

  /**
   * @brief Get the name corresponding to a std::type_index
   *
   * @param index The std::type_index corresponding to the parameter.
   * @return The name of the emitter/receiver.
   */
  expected<std::string, RuntimeError> index_to_name(const std::type_index& index) const;

  /**
   * @brief Add a emitter/receiver for the type.
   *
   * @tparam typeT the type for which a emitter/receiver is being added
   * @param name The name of the emitter/receiver to add.
   * @param overwrite if true, any existing emitter/receiver with matching name will be overwritten.
   */
  template <typename typeT>
  void add_emitter_receiver(const std::string& name, bool overwrite = false) {
    auto name_search = name_to_index_map_.find(name);
    auto index = std::type_index(typeid(typeT));
    if (name_search != name_to_index_map_.end()) {
      if (!overwrite) {
        HOLOSCAN_LOG_WARN(
            "Existing emitter_receiver for name '{}' found, keeping the previous one.", name);
        return;
      }
      if (index != name_search->second) {
        HOLOSCAN_LOG_ERROR(
            "Existing emitter_receiver for name '{}' found, but with non-matching type_index. ",
            "If you did not intend to replace the existing emitter_receiver, please choose a "
            "different name.",
            name);
      }
      HOLOSCAN_LOG_DEBUG("Replacing existing emitter_receiver with name '{}'.", name);
      emitter_receiver_map_.erase(name);
    }
    HOLOSCAN_LOG_DEBUG("Added emitter/receiver for type named: {}", name);
    name_to_index_map_.try_emplace(name, index);
    index_to_name_map_.try_emplace(index, name);
    emitter_receiver_map_.emplace(
        name, std::make_pair(emitter_receiver<typeT>::emit, emitter_receiver<typeT>::receive));
  }

  /**
   * @brief List the names of the types with an emitter and/or receiver registered.
   *
   * @return A vector of the names of the types with an emitter and/or receiver registered.
   */
  std::vector<std::string> registered_types() const;

 private:
  // private constructor (retrieve static instance via get_instance)
  EmitterReceiverRegistry() {}

  // define maps to and from type_index and string (since type_index may vary across platforms)
  std::unordered_map<std::type_index, std::string>
      index_to_name_map_;  ///< Mapping from type_index to name
  std::unordered_map<std::string, std::type_index>
      name_to_index_map_;  ///< Mapping from name to type_index

  std::unordered_map<std::string, EmitterReceiver>
      emitter_receiver_map_;  ///< Map of emitter/receiver name to function tuple
};

}  // namespace holoscan

#endif /* HOLOSCAN_CORE_EMITTER_RECEIVER_REGISTRY_HPP */
