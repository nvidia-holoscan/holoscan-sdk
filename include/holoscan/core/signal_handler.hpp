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

#ifndef HOLOSCAN_CORE_SIGNAL_HANDLER_HPP
#define HOLOSCAN_CORE_SIGNAL_HANDLER_HPP

#include <csignal>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace holoscan {

/**
 * @brief The method to handle the specified signal.
 *
 * @param signal The signal to handle.
 */
extern void static_handle_signal(int signal);

/**
 * @brief The SignalHandler class provides a mechanism to handle signals in a C++ program.
 *
 * The SignalHandler class provides a way to handle signals in a C++ program. It allows registering
 * global signal handlers and context-specific signal handlers. The class is implemented as a
 * singleton, and its instance can be obtained using the get_instance() method.
 */
class SignalHandler {
 public:
  /// Type definition for a global signal handler map.
  using GlobalSignalHandlerMap = std::unordered_map<int, std::function<void(int)>>;
  /// Type definition for a signal handler map.
  using SignalHandlerMap = std::unordered_map<int, std::function<void(void*, int)>>;
  /// Type definition for a context-specific signal handler map.
  using ContextSignalHandlerMap = std::unordered_map<void*, SignalHandlerMap>;

  /**
   * @brief Returns the singleton instance of the SignalHandler class.
   *
   * @return SignalHandler& The singleton instance of the SignalHandler class.
   */
  static SignalHandler& get_instance();

  /**
   * @brief The static method to handles the specified signal.
   *
   * @param signal The signal to handle.
   */
  static void static_handle_signal(int signal);

  /**
   * @brief Installs the signal handler for the specified signal.
   *
   * @param signal The signal to install the signal handler for. If signal is 0, the signal handler
   * is installed for all existing signals.
   */
  static void install_signal_handler(int signal = 0);

  /**
   * @brief Registers a global signal handler for the specified signal.
   *
   * @param signal The signal to register the global signal handler for.
   * @param handler The global signal handler function.
   * @param overwrite If true, overwrites any existing global signal handler for the specified
   * signal.
   */
  static void register_global_signal_handler(int signal, std::function<void(int)> handler,
                                             bool overwrite = false);

  /**
   * @brief Registers a context-specific signal handler for the specified signal.
   *
   * @param context The context to register the signal handler for.
   * @param signal The signal to register the signal handler for.
   * @param handler The signal handler function.
   * @param overwrite If true, overwrites any existing signal handler for the specified context and
   * signal.
   */
  static void register_signal_handler(void* context, int signal,
                                      std::function<void(void*, int)> handler,
                                      bool overwrite = false);

  /**
   * @brief Unregisters the global signal handler for the specified signal.
   *
   * @param signal The signal to unregister the global signal handler for.
   */
  static void unregister_global_signal_handler(int signal);

  /**
   * @brief Unregisters the context-specific signal handler for the specified context and signal.
   *
   * @param context The context to unregister the signal handler for.
   * @param signal The signal to unregister the signal handler for.
   */
  static void unregister_signal_handler(void* context, int signal);

  /**
   * @brief Clears all signal handlers.
   */
  static void clear_all_signal_handlers();

  /**
   * @brief Clears all global signal handlers.
   */
  static void clear_global_signal_handlers();

  /**
   * @brief Clears all context-specific signal handlers.
   */
  static void clear_signal_handlers();

 private:
  /// Constructs a SignalHandler object.
  SignalHandler();
  /// Destructs a SignalHandler object.
  ~SignalHandler();

  void install_signal_handler_impl(int signal = 0);

  void register_global_signal_handler_impl(int signal, std::function<void(int)> handler,
                                           bool overwrite = false);
  void register_signal_handler_impl(void* context, int signal,
                                    std::function<void(void*, int)> handler,
                                    bool overwrite = false);

  void unregister_global_signal_handler_impl(int signal);
  void unregister_signal_handler_impl(void* context, int signal);
  void handle_signal(int signal);

  GlobalSignalHandlerMap global_signal_handlers_;  ///< The global signal handler map.
  ContextSignalHandlerMap signal_handlers_;        ///< The context-specific signal handler map.
  std::recursive_mutex signal_handlers_mutex_;  ///< The mutex to protect the signal handler maps.

  static struct sigaction signal_handler_;                         ///< The signal handler struct.
  std::unordered_map<int, struct sigaction> old_signal_handlers_;  ///< The old signal handler map.
};

}  // namespace holoscan

#endif  // HOLOSCAN_CORE_SIGNAL_HANDLER_HPP
