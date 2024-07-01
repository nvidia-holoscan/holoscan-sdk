/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/signal_handler.hpp"

#include <memory>
#include <utility>

#include "holoscan/logger/logger.hpp"

namespace holoscan {

// Static member initializations
struct sigaction SignalHandler::signal_handler_ {};

void static_handle_signal(int signal) {
  SignalHandler::static_handle_signal(signal);
}

SignalHandler& SignalHandler::get_instance() {
  static SignalHandler instance;
  return instance;
}

void SignalHandler::static_handle_signal(int signal) {
  SignalHandler::get_instance().handle_signal(signal);
}

void SignalHandler::install_signal_handler(int signal) {
  SignalHandler::get_instance().install_signal_handler_impl(signal);
}

void SignalHandler::register_global_signal_handler(int signal, std::function<void(int)> handler,
                                                   bool overwrite) {
  SignalHandler::get_instance().register_global_signal_handler_impl(
      signal, std::move(handler), overwrite);
}

void SignalHandler::register_signal_handler(void* context, int signal,
                                            std::function<void(void*, int)> handler,
                                            bool overwrite) {
  SignalHandler::get_instance().register_signal_handler_impl(
      context, signal, std::move(handler), overwrite);
}

void SignalHandler::unregister_global_signal_handler(int signal) {
  SignalHandler::get_instance().unregister_global_signal_handler_impl(signal);
}

void SignalHandler::unregister_signal_handler(void* context, int signal) {
  SignalHandler::get_instance().unregister_signal_handler_impl(context, signal);
}

void SignalHandler::clear_all_signal_handlers() {
  clear_global_signal_handlers();
  clear_signal_handlers();
}

void SignalHandler::clear_global_signal_handlers() {
  auto& global_signal_handlers = SignalHandler::get_instance().global_signal_handlers_;
  HOLOSCAN_LOG_DEBUG("Clearing global signal handlers (size: {})", global_signal_handlers.size());
  global_signal_handlers.clear();
}

void SignalHandler::clear_signal_handlers() {
  auto& signal_handlers = SignalHandler::get_instance().signal_handlers_;
  HOLOSCAN_LOG_DEBUG("Clearing signal handlers (size: {})", signal_handlers.size());
  signal_handlers.clear();
}

SignalHandler::SignalHandler() {
  signal_handler_.sa_handler = SignalHandler::static_handle_signal;
  sigemptyset(&signal_handler_.sa_mask);
  signal_handler_.sa_flags = 0;

  // Save old handler when registering new one
  struct sigaction old_signal_handler;

  sigaction(SIGINT, &signal_handler_, &old_signal_handler);
  old_signal_handlers_[SIGINT] = old_signal_handler;

  sigaction(SIGTERM, &signal_handler_, &old_signal_handler);
  old_signal_handlers_[SIGTERM] = old_signal_handler;
}

SignalHandler::~SignalHandler() {
  // Clear handler
  signal_handler_.sa_handler = nullptr;
  sigemptyset(&signal_handler_.sa_mask);
  signal_handler_.sa_flags = 0;

  signal_handlers_.clear();

  // Restore old handler
  for (auto& [signal, old_signal_handler] : old_signal_handlers_) {
    sigaction(signal, &old_signal_handler, nullptr);
  }
}

void SignalHandler::install_signal_handler_impl(int signal) {
  if (signal != 0) {
    // Install handler for specific signal
    HOLOSCAN_LOG_DEBUG("Installing signal handler for signal {}", signal);
    sigaction(signal, &signal_handler_, nullptr);  // can ignore storing old handler
    return;
  }

  for (auto& [sig, handler] : old_signal_handlers_) {
    HOLOSCAN_LOG_DEBUG("Installing signal handler for signal {}", sig);
    sigaction(sig, &signal_handler_, nullptr);  // can ignore storing old handler
  }
}

void SignalHandler::register_global_signal_handler_impl(int signal,
                                                        std::function<void(int)> handler,
                                                        bool overwrite) {
  std::lock_guard lock(signal_handlers_mutex_);
  if (!overwrite) {
    // Check if handler is already registered
    auto it = global_signal_handlers_.find(signal);
    if (it != global_signal_handlers_.end()) {
      HOLOSCAN_LOG_DEBUG("Global signal handler for signal {} already registered. Skipping",
                         signal);
      return;
    }
  }

  HOLOSCAN_LOG_DEBUG("Registering global signal handler for signal {}", signal);
  global_signal_handlers_[signal] = std::move(handler);
}

void SignalHandler::register_signal_handler_impl(void* context, int signal,
                                                 std::function<void(void*, int)> handler,
                                                 bool overwrite) {
  std::lock_guard lock(signal_handlers_mutex_);
  if (!overwrite) {
    // Check if handler is already registered
    auto it = signal_handlers_.find(context);
    if (it != signal_handlers_.end()) {
      auto it2 = it->second.find(signal);
      if (it2 != it->second.end()) {
        HOLOSCAN_LOG_DEBUG(
            "Signal ({}) handler for context: {} already registered. Skipping", signal, context);
        return;
      }
    }
  }

  HOLOSCAN_LOG_DEBUG("Registering signal ({}) handler for context: {}", signal, context);
  signal_handlers_[context][signal] = std::move(handler);
}

void SignalHandler::unregister_global_signal_handler_impl(int signal) {
  std::lock_guard lock(signal_handlers_mutex_);
  auto it = global_signal_handlers_.find(signal);
  if (it != global_signal_handlers_.end()) {
    HOLOSCAN_LOG_DEBUG("Unregistering global signal handler for signal {}", signal);
    global_signal_handlers_.erase(signal);
  }
}

void SignalHandler::unregister_signal_handler_impl(void* context, int signal) {
  std::lock_guard lock(signal_handlers_mutex_);
  auto it = signal_handlers_.find(context);
  if (it != signal_handlers_.end()) {
    HOLOSCAN_LOG_DEBUG("Unregistering signal ({}) handler for context: {}", signal, context);
    it->second.erase(signal);
    if (it->second.empty()) { signal_handlers_.erase(it); }
  }
}

void SignalHandler::handle_signal(int signal) {
  std::lock_guard lock(signal_handlers_mutex_);

  // Call global handlers. This takes precedence over context specific handlers
  auto it = global_signal_handlers_.find(signal);
  if (it != global_signal_handlers_.end()) {
    // Call registered handler
    HOLOSCAN_LOG_DEBUG("Calling global signal handler for signal {}", signal);
    it->second(signal);
  } else {
    for (auto& [context, signal_handler_map] : signal_handlers_) {
      auto it = signal_handler_map.find(signal);
      if (it != signal_handler_map.end()) {
        // Call registered handler for each context
        HOLOSCAN_LOG_DEBUG("Calling signal ({}) handler for context: {}", signal, context);
        it->second(context, signal);
      }
    }
  }

  // If the existing handler exists, pass it to the old signal handler
  auto old_signal_handler_it = old_signal_handlers_.find(signal);
  if (old_signal_handler_it != old_signal_handlers_.end()) {
    auto& old_signal_handler = old_signal_handler_it->second;

    if (old_signal_handler.sa_handler != nullptr && old_signal_handler.sa_handler != SIG_IGN &&
        old_signal_handler.sa_handler != SIG_DFL) {
      HOLOSCAN_LOG_DEBUG("Calling old signal handler for signal {}", signal);
      old_signal_handler.sa_handler(signal);
    }
  }
}

}  // namespace holoscan
