/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <utility>

#include "holoscan/logger/logger.hpp"

namespace holoscan {

// Static member initializations
struct sigaction SignalHandler::signal_handler_{};

// Constants for signal handler timeouts
constexpr auto kSignalHandlerStuckTimeoutMs =
    2000;  // timeout before force exit if signal handler is stuck
constexpr auto kSignalHandlerWatchdogTimeoutSec = 1;  // timeout for watchdog thread

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
  // To maintain async-signal-safety, we need an extremely minimal signal handler
  // Use a static atomic flag that can be safely accessed from multiple threads or signal handlers
  static std::atomic<bool> signal_in_progress{false};
  static std::atomic<std::chrono::steady_clock::time_point> signal_start_time{
      std::chrono::steady_clock::now()};

  // Try to set the flag - if it was already set, check how long it's been active
  bool expected = false;
  if (!signal_in_progress.compare_exchange_strong(expected, true)) {
    // Another thread is already handling a signal - check how long it's been active
    auto current_time = std::chrono::steady_clock::now();
    auto start_time = signal_start_time.load();
    auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time).count();

    // If signal handler is stuck for more than kSignalHandlerStuckTimeoutMs, force exit
    if (elapsed_ms > kSignalHandlerStuckTimeoutMs) {
      // We're stuck in signal handler for too long - force immediate exit
      std::quick_exit(128 + signal);  // Standard exit code for signal termination
    }

    // Just return if another thread is already handling
    return;
  }

  // Update start time for deadlock detection
  signal_start_time.store(std::chrono::steady_clock::now());

  // Since we set the flag, we're responsible for handling the signal
  // We must be extremely careful not to use any non-async-signal-safe functions here

  // Call global handlers first (no mutex)
  auto global_handler_it = global_signal_handlers_.find(signal);
  if (global_handler_it != global_signal_handlers_.end() && global_handler_it->second) {
    // Call the handler without any locks or logging
    global_handler_it->second(signal);
  } else {
    // Call context-specific handlers (no mutex)
    for (auto& [context, signal_handler_map] : signal_handlers_) {
      auto handler_it = signal_handler_map.find(signal);
      if (handler_it != signal_handler_map.end() && handler_it->second) {
        // Call the handler without any locks or logging
        handler_it->second(context, signal);
      }
    }
  }

  // Call old signal handler if it exists
  auto old_signal_handler_it = old_signal_handlers_.find(signal);
  if (old_signal_handler_it != old_signal_handlers_.end()) {
    auto& old_signal_handler = old_signal_handler_it->second;
    if (old_signal_handler.sa_handler != nullptr && old_signal_handler.sa_handler != SIG_IGN &&
        old_signal_handler.sa_handler != SIG_DFL) {
      old_signal_handler.sa_handler(signal);
    }
  }

  // Create a watchdog thread that will force exit if the signal handler itself gets stuck
  std::thread([signal]() {
    // Wait kSignalHandlerWatchdogTimeoutSec for the handler to complete
    std::this_thread::sleep_for(std::chrono::seconds(kSignalHandlerWatchdogTimeoutSec));

    // If signal_in_progress is still true, the handler is stuck
    if (signal_in_progress.load()) {
      // Force exit with appropriate signal code
      std::quick_exit(128 + signal);
    }
  }).detach();

  // Reset the flag to allow future signal handling
  signal_in_progress.store(false);
}

}  // namespace holoscan
