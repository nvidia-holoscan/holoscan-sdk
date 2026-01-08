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

#ifndef HOLOSCAN_TESTS_SYSTEM_ASYNC_DATA_LOGGER_TEST_HELPERS_HPP
#define HOLOSCAN_TESTS_SYSTEM_ASYNC_DATA_LOGGER_TEST_HELPERS_HPP

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <thread>

#include <holoscan/holoscan.hpp>
#include "holoscan/core/resources/async_data_logger.hpp"

namespace holoscan {

/**
 * @brief A slow backend that introduces configurable delays when processing entries.
 *
 * This backend is used to test the shutdown timeout behavior of AsyncDataLoggerResource.
 * By introducing delays, we can ensure that entries remain in the queue during shutdown,
 * allowing us to verify that the timeout mechanism works correctly.
 */
class SlowTestBackend : public AsyncDataLoggerBackend {
 public:
  explicit SlowTestBackend(std::chrono::milliseconds process_delay = std::chrono::milliseconds(100))
      : process_delay_(process_delay) {}

  bool initialize() override {
    initialized_.store(true);
    entries_processed_.store(0);
    large_entries_processed_.store(0);
    return true;
  }

  void shutdown() override {
    initialized_.store(false);
    HOLOSCAN_LOG_INFO(
        "SlowTestBackend shutdown - processed {} data entries and {} large data entries",
        entries_processed_.load(),
        large_entries_processed_.load());
  }

  bool process_data_entry(const DataEntry& entry) override {
    if (!initialized_.load()) {
      return false;
    }
    // Simulate slow processing
    std::this_thread::sleep_for(process_delay_);
    entries_processed_.fetch_add(1, std::memory_order_relaxed);
    HOLOSCAN_LOG_DEBUG("SlowTestBackend processed data entry: {}", entry.unique_id);
    return true;
  }

  bool process_large_data_entry(const DataEntry& entry) override {
    if (!initialized_.load()) {
      return false;
    }
    // Simulate slow processing
    std::this_thread::sleep_for(process_delay_);
    large_entries_processed_.fetch_add(1, std::memory_order_relaxed);
    HOLOSCAN_LOG_DEBUG("SlowTestBackend processed large data entry: {}", entry.unique_id);
    return true;
  }

  std::string get_statistics() const override {
    return fmt::format("SlowTestBackend - processed: {}, large processed: {}",
                       entries_processed_.load(),
                       large_entries_processed_.load());
  }

  size_t get_entries_processed() const { return entries_processed_.load(); }
  size_t get_large_entries_processed() const { return large_entries_processed_.load(); }

  void set_process_delay(std::chrono::milliseconds delay) { process_delay_ = delay; }

 private:
  std::chrono::milliseconds process_delay_;
  std::atomic<bool> initialized_{false};
  std::atomic<size_t> entries_processed_{0};
  std::atomic<size_t> large_entries_processed_{0};
};

/**
 * @brief An async data logger with a slow backend for testing shutdown timeout behavior.
 */
class SlowAsyncLogger : public AsyncDataLoggerResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(SlowAsyncLogger, AsyncDataLoggerResource)
  SlowAsyncLogger() = default;

  void setup(ComponentSpec& spec) override { AsyncDataLoggerResource::setup(spec); }

  void initialize() override {
    // Create the slow backend with configurable delay
    auto backend = std::make_shared<SlowTestBackend>(process_delay_);
    slow_backend_ = backend;

    // Set the backend BEFORE calling parent initialize (which starts worker threads)
    set_backend(backend);

    // Call parent initialize to start the worker threads
    AsyncDataLoggerResource::initialize();
    HOLOSCAN_LOG_INFO("SlowAsyncLogger initialized with {}ms processing delay",
                      process_delay_.count());
  }

  void set_process_delay(std::chrono::milliseconds delay) {
    process_delay_ = delay;
    if (slow_backend_) {
      slow_backend_->set_process_delay(delay);
    }
  }

  size_t get_entries_processed() const {
    return slow_backend_ ? slow_backend_->get_entries_processed() : 0;
  }

  size_t get_large_entries_processed() const {
    return slow_backend_ ? slow_backend_->get_large_entries_processed() : 0;
  }

 private:
  std::chrono::milliseconds process_delay_{100};  // 100ms default delay per entry
  std::shared_ptr<SlowTestBackend> slow_backend_;
};

}  // namespace holoscan

#endif  // HOLOSCAN_TESTS_SYSTEM_ASYNC_DATA_LOGGER_TEST_HELPERS_HPP
