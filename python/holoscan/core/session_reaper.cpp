/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "session_reaper.hpp"

#include <atomic>
#include <chrono>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#include <pthread.h>

#include <holoscan/logger/logger.hpp>

#include "../../../src/runtime/runtime_callback_context.hpp"

namespace holoscan::python::detail {
namespace {

std::mutex& reaper_instance_mutex() {
  static auto* mutex = new std::mutex;
  return *mutex;
}

std::atomic<RunSessionReaper*>& reaper_instance() {
  static auto* instance = new std::atomic<RunSessionReaper*>{nullptr};
  return *instance;
}

std::atomic<RunSessionReaper::RetirementNotifier>& retirement_notifier() {
  static auto* notifier = new std::atomic<RunSessionReaper::RetirementNotifier>{nullptr};
  return *notifier;
}

void prepare_reaper_fork() noexcept {
  reaper_instance_mutex().lock();
}

void finish_reaper_fork_parent() noexcept {
  reaper_instance_mutex().unlock();
}

void finish_reaper_fork_child() noexcept {
  // The inherited detached worker no longer exists. Intentionally abandon its process-private
  // state; a child-side enqueue creates a fresh reaper.
  reaper_instance().store(nullptr, std::memory_order_release);
  reaper_instance_mutex().unlock();
}

void log_reaper_failure(const char* message) noexcept {
  try {
    HOLOSCAN_LOG_ERROR("{}", message);
  } catch (...) {  // NOLINT(bugprone-empty-catch)
    // Logging is subordinate to preserving noexcept cleanup.
  }
}

}  // namespace

RunSessionReaper& RunSessionReaper::instance() {
  static const bool fork_handlers_registered = [] {
    if (pthread_atfork(
            &prepare_reaper_fork, &finish_reaper_fork_parent, &finish_reaper_fork_child) != 0) {
      log_reaper_failure("PYTHON_RUN_SESSION_REAPER_ATFORK_REGISTRATION_FAILED");
    }
    return true;
  }();
  static_cast<void>(fork_handlers_registered);

  const std::lock_guard lock(reaper_instance_mutex());
  RunSessionReaper* reaper = reaper_instance().load(std::memory_order_acquire);
  if (reaper == nullptr) {
    reaper = new RunSessionReaper{};
    reaper_instance().store(reaper, std::memory_order_release);
  }
  return *reaper;
}

RunSessionReaper::RunSessionReaper() {
  std::thread worker{[this] { worker_loop(); }};
  worker.detach();
}

void RunSessionReaper::enqueue(std::unique_ptr<Retirement> node) noexcept {
  if (!node) {
    return;
  }
  bool queued{};
  try {
    const std::lock_guard lock(mutex_);
    queue_.push_back(std::move(node));
    ++pending_;
    queued = true;
  } catch (...) {
    // Destroying a session whose join already failed would retry the fatal strict-RAII path.
    // Preserve process liveness by intentionally abandoning this process-lifetime fallback node.
    // NOLINTNEXTLINE(bugprone-unused-return-value)
    static_cast<void>(node.release());
    log_reaper_failure("PYTHON_RUN_SESSION_REAPER_ENQUEUE_FAILED");
  }
  if (queued) {
    ready_.notify_one();
  }
}

bool RunSessionReaper::wait_for_existing_for(std::chrono::nanoseconds timeout) noexcept {
  RunSessionReaper* reaper = reaper_instance().load(std::memory_order_acquire);
  if (reaper == nullptr) {
    return true;
  }
  try {
    if (!reaper->wait_until_drained_for(timeout)) {
      log_reaper_failure("PYTHON_RUN_SESSION_REAPER_EXIT_DRAIN_TIMEOUT");
      return false;
    }
    return true;
  } catch (...) {
    log_reaper_failure("PYTHON_RUN_SESSION_REAPER_EXIT_DRAIN_FAILED");
    return false;
  }
}

void RunSessionReaper::destroy_retired() noexcept {
  // Even a joined RunSession must be destroyed outside Runtime callbacks. Leave the
  // complete retired owner queued until an ordinary host thread drains it.
  if (holoscan::detail::runtime_callback_context_restricted()) {
    return;
  }
  RunSessionReaper* reaper = reaper_instance().load(std::memory_order_acquire);
  if (reaper == nullptr) {
    return;
  }
  std::deque<std::unique_ptr<Retirement>> retired;
  try {
    const std::lock_guard lock(reaper->mutex_);
    retired.swap(reaper->retired_);
  } catch (...) {
    log_reaper_failure("PYTHON_RUN_SESSION_REAPER_RETIREMENT_DRAIN_FAILED");
  }
}

void RunSessionReaper::set_retirement_notifier(RetirementNotifier notifier) noexcept {
  RunSessionReaper* reaper = reaper_instance().load(std::memory_order_acquire);
  if (reaper != nullptr) {
    const std::lock_guard lock(reaper->mutex_);
    retirement_notifier().store(notifier, std::memory_order_release);
    return;
  }
  retirement_notifier().store(notifier, std::memory_order_release);
}

bool RunSessionReaper::wait_until_idle_for(std::chrono::nanoseconds timeout) {
  std::unique_lock lock(mutex_);
  return idle_.wait_for(lock, timeout, [this] { return pending_ == 0U; });
}

bool RunSessionReaper::wait_until_drained_for(std::chrono::nanoseconds timeout) {
  std::unique_lock lock(mutex_);
  return idle_.wait_for(lock, timeout, [this] {
    return pending_ == 0U && retirement_notifications_in_flight_ == 0U;
  });
}

void RunSessionReaper::worker_loop() noexcept {
  while (true) {
    std::unique_ptr<Retirement> node;
    {
      std::unique_lock lock(mutex_);
      ready_.wait(lock, [this] { return !queue_.empty(); });
      node = std::move(queue_.front());
      queue_.pop_front();
    }

    node->request_stop();
    bool joined{};
    try {
      node->join();
      joined = true;
    } catch (const std::exception& error) {
      try {
        HOLOSCAN_LOG_ERROR("PYTHON_RUN_SESSION_REAPER_WAIT_FAILED: {}", error.what());
      } catch (...) {  // NOLINT(bugprone-empty-catch)
        // Logging is subordinate to preserving noexcept cleanup.
      }
    } catch (...) {
      log_reaper_failure("PYTHON_RUN_SESSION_REAPER_WAIT_FAILED");
    }
    if (!joined) {
      // RunSession destruction retries wait() and terminates if it fails again. The only safe
      // recovery after a platform join failure is to abandon the process-lifetime cleanup node.
      // NOLINTNEXTLINE(bugprone-unused-return-value)
      static_cast<void>(node.release());
    }

    RetirementNotifier notifier{};
    {
      const std::lock_guard lock(mutex_);
      if (node) {
        try {
          retired_.push_back(std::move(node));
          notifier = retirement_notifier().load(std::memory_order_acquire);
          if (notifier != nullptr) {
            ++retirement_notifications_in_flight_;
          }
        } catch (...) {
          // The session is joined, but its hosted-language objects cannot be reclaimed safely on
          // this worker. Preserve process liveness by abandoning the fallback node.
          [[maybe_unused]] auto* const abandoned_node = node.release();
          log_reaper_failure("PYTHON_RUN_SESSION_REAPER_RETIREMENT_QUEUE_FAILED");
        }
      }
      --pending_;
    }
    idle_.notify_all();
    if (notifier != nullptr) {
      notifier();
      {
        const std::lock_guard lock(mutex_);
        --retirement_notifications_in_flight_;
      }
      idle_.notify_all();
    }
  }
}

}  // namespace holoscan::python::detail
