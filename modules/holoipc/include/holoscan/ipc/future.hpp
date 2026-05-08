/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use it except in compliance with the License.
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

/**
 * @file future.hpp
 *
 * Future<T> and Promise<T>, similar to std::future and std::promise.
 * Behavior matches std::future / std::promise where the APIs overlap.
 * then() is an extension; semantics follow std::experimental::future::then (Concurrency TS).
 *
 * Comparison with std::promise / std::future:
 * - Promise: default ctor, get_future() once, set_value(T), set_exception(exception_ptr),
 * move-only. Same as std::promise. No set_value() overload for void (no Promise<void>
 * specialization).
 * - Future: get(), wait(), wait_for(), wait_until(), valid(), move-only. get() invalidates *this.
 *   Same as std::future. No shared_future (no copy; only one get() per result).
 * - Error codes: no_state, future_already_retrieved, promise_already_satisfied, broken_promise.
 *   Same conditions and std::future_error as std.
 * - Broken promise: if the promise is destroyed without set_value/set_exception, the future's get()
 *   receives std::future_error(broken_promise). Same as std.
 * - then(): extension; returns Future<R> that shares this future's state; continuation runs on
 *   the thread that calls get() on the returned future (lazy).
 */

#ifndef HOLOSCAN_IPC_FUTURE_HPP
#define HOLOSCAN_IPC_FUTURE_HPP

#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

namespace holoscan {
namespace ipc {

template <typename T>
class Future;

template <typename T>
class Promise;

namespace detail {

/**
 * Type-erased base for shared state. Holds mutex, condition variable,
 * get_future() guard, and is_ready() so wait() can block until result is set.
 * FutureState<T> adds result_ (value or exception); ready is derived from result_.
 */
struct FutureStateBase {
  virtual ~FutureStateBase() = default;
  /** True when result holds a value or exception (not monostate). */
  virtual bool is_ready() const = 0;

  mutable std::mutex mutex_;
  mutable std::condition_variable cv_;
  bool future_retrieved_ = false;
};

/**
 * Shared state for one result: sync primitives (from base) plus result_ (value or exception).
 * By design only one is set. Promise sets via set_value/set_exception; Future retrieves via get().
 * is_ready() is true when result_ is not monostate.
 */
template <typename T>
struct FutureState final : public detail::FutureStateBase {
  bool is_ready() const override { return !std::holds_alternative<std::monostate>(result_); }
  std::variant<std::monostate, T, std::exception_ptr> result_;
};

}  // namespace detail

/**
 * Producer side: creates shared state, get_future() once, then set_value() or set_exception().
 * Move-only. If destroyed without being satisfied, the associated future gets broken_promise.
 */
template <typename T>
class Promise {
 public:
  Promise() = default;
  Promise(const Promise&) = delete;
  Promise& operator=(const Promise&) = delete;
  Promise(Promise&&) = default;
  Promise& operator=(Promise&&) = default;
  ~Promise() {
    if (!state_) {
      return;
    }
    std::lock_guard<std::mutex> lock(state_->mutex_);
    if (state_->is_ready()) {
      return;
    }
    state_->result_ = std::make_exception_ptr(std::future_error(std::future_errc::broken_promise));
    state_->cv_.notify_all();
  }

  /**
   * Returns a Future that shares this promise's state. May only be called once; second call
   * throws std::future_error with future_already_retrieved.
   */
  Future<T> get_future() {
    if (!state_) {
      throw std::future_error(std::future_errc::no_state);
    }
    std::lock_guard<std::mutex> lock(state_->mutex_);
    if (state_->future_retrieved_) {
      throw std::future_error(std::future_errc::future_already_retrieved);
    }
    state_->future_retrieved_ = true;
    return Future<T>(state_);
  }

  /**
   * Completes the promise with a value. Unblocks waiters; the associated future's get() will
   * return it. Throws std::future_error with promise_already_satisfied if already set.
   */
  void set_value(T value) {
    if (!state_) {
      throw std::future_error(std::future_errc::no_state);
    }
    std::lock_guard<std::mutex> lock(state_->mutex_);
    if (state_->is_ready()) {
      throw std::future_error(std::future_errc::promise_already_satisfied);
    }
    state_->result_ = std::move(value);
    state_->cv_.notify_all();
  }

  /**
   * Completes the promise with an exception. Unblocks waiters; the associated future's get()
   * will rethrow it. Throws std::future_error with promise_already_satisfied if already set.
   */
  void set_exception(std::exception_ptr e) {
    if (!state_) {
      throw std::future_error(std::future_errc::no_state);
    }
    std::lock_guard<std::mutex> lock(state_->mutex_);
    if (state_->is_ready()) {
      throw std::future_error(std::future_errc::promise_already_satisfied);
    }
    state_->result_ = std::move(e);
    state_->cv_.notify_all();
  }

 private:
  std::shared_ptr<detail::FutureState<T>> state_ = std::make_shared<detail::FutureState<T>>();
};

/**
 * Consumer side: holds type-erased shared state and a getter. get() blocks until ready, then
 * returns the value (or rethrows). wait() / wait_for() / wait_until() match std::future. then()
 * chains a continuation and returns a Future<R> that shares the same state for synchronization.
 */
template <typename T>
class Future {
  template <typename U>
  friend class Future;  // then() constructs Future<R>
  template <typename U>
  friend class Promise;  // get_future() constructs Future<T>(state_)

 public:
  Future() = default;

  Future(const Future&) = delete;
  Future& operator=(const Future&) = delete;
  Future(Future&&) = default;
  Future& operator=(Future&&) = default;

  /**
   * Blocks until ready, then returns the stored value (moved out) or rethrows the stored
   * exception.
   * Requires valid() is true; otherwise undefined behavior (matches std::future).
   * After this call, valid() is false whether get() returns or throws (single use).
   */
  T get() {
    wait();
    std::exception_ptr e;
    std::optional<T> result;
    // Capture value or exception so we can invalidate the future once, then return or rethrow.
    try {
      result = getter_();
    } catch (...) {
      e = std::current_exception();
    }
    // Invalidate future (one place for both success and failure).
    state_.reset();
    getter_ = {};
    if (e) {
      std::rethrow_exception(e);
    }

    return std::move(*result);
  }

  /**
   * Blocks until the result is ready. Only valid when valid() is true; otherwise undefined
   * behavior (matches std::future).
   */
  void wait() const {
    std::unique_lock<std::mutex> lock(state_->mutex_);
    state_->cv_.wait(lock, [this]() { return state_->is_ready(); });
  }

  /**
   * Blocks until ready or the timeout expires. Returns future_status::ready or
   * future_status::timeout. Only valid when valid() is true; otherwise undefined behavior
   * (matches std::future).
   */
  template <typename Rep, typename Period>
  std::future_status wait_for(const std::chrono::duration<Rep, Period>& rel_time) const {
    std::unique_lock<std::mutex> lock(state_->mutex_);
    const bool ready =
        state_->cv_.wait_for(lock, rel_time, [this]() { return state_->is_ready(); });
    return ready ? std::future_status::ready : std::future_status::timeout;
  }

  /**
   * Blocks until ready or the time point is reached. Returns future_status::ready or
   * future_status::timeout. Only valid when valid() is true; otherwise undefined behavior
   * (matches std::future).
   */
  template <typename Clock, typename Duration>
  std::future_status wait_until(const std::chrono::time_point<Clock, Duration>& abs_time) const {
    std::unique_lock<std::mutex> lock(state_->mutex_);
    const bool ready =
        state_->cv_.wait_until(lock, abs_time, [this]() { return state_->is_ready(); });
    return ready ? std::future_status::ready : std::future_status::timeout;
  }

  /** Returns true if this object holds shared state. */
  bool valid() const noexcept { return state_ != nullptr; }

  /**
   * Chains a continuation. Returns a Future<R> that shares this future's state (so wait() blocks
   * on the same condition); its get() runs \a f on the result.
   * The continuation runs on the thread that calls get().
   * The behavior is undefined if *this has no associated shared state (i.e., valid() == false).
   * (Like with most of the std::future functions.)
   * After the call, *this is invalid (state and getter moved to the returned future).
   * Extension (Concurrency TS std::experimental::future::then).
   */
  template <typename F>
  auto then(F&& f) -> Future<std::invoke_result_t<F, T>> {
    using R = std::invoke_result_t<F, T>;
    return Future<R>(std::move(state_), [getter = std::move(getter_), f = std::forward<F>(f)]() {
      return f(getter());
    });
  }

 private:
  /** Used by Promise::get_future(). Not for direct use. */
  explicit Future(std::shared_ptr<detail::FutureState<T>> state) : state_(std::move(state)) {
    getter_ = [s = std::static_pointer_cast<detail::FutureState<T>>(state_)]() {
      std::lock_guard<std::mutex> lock(s->mutex_);
      if (std::holds_alternative<std::exception_ptr>(s->result_)) {
        std::rethrow_exception(std::get<std::exception_ptr>(s->result_));
      }
      return std::move(std::get<T>(s->result_));
    };
  }

  /** Used by then() to build the continuation future; state is type-erased, getter carries T. */
  Future(std::shared_ptr<detail::FutureStateBase> state, std::function<T()> getter)
      : state_(std::move(state)), getter_(std::move(getter)) {}

  std::shared_ptr<detail::FutureStateBase> state_;
  std::function<T()> getter_;
};

}  // namespace ipc
}  // namespace holoscan

#endif  // HOLOSCAN_IPC_FUTURE_HPP
