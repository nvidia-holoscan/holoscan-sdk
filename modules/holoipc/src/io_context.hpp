/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_IPC_IOCONTEXT_HPP
#define HOLOSCAN_IPC_IOCONTEXT_HPP

#include <sys/epoll.h>
#include <sys/eventfd.h>
#ifdef HOLOSCAN_IPC_IOCONTEXT_TIMER_SUPPORT
#include <sys/timerfd.h>
#endif
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace holoscan {
namespace ipc {

/**
 * @brief RAII wrapper for a Linux epoll instance.
 *
 * Calls epoll_create1(flags) in the constructor and close() in the destructor.
 * Non-copyable, non-movable. Use get() to obtain the fd for epoll_ctl() and epoll_wait().
 * Linux only.
 */
class Epoll {
 public:
  /** @brief Create an epoll instance. @param flags Passed to epoll_create1() (e.g.
   * EPOLL_CLOEXEC). */
  explicit Epoll(int flags);
  ~Epoll();

  Epoll(const Epoll&) = delete;
  Epoll& operator=(const Epoll&) = delete;
  Epoll(Epoll&&) = delete;
  Epoll& operator=(Epoll&&) = delete;

  /** @brief Return the epoll file descriptor. */
  int get() const { return fd_; }

  /**
   * @brief Wrapper for epoll_ctl(2).
   * @param op EPOLL_CTL_ADD, EPOLL_CTL_MOD, or EPOLL_CTL_DEL.
   * @param fd The target file descriptor.
   * @param event Pointer to epoll_event; may be null for EPOLL_CTL_DEL.
   * @return 0 on success, -1 on error (see errno).
   */
  int ctl(int op, int fd, struct epoll_event* event) const;

  /**
   * @brief Wrapper for epoll_wait(2).
   * @param events Buffer to store ready events.
   * @param maxevents Maximum number of events to return (size of \a events).
   * @param timeout_ms Timeout in milliseconds; -1 = block indefinitely, 0 = return immediately.
   * @return Number of ready file descriptors on success; 0 on timeout; -1 on error (see errno).
   */
  int wait(struct epoll_event* events, int maxevents, int timeout_ms) const;

 private:
  int fd_{-1};
};

/**
 * @brief RAII wrapper for a Linux eventfd.
 *
 * Calls eventfd(initval, flags) in the constructor and close() in the destructor.
 * Non-copyable, non-movable. Use get() for epoll registration and read(); use write(buf, count)
 * to signal.
 * Linux only.
 */
class Event {
 public:
  /** @brief Create an eventfd. @param initval Initial counter value. @param flags Passed to
   * eventfd() (e.g. EFD_CLOEXEC | EFD_NONBLOCK). */
  explicit Event(unsigned int initval, int flags);
  ~Event();

  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  Event(Event&&) = delete;
  Event& operator=(Event&&) = delete;

  /** @brief Return the eventfd file descriptor. */
  int get() const { return fd_; }

  /** @brief Write count bytes from buf to the eventfd. Returns bytes written on success, -1 on
   * error (see errno). Thread-safe. */
  ssize_t write(const void* buf, size_t count);

 private:
  int fd_{-1};
};

#ifdef HOLOSCAN_IPC_IOCONTEXT_TIMER_SUPPORT
/**
 * @brief RAII wrapper for a Linux timerfd.
 *
 * Calls timerfd_create(clockid, flags) in the constructor and close() in the destructor.
 * Non-copyable, non-movable. Use get() for epoll registration; use set_oneshot() to arm a one-shot
 * timer.
 * When the timer expires the fd becomes readable; read() to drain. Linux only.
 */
class Timer {
 public:
  /** @brief Create a timerfd. @param clockid e.g. CLOCK_MONOTONIC. @param flags Passed to
   * timerfd_create() (e.g. TFD_CLOEXEC | TFD_NONBLOCK). */
  explicit Timer(int clockid, int flags);
  ~Timer();

  Timer(const Timer&) = delete;
  Timer& operator=(const Timer&) = delete;
  Timer(Timer&&) = delete;
  Timer& operator=(Timer&&) = delete;

  /** @brief Return the timerfd file descriptor. */
  int get() const { return fd_; }

  /**
   * @brief Arm a one-shot timer to expire after \a delay.
   * @param delay Time until expiration.
   * @return 0 on success, -1 on error (see errno).
   */
  int set_oneshot(std::chrono::nanoseconds delay);

 private:
  int fd_{-1};
};
#endif  // HOLOSCAN_IPC_IOCONTEXT_TIMER_SUPPORT

/**
 * @brief Minimal event loop similar to boost::asio::io_context: run a single thread that processes
 * posted work.
 *
 * Work is posted via post(). A single thread calls run(), which blocks in epoll_wait until work is
 * available, stop() is called, or epoll_wait fails. Waking uses eventfd (no condition variable).
 * stop() clears running_; before run() returns, all work already in the queue (and work posted
 * while draining) is executed so posted control tasks are not silently dropped. A second run()
 * after stop() skips the epoll loop (running_ is not reset) but still drains any queued work then
 * returns. Linux only.
 */
class IOContext {
 public:
  IOContext();
  ~IOContext();

  IOContext(const IOContext&) = delete;
  IOContext& operator=(const IOContext&) = delete;
  IOContext(IOContext&&) = delete;
  IOContext& operator=(IOContext&&) = delete;

  /**
   * @brief Enqueue work and wake the run() loop. Thread-safe.
   *
   * @param f A valid nullary callable with void return (invocable and, for std::function,
   *          non-empty). Move-only callables are supported.
   */
  template <typename F>
  void post(F&& f) {
    static_assert(std::is_invocable_r_v<void, std::decay_t<F>&>,
                  "IOContext::post requires a callable void()");
    {
      std::lock_guard<std::mutex> lock(mutex_);
      work_queue_.push(make_posted_work(std::forward<F>(f)));
    }
    wake();
  }

#ifdef HOLOSCAN_IPC_IOCONTEXT_TIMER_SUPPORT
  /** @brief Run \a f once after \a delay. Callback runs on the thread that called run(). Safe to
   * call from any thread. */
  void schedule_after(std::chrono::nanoseconds delay, std::function<void()> f);
#endif

  /** @brief Run the loop until stop() is called. Blocks. Typically called from one thread. */
  void run();

  /** @brief Request stop and wake run(). Thread-safe. */
  void stop();

 private:
  /** @brief Signal the eventfd (write one) so that run() wakes from epoll_wait. Used by post()
   * and stop(). Thread-safe. */
  void wake();

  /** @brief Run every queued PostedWork until work_queue_ is empty (handles nested post). Caller
   * must not hold mutex_. */
  void drain_posted_work_until_empty();

  struct PostedWork {
    virtual ~PostedWork() = default;
    virtual void run() = 0;
  };

  template <typename F>
  struct PostedWorkImpl final : PostedWork {
    explicit PostedWorkImpl(F fn) : f_(std::move(fn)) {}
    void run() override { f_(); }

   private:
    F f_;
  };

  template <typename F>
  static std::unique_ptr<PostedWork> make_posted_work(F&& f) {
    return std::make_unique<PostedWorkImpl<std::decay_t<F>>>(std::forward<F>(f));
  }

  /** Swallows exceptions so posted/timer work cannot terminate the IO thread. */
  void run_posted_job_noexcept(PostedWork& job) noexcept;

  Epoll epoll_;
  Event wake_;
  std::atomic<bool> running_{true};
  std::queue<std::unique_ptr<PostedWork>> work_queue_;
#ifdef HOLOSCAN_IPC_IOCONTEXT_TIMER_SUPPORT
  std::unordered_map<int, std::pair<std::unique_ptr<Timer>, std::function<void()>>> timers_;
#endif
  std::mutex mutex_;
};

}  // namespace ipc
}  // namespace holoscan

#endif  // HOLOSCAN_IPC_IOCONTEXT_HPP
