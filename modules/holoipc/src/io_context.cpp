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

#include "io_context.hpp"

#include <sys/epoll.h>
#include <sys/eventfd.h>
#ifdef HOLOSCAN_IPC_IOCONTEXT_TIMER_SUPPORT
#include <sys/timerfd.h>
#endif
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

#include <holoscan/ipc/log.hpp>

namespace holoscan {
namespace ipc {

namespace {

constexpr int EPollMaxEvents = 8;

::epoll_event make_epoll_event(uint32_t events, int fd) {
  ::epoll_event ev = {};
  ev.events = events;
  ev.data.fd = fd;
  return ev;
}

#ifdef HOLOSCAN_IPC_IOCONTEXT_TIMER_SUPPORT
::itimerspec make_itimerspec(std::chrono::nanoseconds delay) {
  auto sec = std::chrono::duration_cast<std::chrono::seconds>(delay);
  ::itimerspec spec = {};
  spec.it_value.tv_sec = sec.count();
  spec.it_value.tv_nsec = (delay - sec).count();
  return spec;
}
#endif

}  // namespace

Epoll::Epoll(int flags) : fd_(::epoll_create1(flags)) {
  if (fd_ == -1) {
    throw std::runtime_error("[IOContext] epoll_create1: " + std::string(strerror(errno)));
  }
}

Epoll::~Epoll() {
  (void)::close(fd_);
}

int Epoll::ctl(int op, int fd, struct ::epoll_event* event) const {
  return ::epoll_ctl(fd_, op, fd, event);
}

int Epoll::wait(struct ::epoll_event* events, int maxevents, int timeout_ms) const {
  return ::epoll_wait(fd_, events, maxevents, timeout_ms);
}

Event::Event(unsigned int initval, int flags) : fd_(::eventfd(initval, flags)) {
  if (fd_ == -1) {
    throw std::runtime_error("[IOContext] eventfd: " + std::string(::strerror(errno)));
  }
}

Event::~Event() {
  (void)::close(fd_);
}

ssize_t Event::write(const void* buf, size_t count) {
  return ::write(fd_, buf, count);
}

#ifdef HOLOSCAN_IPC_IOCONTEXT_TIMER_SUPPORT
Timer::Timer(int clockid, int flags) : fd_(::timerfd_create(clockid, flags)) {
  if (fd_ == -1) {
    throw std::runtime_error("[IOContext] timerfd_create: " + std::string(::strerror(errno)));
  }
}

Timer::~Timer() {
  (void)::close(fd_);
}

int Timer::set_oneshot(std::chrono::nanoseconds delay) {
  ::itimerspec spec = make_itimerspec(delay);
  // flags=0: spec is relative (duration from now), not absolute time.
  return ::timerfd_settime(fd_, 0, &spec, nullptr);
}
#endif  // HOLOSCAN_IPC_IOCONTEXT_TIMER_SUPPORT

IOContext::IOContext() : epoll_(EPOLL_CLOEXEC), wake_(0, EFD_CLOEXEC | EFD_NONBLOCK) {
  auto ev = make_epoll_event(EPOLLIN, wake_.get());
  if (epoll_.ctl(EPOLL_CTL_ADD, wake_.get(), &ev) == -1) {
    throw std::runtime_error("[IOContext] epoll_ctl ADD: " + std::string(::strerror(errno)));
  }
}

IOContext::~IOContext() {
  stop();
}

void IOContext::run_posted_job_noexcept(PostedWork& job) noexcept {
  try {
    job.run();
  } catch (const std::exception& e) {
    HOLOSCAN_IPC_LOG_ERROR("[IOContext] posted work threw: {}", e.what());
  } catch (...) {
    HOLOSCAN_IPC_LOG_ERROR("[IOContext] posted work threw (non-std::exception)");
  }
}

void IOContext::wake() {
  uint64_t one = 1;
  const ssize_t w = wake_.write(&one, sizeof(one));
  if (w != static_cast<ssize_t>(sizeof(one))) {
    HOLOSCAN_IPC_LOG_ERROR("[IOContext] wake: eventfd write failed (w={} errno={})", w, errno);
  }
}

#ifdef HOLOSCAN_IPC_IOCONTEXT_TIMER_SUPPORT
void IOContext::schedule_after(std::chrono::nanoseconds delay, std::function<void()> f) {
  if (!f) {
    return;
  }
  auto timer = std::make_unique<Timer>(CLOCK_MONOTONIC, TFD_CLOEXEC | TFD_NONBLOCK);
  if (timer->set_oneshot(delay) == -1) {
    throw std::runtime_error("[IOContext] timerfd_settime: " + std::string(::strerror(errno)));
  }
  int fd = timer->get();
  auto ev = make_epoll_event(EPOLLIN, fd);
  std::lock_guard<std::mutex> lock(mutex_);
  if (epoll_.ctl(EPOLL_CTL_ADD, fd, &ev) == -1) {
    throw std::runtime_error("[IOContext] epoll_ctl ADD timer: " + std::string(::strerror(errno)));
  }
  timers_.emplace(fd, std::make_pair(std::move(timer), std::move(f)));
}
#endif  // HOLOSCAN_IPC_IOCONTEXT_TIMER_SUPPORT

void IOContext::drain_posted_work_until_empty() {
  while (true) {
    std::queue<std::unique_ptr<PostedWork>> batch;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (work_queue_.empty()) {
        return;
      }
      batch.swap(work_queue_);
    }
    while (!batch.empty()) {
      auto job = std::move(batch.front());
      batch.pop();
      run_posted_job_noexcept(*job);
    }
  }
}

void IOContext::run() {
  std::array<::epoll_event, EPollMaxEvents> events;
  while (running_) {
    // Block until at least one fd is ready; -1 = no timeout.
    int n = epoll_.wait(events.data(), static_cast<int>(events.size()), -1);
    if (n == -1) {
      if (errno == EINTR) {
        continue;
      }
      // epoll_wait failed (e.g. EBADF); treat like stop: drain posted work then exit.
      running_.store(false);
      continue;
    }
    for (int i = 0; i < n; ++i) {
      const int fd = events[i].data.fd;
      // Wake eventfd: post() or stop() signaled; drain and run all posted work.
      if (fd == wake_.get()) {
        uint64_t count = 0;
        auto r = ::read(wake_.get(), &count, sizeof(count));
        (void)r;
        if (!running_) {
          break;
        }
        // Take the whole queue under lock so we don't hold mutex during callbacks.
        std::queue<std::unique_ptr<PostedWork>> batch;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          batch.swap(work_queue_);
        }
        while (!batch.empty()) {
          auto job = std::move(batch.front());
          batch.pop();
          run_posted_job_noexcept(*job);
        }
        // Move to the next event in this epoll_wait batch.
        continue;
      }
#ifdef HOLOSCAN_IPC_IOCONTEXT_TIMER_SUPPORT
      // Timerfd: schedule_after() callback fired; drain, unregister, run callback.
      std::function<void()> callback;
      {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = timers_.find(fd);
        if (it == timers_.end()) {
          continue;
        }
        uint64_t expirations = 0;
        auto r = ::read(fd, &expirations, sizeof(expirations));
        (void)r;
        epoll_.ctl(EPOLL_CTL_DEL, fd, nullptr);
        callback = std::move(it->second.second);
        timers_.erase(it);
      }
      try {
        callback();
      } catch (const std::exception& e) {
        HOLOSCAN_IPC_LOG_ERROR("[IOContext] timer callback threw: {}", e.what());
      } catch (...) {
        HOLOSCAN_IPC_LOG_ERROR("[IOContext] timer callback threw (non-std::exception)");
      }
#endif
    }
  }
  drain_posted_work_until_empty();
}

void IOContext::stop() {
  running_.store(false);
  wake();
}

}  // namespace ipc
}  // namespace holoscan
