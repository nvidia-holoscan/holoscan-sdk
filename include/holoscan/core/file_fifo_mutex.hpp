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

#ifndef HOLOSCAN_CORE_FILE_FIFO_MUTEX_HPP
#define HOLOSCAN_CORE_FILE_FIFO_MUTEX_HPP

#include <sys/types.h>
#include <memory>
#include <string>
#include <utility>

namespace holoscan {
constexpr int DEFAULT_FIFOMUTEX_WAIT_TIME_MS = 10;

/**
 * @brief RAII wrapper class over a file descriptor lock with flock.
 *
 */
class ScopedFlock {
 public:
  // Deleting default constructors
  ScopedFlock() = delete;
  ScopedFlock(const ScopedFlock&) = delete;
  ScopedFlock& operator=(const ScopedFlock&) = delete;

  ScopedFlock(int fd, int lock_type, int unlock_type);
  ~ScopedFlock();
  bool locked() const { return locked_; }

 private:
  int fd_ = -1;
  int unlock_type_;
  bool locked_ = false;
};

/**
 * @brief RAII wrapper over a waited ScopedFlock. At the initialization, it
 * waits for the lock to be available until wait_time_ms_ ms is reached. It stops
 * trying to lock after that and does not lock the file descriptor.
 *
 * When wait time is set to zero or less, then it falls back to ScopedFlock
 *
 */
class ScopedWaitedFlock {
 public:
  // Deleting default constructors
  ScopedWaitedFlock() = delete;
  ScopedWaitedFlock(const ScopedWaitedFlock&) = delete;
  ScopedWaitedFlock& operator=(const ScopedWaitedFlock&) = delete;

  ScopedWaitedFlock(int fd, int lock_type, int unlock_type,
                    int wait_time_ms = DEFAULT_FIFOMUTEX_WAIT_TIME_MS);
  ~ScopedWaitedFlock() = default;

  bool locked() const { return scoped_lock_ && scoped_lock_->locked(); }

 private:
  std::unique_ptr<ScopedFlock> scoped_lock_;
};

/**
 * @brief A class that implements a file-based FIFO mutex.
 *
 * This class implements a file-based mutex that can be used to synchronize
 * multiple processes. It uses a lock file to synchronize the processes and
 * a queue file to store the process IDs of the processes that are waiting to
 * acquire the lock.
 *
 *
 */
class FileFIFOMutex {
 public:
  // Delete default, copy and assignment constructors
  FileFIFOMutex(const FileFIFOMutex&) = delete;
  FileFIFOMutex& operator=(const FileFIFOMutex&) = delete;
  FileFIFOMutex() = delete;
  /**
   * @brief Constructs a new file-backed mutex at the given path and a queue
   * file at `file_path.queue`, if not already present. It also opens both the files.
   *
   * @param file_path the path to the mutex file.
   */
  explicit FileFIFOMutex(std::string file_path);

  /**
   * @brief Unlocks the mutex if it was locked. Closes the queue and lock files.
   *
   */
  ~FileFIFOMutex();
  /**
   * @brief Sets the wait time for the lock. The wait time is applied
   * when the queue file is being locked. If the queue file is not available for
   * lock within the wait time, then the lock is not acquired. If the wait time
   * is less than or equal to 0, then it blocks until the lock is acquired.
   *
   * @param wait_time_ms The wait time in milliseconds.
   */
  void set_wait_time_ms(int wait_time_ms);

  /**
   * @brief Locks the mutex after writing the current process' PID to the queue.
   * Waits for the queue file to be unlocked to write its own PID to the queue,
   * depending on the wait time.
   *
   */
  void lock();
  /**
   * @brief Unlocks the mutex by removing itself from the queue and unlocking the mutex.
   *
   */
  void unlock();
  /**
   * @brief Returns true if the mutex is locked.
   *
   * @return true When the mutex is locked.
   * @return false When the mutex is not locked.
   */
  bool locked() const;

 private:
  std::unique_ptr<ScopedFlock> main_lock_;
  int fd_ = -1;
  int queue_fd_ = -1;
  pid_t pid_;
  bool locked_ = false;
  int wait_time_ms_ = DEFAULT_FIFOMUTEX_WAIT_TIME_MS;
};
}  // namespace holoscan

#endif /* HOLOSCAN_CORE_FILE_FIFO_MUTEX_HPP */
