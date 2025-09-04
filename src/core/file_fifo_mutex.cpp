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

#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "holoscan/core/file_fifo_mutex.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

/**
 * @brief Removes a PID from the top of a queue file. The top of a queue file is
 * the first line in the file.
 *
 * @param queue_fd
 * @param pid
 */
inline void pop_pid_from_queue(int queue_fd, pid_t pid) {
  if (queue_fd < 0 || pid < 0) {
    HOLOSCAN_LOG_DEBUG(
        fmt::format("Invalid queue file descriptor ({}) or PID ({})", queue_fd, pid));
    return;
  }

  // Lock the queue file for modification with exclusive access in blocking
  // mode because removing our PID is essential to be done atomically.
  ScopedFlock scoped_queue_lock(queue_fd, LOCK_EX, LOCK_UN);

  // Check the queue file size to see if we can read the file in a fixed-sized buffer
  struct stat st;
  int ret = fstat(queue_fd, &st);
  if (ret < 0) {
    throw std::runtime_error(fmt::format("Failed to get queue file size: {}", strerror(errno)));
  }
  size_t file_size = st.st_size;
  constexpr size_t MAX_QUEUE_FILESIZE = 4096;

  if (file_size > MAX_QUEUE_FILESIZE) {
    throw std::runtime_error("Too many processes in the queue. Queue file is too large to read");
  }

  lseek(queue_fd, 0, SEEK_SET);
  char buffer[MAX_QUEUE_FILESIZE + 1];
  ssize_t bytes_read = read(queue_fd, buffer, sizeof(buffer) - 1);

  if (bytes_read == 0) {
    HOLOSCAN_LOG_ERROR("EOF reached while reading queue file.");
    return;
  } else if (bytes_read < 0) {
    throw std::runtime_error(fmt::format("Failed to read queue file: {}", strerror(errno)));
  } else {
    buffer[bytes_read] = '\0';

    // Find the first newline to skip our PID
    char* rest = strchr(buffer, '\n');
    if (rest) {
      // Truncate the file
      lseek(queue_fd, 0, SEEK_SET);
      if (*(rest + 1) != '\0') {  // If there's more data after our PID
        int ret = write(queue_fd, rest + 1, buffer + bytes_read - (rest + 1));
        if (ret < 0) {
          throw std::runtime_error(
              fmt::format("Failed to write to queue file: {}", strerror(errno)));
        }
      }
      int ret = ftruncate(queue_fd, bytes_read - (rest - buffer + 1));
      if (ret < 0) {
        throw std::runtime_error(fmt::format("Failed to truncate queue file: {}", strerror(errno)));
      }
    } else {
      HOLOSCAN_LOG_ERROR("Failed find PID {} in the queue file", pid);
    }
  }
}

ScopedFlock::ScopedFlock(int fd, int lock_type, int unlock_type) {
  fd_ = fd;
  unlock_type_ = unlock_type;
  if (fd_ != -1) {
    int ret = flock(fd_, lock_type);
    if (ret < 0) {
      locked_ = false;
      return;
    }
    locked_ = true;
  }
}

ScopedFlock::~ScopedFlock() {
  if (fd_ != -1 && locked_) {
    int ret = flock(fd_, unlock_type_);
    if (ret < 0) {
      // This is a severe error as we have locked a file descriptor but failed
      // to unlock it.
      try {
        HOLOSCAN_LOG_ERROR("FATAL: Failed to unlock file descriptor: {}", strerror(errno));
      } catch (...) {
      }
    }
  }
}

ScopedWaitedFlock::ScopedWaitedFlock(int fd, int lock_type, int unlock_type, int wait_time_ms) {
  if (wait_time_ms > 0) {
    int non_blocking_lock_type = lock_type | LOCK_NB;
    int current_wait = wait_time_ms;
    do {
      std::unique_ptr local_scoped_lock_ =
          std::make_unique<ScopedFlock>(fd, non_blocking_lock_type, unlock_type);
      if (local_scoped_lock_->locked()) {
        scoped_lock_ = std::move(local_scoped_lock_);
        return;
      }
      usleep(2000);  // wait for 2 ms; 2 ms granularity wait to reduce CPU usage
      current_wait -= 2;
    } while (current_wait > 0);
    if (current_wait <= 0 && !scoped_lock_) {
      HOLOSCAN_LOG_DEBUG(
          "Failed to lock file descriptor: {} after waiting for {} ms", fd, wait_time_ms);
    }
  } else {  // act as normal ScopedFlock i.e., blocking lock
    scoped_lock_ = std::make_unique<ScopedFlock>(fd, lock_type, unlock_type);
  }
}

FileFIFOMutex::FileFIFOMutex(std::string file_path) {
  if (file_path.empty()) {
    throw std::invalid_argument("The file path cannot be empty.");
  }

  // Main lock file
  fd_ = open(file_path.c_str(), O_CREAT | O_RDWR, 0666);
  if (fd_ == -1) {
    throw std::invalid_argument(fmt::format("Failed to open/create lock file: {}", file_path));
  }

  // Queue file to maintain FIFO order
  std::string queue_path = std::string(file_path) + ".queue";
  queue_fd_ = open(queue_path.c_str(), O_CREAT | O_RDWR, 0666);
  if (queue_fd_ == -1) {
    throw std::invalid_argument(fmt::format("Failed to open/create queue file: {}", queue_path));
  }

  // Store our PID for the queue
  pid_ = getpid();
}

FileFIFOMutex::~FileFIFOMutex() {
  try {
    unlock();
  } catch (const std::exception& e) {
    // Silently handle any exceptions during cleanup
    try {
      HOLOSCAN_LOG_ERROR("FileFIFOMutex destructor failed with {}", e.what());
    } catch (...) {
    }
  }
  if (fd_ != -1) {
    close(fd_);
  }
  if (queue_fd_ != -1) {
    close(queue_fd_);
  }
}

void FileFIFOMutex::set_wait_time_ms(int wait_time_ms) {
  wait_time_ms_ = wait_time_ms;
}

void FileFIFOMutex::lock() {
  if (fd_ == -1 || queue_fd_ == -1) {
    return;
  }

  size_t file_size = 0;
  {
    ScopedWaitedFlock queue_lock(queue_fd_, LOCK_EX, LOCK_UN, wait_time_ms_);
    if (!queue_lock.locked()) {
      HOLOSCAN_LOG_DEBUG("Failed to lock queue file after waiting for {} ms", wait_time_ms_);
      return;
    }

    struct stat st;
    int ret = fstat(queue_fd_, &st);
    if (ret < 0) {
      throw std::runtime_error(fmt::format("Failed to get queue file size: {}", strerror(errno)));
    }
    file_size = st.st_size;
    // Add our PID to the queue
    lseek(queue_fd_, 0, SEEK_END);
    std::string pid_str = std::to_string(pid_) + "\n";
    ret = write(queue_fd_, pid_str.c_str(), pid_str.length());
    if (ret < 0) {
      throw std::runtime_error(fmt::format("Failed to write to queue file: {}", strerror(errno)));
    }
  }

  // Now wait for our turn (FIFO order)
  bool found_pid = false;
  while (file_size > 0) {
    // Check if we're at the front of the queue
    lseek(queue_fd_, 0, SEEK_SET);
    char buffer[32];
    ssize_t bytes_read = 0;
    {
      // Make sure we are reading while others are also reading with a shared
      // lock and not when others are modifying the queue file
      ScopedFlock queue_lock(queue_fd_, LOCK_SH, LOCK_UN);
      bytes_read = read(queue_fd_, buffer, sizeof(buffer) - 1);
    }

    if (bytes_read > 0) {
      // Null-terminate and parse first PID in queue
      buffer[bytes_read] = '\0';
      pid_t first_pid = strtol(buffer, nullptr, 10);

      // If we're first in line, then flag that we found our PID and break
      if (first_pid == pid_) {
        found_pid = true;
        break;
      }
    }

    // Not our turn yet, wait a bit and check again
    usleep(1000);
  }

  // file size was 0 implies: nothing in the queue before we wrote our PID to
  // the queue, so we are first in line implicitly
  // Otherwise, we found our PID in the queue, so we are next in line
  if (file_size == 0 || found_pid) {
    main_lock_ = std::make_unique<ScopedFlock>(fd_, LOCK_EX, LOCK_UN);
    locked_ = true;
  }
}

void FileFIFOMutex::unlock() {
  // Only unlock if we actually have the lock
  if (locked_) {
    // First remove our PID from the front of the queue
    if (queue_fd_ != -1) {
      pop_pid_from_queue(queue_fd_, pid_);
    }

    // Now release the main lock after queue is updated
    main_lock_.reset();  // early cleanup
    locked_ = false;
  }
}

bool FileFIFOMutex::locked() const {
  return locked_;
}

}  // namespace holoscan
