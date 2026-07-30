/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOSCAN_CORE_RESOURCES_DATA_LOGGER_QUEUE_HPP
#define HOLOSCAN_CORE_RESOURCES_DATA_LOGGER_QUEUE_HPP

#include <algorithm>
#include <cctype>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

#include "concurrentqueue.h"
#include "yaml-cpp/yaml.h"

namespace holoscan {

/**
 * @brief Queue type selection for AsyncDataLoggerResource.
 *
 * Both queue types are fully thread-safe but offer different trade-offs:
 * - LockFree: Higher throughput, per-producer FIFO ordering only
 * - Ordered: Lower throughput, strict global FIFO ordering across all producers
 */
enum class DataLoggerQueueType {
  LockFree,  ///< Lock-free queue (highest performance, per-producer FIFO only)
  Ordered    ///< Mutex-based queue (strict global FIFO order, lower throughput)
};

/**
 * @brief Abstract base class for data logger queues.
 *
 * This interface allows different queue implementations to be used by AsyncDataLoggerResource,
 * enabling trade-offs between performance and ordering guarantees.
 *
 * All implementations must be thread-safe for concurrent access from multiple producers
 * and a single consumer (MPSC - Multiple Producer Single Consumer).
 *
 * @tparam T The type of elements stored in the queue (typically DataEntry)
 */
template <typename T>
class DataLoggerQueue {
 public:
  virtual ~DataLoggerQueue() = default;

  /**
   * @brief Attempt to enqueue an item (thread-safe).
   *
   * @param item The item to enqueue (will be moved)
   * @return true if successfully enqueued, false if queue is full
   */
  virtual bool try_enqueue(T&& item) = 0;

  /**
   * @brief Attempt to dequeue an item (thread-safe).
   *
   * @param item Output parameter where the dequeued item will be moved
   * @return true if an item was dequeued, false if queue is empty
   */
  virtual bool try_dequeue(T& item) = 0;

  /**
   * @brief Get size of the queue (thread-safe).
   *
   * Note: The accuracy of this value depends on the implementation:
   * - LockFreeQueue: Returns an approximation that may be stale by the time it's returned
   * - OrderedQueue: Returns the exact size at the moment the mutex was acquired
   *
   * @return Number of items in the queue (approximation or exact, depending on implementation)
   */
  virtual size_t size_approx() const = 0;
};

/**
 * @brief Lock-free queue implementation using MoodyCamel's ConcurrentQueue.
 *
 * This is a high-performance lock-free MPMC (Multiple Producer Multiple Consumer) queue
 * that uses per-producer sub-queues for optimal concurrency.
 *
 * Thread Safety:
 * - Fully thread-safe for concurrent enqueue and dequeue operations
 * - No locks or blocking operations (wait-free for producers, lock-free for consumers)
 * - Safe to call from multiple threads simultaneously
 *
 * Ordering Guarantees:
 * - Maintains FIFO order per-producer (items from the same thread stay ordered)
 * - Does NOT guarantee global FIFO order across multiple producers
 * - Items from different threads may be interleaved in any order
 *
 * Performance Characteristics:
 * - Excellent throughput with minimal contention
 * - size_approx() returns an estimate that may be stale immediately after return
 *
 * @tparam T The type of elements stored in the queue
 */
template <typename T>
class LockFreeQueue : public DataLoggerQueue<T> {
 public:
  /**
   * @brief Construct a lock-free queue with the specified capacity.
   *
   * @param capacity The target maximum capacity (note: ConcurrentQueue treats this as a hint)
   */
  explicit LockFreeQueue(size_t capacity) : queue_(capacity) {}

  bool try_enqueue(T&& item) override { return queue_.try_enqueue(std::move(item)); }

  bool try_dequeue(T& item) override { return queue_.try_dequeue(item); }

  size_t size_approx() const override { return queue_.size_approx(); }

 private:
  moodycamel::ConcurrentQueue<T> queue_;
};

/**
 * @brief Ordered queue implementation using mutex-protected std::queue.
 *
 * This queue uses a mutex to serialize all operations, ensuring both thread-safety
 * and strict global FIFO ordering across all producers and the single consumer.
 *
 * Thread Safety:
 * - Fully thread-safe for concurrent enqueue and dequeue operations
 * - Uses a mutex to serialize access (blocking if contended)
 * - Safe to call from multiple producer threads and a single consumer thread simultaneously
 *
 * Ordering Guarantees:
 * - Maintains strict global FIFO order across ALL producers
 * - Items are dequeued in the exact order they were enqueued
 * - Guaranteed temporal ordering regardless of which thread enqueued the item
 *
 * Performance Characteristics:
 * - Lower throughput than lock-free queues under high contention
 * - Predictable latency (no worst-case scenarios from lock-free algorithms)
 * - size_approx() returns the exact size (not approximate, despite the name)
 * - Capacity is a strict limit (will reject enqueue when full)
 * - Mutex is held during the entire move operation: if T has expensive move semantics
 *   (beyond simple pointer swaps), this may increase contention under heavy load
 * - Best suited for scenarios where ordering is critical and producer contention is moderate
 *
 * @tparam T The type of elements stored in the queue
 */
template <typename T>
class OrderedQueue : public DataLoggerQueue<T> {
 public:
  /**
   * @brief Construct an ordered queue with the specified capacity.
   *
   * @param capacity The maximum capacity (strict limit, unlike ConcurrentQueue)
   */
  explicit OrderedQueue(size_t capacity) : capacity_(capacity) {}

  bool try_enqueue(T&& item) override {
    std::lock_guard<std::mutex> lock(mutex_);

    if (queue_.size() >= capacity_) {
      return false;  // Queue full
    }

    queue_.push(std::move(item));
    return true;
  }

  /**
   * @brief Dequeue an item from the queue.
   *
   * @note Uses std::optional to defer destruction of the output parameter's previous value
   * until after the mutex is released. This prevents deadlock when T's destructor acquires
   * other locks (e.g., GILGuardedPyObject acquiring the Python GIL).
   *
   * Without this pattern, the naive `item = std::move(queue_.front())` would destroy the
   * old item value inside the mutex, causing deadlock: [Queue Mutex] → [GIL] → [Thread Join].
   *
   * Trade-off: One extra move for movable types, one extra copy for copy-only types.
   *
   * @param item Output parameter. Its destructor runs outside the critical section.
   * @return true if an item was dequeued, false if queue was empty
   */
  bool try_dequeue(T& item) override {
    // std::optional is used instead of a default-constructed temporary to avoid:
    // - Requiring T to have a default constructor
    // - Paying the cost of default construction for types with non-trivial constructors
    std::optional<T> deferred_item;

    {
      std::lock_guard<std::mutex> lock(mutex_);

      if (queue_.empty()) {
        return false;
      }

      // Extract item into temporary storage while holding mutex
      deferred_item = std::move(queue_.front());
      queue_.pop();
    }  // Mutex released here

    // Assign to output parameter OUTSIDE critical section.
    // Destructor of old 'item' value (containing potential GILGuardedPyObject from
    // previous iteration) runs here without holding the mutex, preventing deadlock.
    item = std::move(*deferred_item);
    return true;
  }

  size_t size_approx() const override {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  std::queue<T> queue_;
  mutable std::mutex mutex_;
  const size_t capacity_;
};

/**
 * @brief Factory function to create a queue of the specified type.
 *
 * This factory must be defined after the concrete queue classes to avoid forward declaration
 * issues.
 *
 * @tparam T The type of elements stored in the queue
 * @param type The queue type to create
 * @param capacity The queue capacity
 * @return A unique pointer to the created queue
 */
template <typename T>
inline std::unique_ptr<DataLoggerQueue<T>> create_data_logger_queue(DataLoggerQueueType type,
                                                                    size_t capacity) {
  switch (type) {
    case DataLoggerQueueType::LockFree:
      return std::make_unique<LockFreeQueue<T>>(capacity);
    case DataLoggerQueueType::Ordered:
      return std::make_unique<OrderedQueue<T>>(capacity);
    default:
      throw std::runtime_error("Unknown queue type");
  }
}

}  // namespace holoscan

// YAML converter for DataLoggerQueueType enum
namespace YAML {
template <>
struct convert<holoscan::DataLoggerQueueType> {
  static Node encode(const holoscan::DataLoggerQueueType& rhs) {
    Node node;
    switch (rhs) {
      case holoscan::DataLoggerQueueType::LockFree:
        node = "LockFree";
        break;
      case holoscan::DataLoggerQueueType::Ordered:
        node = "Ordered";
        break;
      default:
        throw std::runtime_error("Unknown DataLoggerQueueType enum value");
    }
    return node;
  }

  static bool decode(const Node& node, holoscan::DataLoggerQueueType& rhs) {
    if (!node.IsScalar()) {
      return false;
    }

    std::string value = node.as<std::string>();
    // Convert to lowercase for case-insensitive comparison (avoid UB on signed char)
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });

    if (value == "lockfree" || value == "lock_free") {
      rhs = holoscan::DataLoggerQueueType::LockFree;
      return true;
    } else if (value == "ordered") {
      rhs = holoscan::DataLoggerQueueType::Ordered;
      return true;
    }
    return false;
  }
};
}  // namespace YAML

#endif  // HOLOSCAN_CORE_RESOURCES_DATA_LOGGER_QUEUE_HPP
