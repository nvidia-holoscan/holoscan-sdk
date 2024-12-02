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

#ifndef HOLOSCAN_CORE_MESSAGELABEL_HPP
#define HOLOSCAN_CORE_MESSAGELABEL_HPP

#include <ctime>
#include <iterator>
#include <string>
#include <unordered_set>
#include <vector>

#include "./forward_def.hpp"
#include "holoscan/logger/logger.hpp"

namespace holoscan {

// The initially reserved length of each path in message_paths
#define DEFAULT_PATH_LENGTH 10
// The initially reserved number of paths in message_paths
#define DEFAULT_NUM_PATHS 5

/**
 * @brief Return the current time in microseconds since the epoch.
 * This function previously used the C++11 standard library's chrono library.
 * This function now uses CLOCK_REALTIME to get the current time, because the clock chrono uses is
 * implementation-dependent. We want a known clock like CLOCK_REALTIME so that we can use the clock
 * across machines. CLOCK_REALTIME is a clock that is supposed to be synchronized with PTP
 * synchronization.
 *
 * @return The current time in microseconds returned by CLOCK_REALTIME. If CLOCK_REALTIME is not
 * available, it returns -1.
 */
static inline int64_t get_current_time_us() {
  struct timespec ts;
  if (clock_gettime(CLOCK_REALTIME, &ts) == 0) {
    return static_cast<int64_t>(ts.tv_sec) * 1000000 + static_cast<int64_t>(ts.tv_nsec) / 1000;
  } else {
    HOLOSCAN_LOG_ERROR("Error in clock_gettime");
    return -1;
  }
}

/** @brief This struct represents a timestamp label for a Holoscan Operator.
 *
 * The class stores information about the timestamps when an operator receives from
 * an input and when it publishes to an output. It also holds a reference to the
 * operator.
 *
 * This class is used by MessageLabel to create an array of Operators representing
 * a path.
 */
struct OperatorTimestampLabel {
 public:
  OperatorTimestampLabel() = default;

  /**
   * @brief Construct a new OperatorTimestampLabel object from an Operator pointer with a receive
   * timestamp equal to the current time and publish timestamp equal to -1.
   *
   * @param op_name The fully qualified name of the operator for which the timestamp label is
   * created.
   */
  explicit OperatorTimestampLabel(const std::string& op_name)
      : operator_name(op_name), rec_timestamp(get_current_time_us()), pub_timestamp(-1) {}

  OperatorTimestampLabel(const std::string& op_name, int64_t rec_t, int64_t pub_t)
      : operator_name(op_name), rec_timestamp(rec_t), pub_timestamp(pub_t) {}

  OperatorTimestampLabel(const OperatorTimestampLabel& o)
      : operator_name(o.operator_name),
        rec_timestamp(o.rec_timestamp),
        pub_timestamp(o.pub_timestamp) {}

  OperatorTimestampLabel& operator=(const OperatorTimestampLabel& o);

  void set_pub_timestamp_to_current() { pub_timestamp = get_current_time_us(); }

  // Operator* operator_ptr = nullptr;
  std::string operator_name = "";

  // The timestamp when an Operator receives from an input
  // For a root Operator, it is the start of the compute call
  int64_t rec_timestamp = 0;

  // The timestamp when an Operator publishes an output
  // For a leaf Operator, it is the end of the compute call
  int64_t pub_timestamp = 0;
};

/**
 * @brief Class to define a MessageLabel that is attached to every GXF Entity being communicated as
 * a message between Operators in Holoscan, for Data Frame Flow Tracking.
 *
 * A MessageLabel has a vector of paths, where each path is a vector of Operator references and
 * their publish and receive timestamps.
 */
class MessageLabel {
 public:
  using TimestampedPath = std::vector<OperatorTimestampLabel>;
  using PathOperators = std::unordered_set<std::string>;

  MessageLabel() {
    // By default, allocate DEFAULT_NUM_PATHS paths in the message_paths
    message_paths.reserve(DEFAULT_NUM_PATHS);
    message_path_operators.reserve(DEFAULT_NUM_PATHS);
  }

  MessageLabel(const MessageLabel& m)
      : message_paths(m.message_paths), message_path_operators(m.message_path_operators) {}

  /**
   * @brief Construct a new Message Label object from a vector of TimestampedPaths. This constructor
   * automatically fills the message_path_operators vector from the argument of the constructor.
   *
   * @param m_paths The vector of TimestampedPaths to create the MessageLabel from.
   */
  explicit MessageLabel(const std::vector<TimestampedPath> m_paths) : message_paths(m_paths) {
    for (auto& path : m_paths) {
      PathOperators new_path_operators;
      for (auto& op : path) { new_path_operators.insert(op.operator_name); }
      message_path_operators.push_back(new_path_operators);
    }
  }

  MessageLabel& operator=(const MessageLabel& m) {
    if (this != &m) {
      this->message_paths = m.message_paths;
      this->message_path_operators = m.message_path_operators;
    }
    return *this;
  }

  /**
   * @brief Get the number of paths in a MessageLabel.
   *
   * @return The number of paths in a MessageLabel.
   */
  int num_paths() const { return message_paths.size(); }

  /**
   * @brief Get all the names of the path in formatted string, which is comma-separated values of
   * the Operator names.
   *
   * @return std::vector<std::string> The vector of strings, where each string is a path name.
   */
  std::vector<std::string> get_all_path_names();

  std::vector<TimestampedPath> paths() const { return message_paths; }

  /**
   * @brief Get the current end-to-end latency of a path in microseconds.
   *
   * MessageLabel and the whole Data Flow Tracking component tracks the latency in microseconds,
   * which is sufficient for our use-cases.
   *
   * @param index the index of the path for which to get the latency
   * @return int64_t The current end-to-end latency of the index path in microseconds
   */
  int64_t get_e2e_latency(int index);

  /**
   * @brief Get the current end-to-end latency of a path in milliseconds.
   *
   * @param index The index of the path for which to get the latency.
   * @return double The current end-to-end latency of the index path in milliseconds.
   */
  double get_e2e_latency_ms(int index) { return ((double)get_e2e_latency(index) / 1000); }

  /**
   * @brief Get the end-to-end latency of a TimestampedPath in microseconds. It is a utility
   * function in the class and not dependent on the object of this class. Therefore, it's a static
   * function.
   *
   * @param path The path for which to get the latency
   * @return The end-to-end latency of the path in ms
   */
  static double get_path_e2e_latency_ms(TimestampedPath path) {
    int64_t latency = path.back().pub_timestamp - path.front().rec_timestamp;
    return (static_cast<double>(latency) / 1000);
  }

  /**
   * @brief Get the Timestamped path at the given index.
   *
   * @param index The index of the path to get
   * @return TimestampedPath& The timestamped path at the given index
   */
  TimestampedPath get_path(int index);

  /**
   * @brief Get the path name string which is comma-separated values of the operator names.
   *
   * @param index The index of the path to get
   * @return The path name string
   */
  std::string get_path_name(int index);

  /**
   * @brief Get the OperatorTimestampLabel at the given path and operator index
   *
   * @param path_index The path index of the OperatorTimestampLabel to get
   * @param op_index The Operator index of the OperatorTimestampLabel to get
   * @return OperatorTimestampLabel& The Operator reference at the given path and operator index
   */
  OperatorTimestampLabel& get_operator(int path_index, int op_index);

  /**
   * @brief Set an Operator's pub_timestamp
   *
   * @param path_index the path index of the Operator
   * @param op_index the index of the Operator in the path
   * @param pub_timestamp the new pub_timestamp to set
   */
  void set_operator_pub_timestamp(int path_index, int op_index, int64_t pub_timestamp);

  /**
   * @brief Set an Operator's pub_timestamp
   *
   * @param path_index the path index of the Operator
   * @param op_index the index of the Operator in the path
   * @param rec_timestamp the new rec_timestamp to set
   */
  void set_operator_rec_timestamp(int path_index, int op_index, int64_t rec_timestamp);

  /**
   * @brief Check if an operator is present in the MessageLabel. Returns an empty vector if the
   * operator is not present in any path.
   *
   * @param op_name The name of the operator to check
   * @return List of path indexes where the operator is present
   */
  std::vector<int> has_operator(const std::string& op_name) const;

  /**
   * @brief Add a new Operator timestamp to all the paths in a message label.
   *
   * @param o_timestamp The new operator timestamp to be added
   */
  void add_new_op_timestamp(holoscan::OperatorTimestampLabel o_timestamp);

  /**
   * @brief Update the publish timestamp of the last operator in all the paths in a message label.
   *
   */
  void update_last_op_publish();

  /**
   * @brief Add a new path to the MessageLabel.
   *
   * @param path The path to be added.
   */
  void add_new_path(TimestampedPath path);

  /**
   * @brief Convert the MessageLabel to a string.
   *
   * @return std::string The formatted string representing the MessageLabel with all the paths and
   * the Operators with their publish and receive timestamps.
   */
  std::string to_string() const;

  static std::string to_string(TimestampedPath path);

  /**
   * @brief Print the to_string() in the standard output with a heading for the MessageLabel.
   *
   */
  void print_all();

 private:
  std::vector<TimestampedPath> message_paths;

  /// List of paths where each path is a set of name of the operators
  /// This variable is used to quickly check whether an operator is in a path
  std::vector<PathOperators> message_path_operators;
};
}  // namespace holoscan

#endif /* HOLOSCAN_CORE_MESSAGELABEL_HPP */
