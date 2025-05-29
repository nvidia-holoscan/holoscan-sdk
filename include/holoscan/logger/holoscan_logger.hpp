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

#ifndef HOLOSCAN_LOGGER_HOLOSCAN_LOGGER_HPP
#define HOLOSCAN_LOGGER_HOLOSCAN_LOGGER_HPP

#include "common/logger/spdlog_logger.hpp"

namespace holoscan {

/**
 * @brief HoloscanLogger is a singleton class that provides a logger for Holoscan.
 *
 * This class extends the SpdlogLogger class from the common/logger/spdlog_logger.hpp file.
 * It provides a static instance of itself that can be accessed using the instance() method.
 *
 * @note This class is a singleton and cannot be copied or assigned.
 */
class HoloscanLogger : public ::nvidia::logger::SpdlogLogger {
 public:
  /**
   * @brief Get the singleton instance of the HoloscanLogger.
   *
   * @return The singleton instance of the HoloscanLogger.
   */
  static HoloscanLogger& instance();

  // Delete the copy constructor and the copy assignment operator.
  HoloscanLogger(const HoloscanLogger&) = delete;
  HoloscanLogger& operator=(const HoloscanLogger&) = delete;

 private:
  using ::nvidia::logger::SpdlogLogger::SpdlogLogger;
};

}  // namespace holoscan


#endif /* HOLOSCAN_LOGGER_HOLOSCAN_LOGGER_HPP */
