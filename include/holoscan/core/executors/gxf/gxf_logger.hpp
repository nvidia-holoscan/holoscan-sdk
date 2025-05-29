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

#ifndef HOLOSCAN_CORE_EXECUTORS_GXF_GXF_LOGGER_HPP
#define HOLOSCAN_CORE_EXECUTORS_GXF_GXF_LOGGER_HPP

#include <gxf/logger/logger.hpp>

namespace holoscan::gxf {

/**
 * @brief Implementation of the ILogger interface for GXF logging.
 *
 * This class implements the nvidia::logger::ILogger interface to provide logging
 * capabilities for the GXF (Graph Execution Framework) system within Holoscan.
 * It overrides GXF's default logging format with Holoscan's format.
 */
class GXFLogger : public nvidia::logger::ILogger {
 public:
  void log(const char* file, int line, const char* name, int level, const char* message,
           void* arg = nullptr) override;

  void pattern(const char* pattern) override;
  const char* pattern() const override;

  void level(int level) override;
  int level() const override;

  void redirect(int level, void* output) override;
  void* redirect(int level) const override;

  /**
   * @brief Set the gxf log level object
   *
   * @param level The log level to set.
   */
  static void set_gxf_log_level(int level);

  /**
   * @brief Get the gxf log level object
   *
   * @return The current log level.
   */
  static int gxf_log_level();
};

}  // namespace holoscan::gxf

#endif /* HOLOSCAN_CORE_EXECUTORS_GXF_GXF_LOGGER_HPP */
