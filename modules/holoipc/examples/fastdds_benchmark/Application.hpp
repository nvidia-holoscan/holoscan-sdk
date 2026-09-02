// SPDX-FileCopyrightText: Copyright 2024 Proyectos y Sistemas de Mantenimiento SL (eProsima).
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file Application.hpp
 *
 */

#ifndef HOLOIPC_EXAMPLES_FASTDDS_BENCHMARK_APPLICATION_HPP
#define HOLOIPC_EXAMPLES_FASTDDS_BENCHMARK_APPLICATION_HPP

#include <atomic>
#include <memory>
#include <string>

#include "CLIParser.hpp"

namespace holoscan {
namespace ipc {
namespace fastdds_benchmark {

class Application {
 public:
  //! Virtual destructor
  virtual ~Application() = default;

  //! Run application
  virtual void run() = 0;

  //! Trigger the end of execution
  virtual void stop() = 0;

  //! Factory method to create applications based on configuration
  static std::shared_ptr<Application> make_app(const CLIParser::hello_world_config& config,
                                               const std::string& topic_name);
};

}  // namespace fastdds_benchmark
}  // namespace ipc
}  // namespace holoscan

#endif  // HOLOIPC_EXAMPLES_FASTDDS_BENCHMARK_APPLICATION_HPP
