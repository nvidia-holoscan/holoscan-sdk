// SPDX-FileCopyrightText: Copyright 2024 Proyectos y Sistemas de Mantenimiento SL (eProsima).
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file main.cpp
 * Entry point for the fastdds_example (publisher or subscriber). Parses CLI, creates the app,
 * runs it and handles signals.
 */

#include <csignal>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

#include <holoscan/logger/logger.hpp>

#include "Application.hpp"
#include "CLIParser.hpp"

using namespace holoscan::ipc::dds_example;

// Signal handling is kept minimal to stay close to the original Fast DDS sample. This pattern
// is not fully async-signal-safe (POSIX §2.4.3): the handler invokes std::function and app
// code. Production or long-running code should use an async-signal-safe approach (e.g.
// self-pipe + a thread that performs stop/cleanup).
std::function<void(int)> stop_app_handler;
void signal_handler(int signum) {
  stop_app_handler(signum);
}

int main(int argc, char** argv) {
  holoscan::Logger::set_level(holoscan::LogLevel::INFO);

  auto ret = EXIT_SUCCESS;
  const std::string topic_name = "cuda_buffer_topic";
  CLIParser::hello_world_config config = CLIParser::parse_cli_options(argc, argv);
  std::size_t samples = 0;
  switch (config.entity) {
    case CLIParser::EntityKind::PUBLISHER:
      samples = config.pub_config.samples;
      break;
    case CLIParser::EntityKind::SUBSCRIBER:
      samples = config.sub_config.samples;
      break;
    default:
      break;
  }

  std::string app_name = CLIParser::parse_entity_kind(config.entity);
  std::shared_ptr<Application> app;

  try {
    app = Application::make_app(config, topic_name);
  } catch (const std::runtime_error& e) {
    std::cerr << app_name << ": " << e.what() << std::endl;
    ret = EXIT_FAILURE;
  }

  if (EXIT_FAILURE != ret) {
    std::thread thread(&Application::run, app);

    if (samples == 0) {
      std::cout << app_name << " running. Please press Ctrl+C to stop the " << app_name
                << " at any time." << std::endl;
    } else {
      std::cout << app_name << " running for " << samples
                << " samples. Please press Ctrl+C to stop the " << app_name << " at any time."
                << std::endl;
    }

    stop_app_handler = [&](int signum) {
      std::cout << "\n"
                << CLIParser::parse_signal(signum) << " received, stopping " << app_name
                << " execution." << std::endl;
      app->stop();
    };

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
#ifndef _WIN32
    signal(SIGQUIT, signal_handler);
    signal(SIGHUP, signal_handler);
#endif  // _WIN32

    thread.join();
  }

  return ret;
}
