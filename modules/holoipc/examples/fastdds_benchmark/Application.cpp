// SPDX-FileCopyrightText: Copyright 2024 Proyectos y Sistemas de Mantenimiento SL (eProsima).
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file Application.cpp
 *
 */

#include "Application.hpp"

#include <memory>
#include <string>

#include "CLIParser.hpp"
#include "ListenerSubscriberApp.hpp"
#include "PublisherApp.hpp"

namespace holoscan {
namespace ipc {
namespace fastdds_benchmark {

//! Factory method to create a publisher or subscriber
std::shared_ptr<Application> Application::make_app(const CLIParser::hello_world_config& config,
                                                   const std::string& topic_name) {
  std::shared_ptr<Application> entity;
  switch (config.entity) {
    case CLIParser::EntityKind::PUBLISHER:
      entity = std::make_shared<PublisherApp>(config.pub_config, topic_name);
      break;
    case CLIParser::EntityKind::SUBSCRIBER:
      entity = std::make_shared<ListenerSubscriberApp>(config.sub_config, topic_name);
      break;
    case CLIParser::EntityKind::UNDEFINED:
    default:
      throw std::runtime_error("Entity initialization failed");
      break;
  }
  return entity;
}

}  // namespace fastdds_benchmark
}  // namespace ipc
}  // namespace holoscan
