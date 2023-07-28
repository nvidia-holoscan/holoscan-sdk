/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <netdb.h>
#include <unistd.h>
#include <cstdlib>  // for rand()
#include <cstring>  // for memset()
#include <unordered_set>
#include <vector>

#include "holoscan/logger/logger.hpp"

namespace holoscan {

static std::vector<int> get_unused_network_ports_impl(uint32_t num_ports, uint32_t min_port,
                                                      uint32_t max_port,
                                                      const std::vector<int>& used_ports) {
  struct addrinfo hints, *res, *p;
  char port_str[6];
  int sockfd, status;
  bool success = false;
  unsigned int seed = static_cast<unsigned int>(time(NULL));

  std::unordered_set<int> ports_set;
  ports_set.reserve(num_ports + used_ports.size());
  // Add exclude ports to the set
  for (auto port : used_ports) { ports_set.insert(port); }

  std::vector<int> unused_ports;
  unused_ports.reserve(num_ports);

  // Set up the hints structure
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;

  // Loop until we find enough unused ports
  while (unused_ports.size() < num_ports) {
    // Generate a random port number using rand_r() method
    int port = min_port + rand_r(&seed) % (max_port - min_port + 1);

    if (ports_set.find(port) != ports_set.end()) { continue; }

    ports_set.insert(port);
    // Convert the port number to a string
    snprintf(port_str, sizeof(port_str), "%d", port);

    // Call getaddrinfo to get a list of available addresses
    if ((status = getaddrinfo(NULL, port_str, &hints, &res)) != 0) {
      HOLOSCAN_LOG_ERROR("Error in getaddrinfo: {}", gai_strerror(status));
      continue;
    }

    // Try each address until we find one that works
    for (p = res; p != NULL; p = p->ai_next) {
      sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol);
      if (sockfd < 0) { continue; }

      int reuse_addr = 1;
      if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &reuse_addr, sizeof(int)) < 0) {
        HOLOSCAN_LOG_ERROR("Error setting socket options");
        close(sockfd);
        continue;
      }

      if (bind(sockfd, p->ai_addr, p->ai_addrlen) == 0) {
        success = true;
        break;
      }

      close(sockfd);
    }

    freeaddrinfo(res);

    if (success) { unused_ports.push_back(port); }
  }

  return unused_ports;
}

std::vector<int> get_unused_network_ports(uint32_t num_ports, uint32_t min_port, uint32_t max_port,
                                          const std::vector<int>& used_ports) {
  // Note:: Since opening and closing sockets makes the open port unavailable for a while, we use a
  // child process to parallelize the process of finding unused ports. The child process writes
  // the unused ports to a pipe that the parent process reads from.

  int pipefd[2];
  if (pipe(pipefd) == -1) {
    HOLOSCAN_LOG_ERROR("Error creating pipe");
    return {};
  }

  int pid = fork();
  if (pid < 0) {
    HOLOSCAN_LOG_ERROR("Error forking process");
    return {};
  } else if (pid > 0) {
    // Parent process
    close(pipefd[1]);  // close the write end of the pipe
    std::vector<int> unused_ports(num_ports);
    ssize_t bytes_read = read(pipefd[0], unused_ports.data(), unused_ports.size() * sizeof(int));
    if (bytes_read < 0) {
      HOLOSCAN_LOG_ERROR("Error reading from pipe");
      close(pipefd[0]);
      return {};
    }
    close(pipefd[0]);
    return unused_ports;
  } else {
    // Child process
    close(pipefd[0]);  // close the read end of the pipe
    std::vector<int> unused_ports =
        get_unused_network_ports_impl(num_ports, min_port, max_port, used_ports);
    ssize_t bytes_written =
        write(pipefd[1], unused_ports.data(), unused_ports.size() * sizeof(int));
    if (bytes_written < 0) {
      HOLOSCAN_LOG_ERROR("Error writing to pipe");
      close(pipefd[1]);
      exit(1);
    }
    close(pipefd[1]);
    exit(0);
  }
  return {};
}

}  // namespace holoscan
