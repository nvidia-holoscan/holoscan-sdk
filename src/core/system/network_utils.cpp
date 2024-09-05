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
#include <arpa/inet.h>   // for inet_ntop()
#include <ifaddrs.h>     // for getifaddrs()
#include <netdb.h>       // for getaddrinfo(), gai_strerror(), freeaddrinfo(), addrinfo
#include <netinet/in.h>  // for sockaddr_in
#include <unistd.h>
#include <cstdlib>  // for rand()
#include <cstring>  // for memset()
#include <iostream>
#include <memory>   // for unique_ptr
#include <sstream>  // for istringstream
#include <string>
#include <unordered_set>
#include <vector>

#include "holoscan/logger/logger.hpp"

namespace holoscan {

class Socket {
 public:
  explicit Socket(int domain, int type, int protocol) : sockfd_(socket(domain, type, protocol)) {
    if (sockfd_ < 0) { throw std::runtime_error("Error creating socket"); }
  }

  ~Socket() {
    if (sockfd_ >= 0) { close(sockfd_); }
    sockfd_ = -1;
  }

  int descriptor() const { return sockfd_; }

 private:
  int sockfd_ = -1;
};

class AddrInfo {
 public:
  AddrInfo(const char* port_str, const addrinfo& hints) {
    int status = getaddrinfo(nullptr, port_str, &hints, &res_);
    if (status != 0) {
      HOLOSCAN_LOG_ERROR("Error in getaddrinfo: {}", gai_strerror(status));
      throw std::runtime_error("Error in getaddrinfo");
    }
  }

  ~AddrInfo() { freeaddrinfo(res_); }

  addrinfo* get() const { return res_; }

 private:
  addrinfo* res_;
};

static bool is_port_available(int port) {
  struct addrinfo hints {};
  // Set up the hints structure
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;

  char port_str[6];
  snprintf(port_str, sizeof(port_str), "%d", port);

  try {
    AddrInfo ai(port_str, hints);

    for (struct addrinfo* p = ai.get(); p != nullptr; p = p->ai_next) {
      try {
        Socket sockfd(p->ai_family, p->ai_socktype, p->ai_protocol);

        int reuse_addr = 1;
        if (setsockopt(sockfd.descriptor(), SOL_SOCKET, SO_REUSEADDR, &reuse_addr, sizeof(int)) <
            0) {
          HOLOSCAN_LOG_ERROR("Error setting socket options");
          continue;
        }

        if (bind(sockfd.descriptor(), p->ai_addr, p->ai_addrlen) == 0) { return true; }
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Error creating socket: {}", e.what());
      }
    }
  } catch (const std::exception& e) { HOLOSCAN_LOG_ERROR("Error creating addrinfo: {}", e.what()); }

  return false;
}

std::vector<int> get_unused_network_ports(uint32_t num_ports, uint32_t min_port, uint32_t max_port,
                                          const std::vector<int>& used_ports,
                                          const std::vector<int>& prefer_ports) {
  // Add exclude ports to the set
  std::unordered_set<int> used_port_set(used_ports.begin(), used_ports.end());
  used_port_set.reserve(num_ports + used_ports.size());

  std::vector<int> unused_ports;
  unused_ports.reserve(num_ports);

  auto try_insert_port = [&unused_ports, num_ports, &used_port_set](int port) {
    if (unused_ports.size() < num_ports && used_port_set.insert(port).second &&
        is_port_available(port)) {
      unused_ports.push_back(port);
    }
  };

  // Try to insert prefer_ports first
  for (int port : prefer_ports) { try_insert_port(port); }

  if (!prefer_ports.empty()) {
    min_port = prefer_ports.back() + 1;
    max_port = 65535;
  }

  // Try to insert ports in the range [min_port, max_port]
  for (int port = min_port; port <= static_cast<int>(max_port) && unused_ports.size() < num_ports;
       ++port) {
    try_insert_port(port);
  }

  HOLOSCAN_LOG_DEBUG(
      "unused_ports={} (size:{})", fmt::join(unused_ports, ","), unused_ports.size());

  return unused_ports;
}

std::vector<int> get_preferred_network_ports(const char* env_var_name) {
  if (env_var_name == nullptr || env_var_name[0] == '\0') { return {}; }
  const char* preferred_ports_str = std::getenv(env_var_name);
  if (preferred_ports_str == nullptr || preferred_ports_str[0] == '\0') { return {}; }

  std::vector<int> preferred_ports;
  std::istringstream iss(preferred_ports_str);
  std::string token;

  while (std::getline(iss, token, ',')) {
    try {
      int port_number = std::stoi(token);
      if (port_number < 0 || port_number > 65535) {
        HOLOSCAN_LOG_ERROR("Invalid port number found in {}: {}", env_var_name, token);
        return {};  // Return an empty vector
      }
      preferred_ports.push_back(port_number);
    } catch (const std::invalid_argument& e) {
      HOLOSCAN_LOG_ERROR("Invalid argument found in {}: {} ({})", env_var_name, e.what(), token);
      return {};  // Return an empty vector
    } catch (const std::out_of_range& e) {
      HOLOSCAN_LOG_ERROR(
          "Out of range argument found in {}: {} ({})", env_var_name, e.what(), token);
      return {};  // Return an empty vector
    }
  }

  return preferred_ports;
}

static std::string convert_sockaddr_to_ip(struct sockaddr* sa) {
  char ip[INET6_ADDRSTRLEN]{};
  if (sa->sa_family == AF_INET) {
    struct sockaddr_in* sa_in = reinterpret_cast<struct sockaddr_in*>(sa);
    inet_ntop(AF_INET, &(sa_in->sin_addr), ip, INET_ADDRSTRLEN);
  } else {
    struct sockaddr_in6* sa_in6 = reinterpret_cast<struct sockaddr_in6*>(sa);
    inet_ntop(AF_INET6, &(sa_in6->sin6_addr), ip, INET6_ADDRSTRLEN);
  }
  return std::string(ip);
}

std::string get_associated_local_ip(const std::string& remote_ip) {
  struct ifaddrs *ifaddr, *ifa;
  struct addrinfo hints, *res;

  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  if (getifaddrs(&ifaddr) == -1) {
    HOLOSCAN_LOG_ERROR("Error happened while calling getifaddrs: {}", std::strerror(errno));
    return "";
  }

  std::unique_ptr<struct ifaddrs, void (*)(struct ifaddrs*)> ifaddr_ptr(ifaddr, freeifaddrs);

  if (getaddrinfo(remote_ip.c_str(), nullptr, &hints, &res) != 0) {
    HOLOSCAN_LOG_ERROR("An error occurred while calling getaddrinfo: {}", std::strerror(errno));
    return "";
  }

  std::unique_ptr<struct addrinfo, void (*)(struct addrinfo*)> res_ptr(res, freeaddrinfo);

  std::string resolved_ip = convert_sockaddr_to_ip(res->ai_addr);

  HOLOSCAN_LOG_DEBUG("Querying local IP for remote IP '{}': {}", remote_ip, resolved_ip);

  for (ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr) continue;

    std::string ip = convert_sockaddr_to_ip(ifa->ifa_addr);
    HOLOSCAN_LOG_DEBUG("  checking interface: {}, IP: {}", ifa->ifa_name, ip);

    if (ip == resolved_ip) {
      HOLOSCAN_LOG_DEBUG("Found the matching interface: {}, IP: {}", ifa->ifa_name, ip);
      return ip;
    }
  }

  return "";
}

}  // namespace holoscan
