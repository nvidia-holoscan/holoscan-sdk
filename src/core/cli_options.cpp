/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/core/cli_options.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <string>
#include <utility>  // std::pair

#include "holoscan/logger/logger.hpp"

namespace holoscan {

static bool is_valid_ipv4(const std::string& ip) {
  struct sockaddr_in sa;
  return inet_pton(AF_INET, ip.c_str(), &(sa.sin_addr)) != 0;
}

static bool is_valid_ipv6(const std::string& ip) {
  struct sockaddr_in6 sa;
  return inet_pton(AF_INET6, ip.c_str(), &(sa.sin6_addr)) != 0;
}

std::string CLIOptions::resolve_hostname(const std::string& hostname) {
  // First check if the hostname is already a valid IPv4 or IPv6 address.
  if (is_valid_ipv4(hostname) || is_valid_ipv6(hostname)) {
    return hostname;  // no need to resolve, return the original IPv4 address.
  }

  struct addrinfo hints, *res, *p;
  int status;
  char ipstr[INET6_ADDRSTRLEN] = {0};

  std::memset(&hints, 0, sizeof hints);
  hints.ai_family = AF_UNSPEC;  // both IPv4 and IPv6
  hints.ai_socktype = SOCK_STREAM;

  if ((status = getaddrinfo(hostname.c_str(), nullptr, &hints, &res)) != 0) {
    HOLOSCAN_LOG_ERROR("getaddrinfo error: {}", gai_strerror(status));
    return "";
  }

  std::string resolved_ip;  // this will hold the final resolved IP address

  // First, look for an IPv4 address.
  for (p = res; p != nullptr && resolved_ip.empty(); p = p->ai_next) {
    if (p->ai_family == AF_INET) {  // IPv4
      struct sockaddr_in* ipv4 = (struct sockaddr_in*)p->ai_addr;
      inet_ntop(p->ai_family, &ipv4->sin_addr, ipstr, sizeof ipstr);
      resolved_ip = ipstr;  // save the IPv4 address
      HOLOSCAN_LOG_DEBUG("Resolved {} to IPv4: {}", hostname, ipstr);
    }
  }

  // If no IPv4 address was found, then look for an IPv6 address.
  if (resolved_ip.empty()) {
    for (p = res; p != nullptr; p = p->ai_next) {
      if (p->ai_family == AF_INET6) {  // IPv6
        struct sockaddr_in6* ipv6 = (struct sockaddr_in6*)p->ai_addr;
        inet_ntop(p->ai_family, &ipv6->sin6_addr, ipstr, sizeof ipstr);
        resolved_ip = ipstr;  // save the IPv6 address
        HOLOSCAN_LOG_DEBUG("Resolved {} to IPv6: {}", hostname, ipstr);
        break;
      }
    }
  }

  freeaddrinfo(res);  // Free the linked list

  return resolved_ip;  // Return the resolved IP address (IPv4 if available, otherwise IPv6)
}

std::string CLIOptions::parse_port(const std::string& address, const std::string& default_port) {
  // First, check if the address has square brackets, indicative of an IPv6 address with a port.
  auto bracket_start = address.find('[');
  auto bracket_end = address.find(']', bracket_start);
  if (bracket_start != std::string::npos && bracket_end != std::string::npos) {
    auto colon_pos = address.find(':', bracket_end);
    if (colon_pos != std::string::npos) {
      return address.substr(colon_pos + 1);
    }
  } else {
    // If there are no brackets, we might be dealing with an IPv4 address, a hostname:port pair,
    // or an IPv6 address without a specified port.
    auto first_colon_pos = address.find(':');
    auto last_colon_pos = address.rfind(':');
    // If the first and last colons are at the same position, there is only one colon in the string.
    if (first_colon_pos != std::string::npos && first_colon_pos == last_colon_pos) {
      // This is potentially an IPv4 address or a hostname with a port number.
      return address.substr(last_colon_pos + 1);
    }
  }
  // If no port is specified, or it's an IPv6 address without a port, return the default port.
  return default_port;
}

std::pair<std::string, std::string> CLIOptions::parse_address(const std::string& address,
                                                              const std::string& default_ip,
                                                              const std::string& default_port,
                                                              bool enclose_ipv6,
                                                              bool resolve_host) {
  std::string ip_address;
  std::string port;

  // Check for IPv6 address in brackets.
  auto bracket_start = address.find('[');
  auto bracket_end = address.find(']', bracket_start);
  if (bracket_start != std::string::npos && bracket_end != std::string::npos) {
    ip_address = address.substr(bracket_start + 1, bracket_end - bracket_start - 1);
    auto colon_pos_after_bracket = address.find(':', bracket_end);
    if (colon_pos_after_bracket != std::string::npos) {
      port = address.substr(colon_pos_after_bracket + 1);
    } else {
      port = default_port;
    }
  } else {
    // Handle IPv4 or hostname:port pair or IPv6 without brackets.
    auto first_colon_pos = address.find(':');
    auto last_colon_pos = address.rfind(':');
    if (first_colon_pos != last_colon_pos) {
      // More than one colon found without brackets, this is an IPv6 address without a port.
      ip_address = address;
      port = default_port;
    } else if (first_colon_pos != std::string::npos) {
      // One colon found, split into IP address/hostname and port.
      ip_address = first_colon_pos == 0 ? default_ip : address.substr(0, first_colon_pos);
      port = address.substr(first_colon_pos + 1);
    } else {
      // No colons found, assume address is an IP address, or a hostname.
      ip_address = address.empty() ? default_ip : address;
      port = default_port;
    }
  }

  // If the IP address is not empty and `resolve_host` is true, resolve it to an IP address.
  if (resolve_host && !ip_address.empty()) {
    ip_address = resolve_hostname(ip_address);
  }

  bool is_ipv6 = false;
  auto first_colon_pos = ip_address.find(':');
  auto last_colon_pos = ip_address.rfind(':');
  if (first_colon_pos != last_colon_pos) {
    // More than one colon found in IP address, this is an IPv6 address
    is_ipv6 = true;
  }

  // Enclose IPv6 addresses in brackets if requested.
  if (is_ipv6 && enclose_ipv6 && !port.empty()) {
    ip_address = '[' + ip_address + ']';
  }

  return std::make_pair(ip_address, port);
}

void CLIOptions::print() const {
  HOLOSCAN_LOG_INFO("CLI Options:");
  HOLOSCAN_LOG_INFO("  run_driver: {}", run_driver);
  HOLOSCAN_LOG_INFO("  run_worker: {}", run_worker);
  HOLOSCAN_LOG_INFO("  driver_address: {}", driver_address);
  HOLOSCAN_LOG_INFO("  worker_address: {}", worker_address);
  HOLOSCAN_LOG_INFO("  worker_targets: {}", fmt::join(worker_targets, ", "));
  HOLOSCAN_LOG_INFO("  config_path: {}", config_path);
}

}  // namespace holoscan
