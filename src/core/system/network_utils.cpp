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
#include <arpa/inet.h>   // for inet_ntop()
#include <ifaddrs.h>     // for getifaddrs()
#include <netdb.h>       // for getaddrinfo(), gai_strerror(), freeaddrinfo(), addrinfo
#include <netinet/in.h>  // for sockaddr_in
#include <spawn.h>       // for posix_spawnp()
#include <sys/wait.h>    // for waitpid()
#include <unistd.h>
#include <cstdlib>  // for rand()
#include <cstring>  // for memset()
#include <iostream>
#include <memory>  // for unique_ptr
#include <sstream>  // for istringstream
#include <string>
#include <unordered_set>
#include <vector>

#include "holoscan/logger/logger.hpp"

/// Global variable that holds the environment variables
extern char** environ;

namespace holoscan {

////////////////////////////////////////////////////////////////////////////////////////////////////
// The following snippet from CLI11 that is under the BSD-3-Clause license:
//     https://github.com/CLIUtils/CLI11/blob/89601ee/include/CLI/impl/Argv_inl.hpp
// We have modified it to use Linux specific functions to get the command line arguments from
// /proc/self/cmdline. Please see https://github.com/CLIUtils/CLI11/pull/804.
// This is to avoid the use of `main()` and `argv` when spawning a child process.
////////////////////////////////////////////////////////////////////////////////////////////////////
namespace cli11_detail {
static const std::vector<const char*>& args() {
  // This function uses initialization via lambdas extensively to take advantage of the thread
  // safety of static variable initialization [stmt.dcl.3]
  static const std::vector<const char*> static_args = [] {
    static const std::vector<char> static_cmdline = [] {
      // On posix, retrieve arguments from /proc/self/cmdline, separated by null terminators.
      std::vector<char> cmdline;

      auto deleter = [](FILE* f) { std::fclose(f); };
      std::unique_ptr<FILE, decltype(deleter)> fp_unique(std::fopen("/proc/self/cmdline", "r"),
                                                         deleter);
      FILE* fp = fp_unique.get();
      if (!fp) {
        throw std::runtime_error(
            "could not open /proc/self/cmdline for reading");  // LCOV_EXCL_LINE
      }

      size_t size = 0;
      while (std::feof(fp) == 0) {
        cmdline.resize(size + 128);
        size += std::fread(cmdline.data() + size, 1, 128, fp);

        if (std::ferror(fp) != 0) {
          throw std::runtime_error("error during reading /proc/self/cmdline");  // LCOV_EXCL_LINE
        }
      }
      cmdline.resize(size);

      return cmdline;
    }();

    std::size_t argc =
        static_cast<std::size_t>(std::count(static_cmdline.begin(), static_cmdline.end(), '\0'));
    std::vector<const char*> static_args_result;
    static_args_result.reserve(argc);

    for (auto it = static_cmdline.begin(); it != static_cmdline.end();
         it = std::find(it, static_cmdline.end(), '\0') + 1) {
      static_args_result.push_back(static_cmdline.data() + (it - static_cmdline.begin()));
    }

    return static_args_result;
  }();

  return static_args;
}
}  // namespace cli11_detail

static const char* const* argv() {
  return cli11_detail::args().data();
}

static int argc() {
  return static_cast<int>(cli11_detail::args().size());
}
////////////////////////////////////////////////////////////////////////////////////////////////////

class Socket {
 public:
  explicit Socket(int domain, int type, int protocol) : sockfd_(socket(domain, type, protocol)) {
    if (sockfd_ < 0) { throw std::runtime_error("Error creating socket"); }
  }

  ~Socket() { close(sockfd_); }

  int descriptor() const { return sockfd_; }

 private:
  int sockfd_;
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

static ssize_t write_all(int fd, const void* buffer, size_t count) {
  const char* buf_ptr = reinterpret_cast<const char*>(buffer);
  size_t bytes_left = count;
  ssize_t total_bytes_written = 0;

  while (bytes_left > 0) {
    ssize_t bytes_written = write(fd, buf_ptr, bytes_left);
    if (bytes_written < 0) {
      if (errno == EINTR) continue;  // Retry if interrupted by signal
      throw std::runtime_error(
          fmt::format("Error writing to fd {}: {} ({})", fd, std::strerror(errno), errno));
    }
    if (bytes_written == 0) { throw std::runtime_error("Unexpected zero write"); }
    total_bytes_written += bytes_written;
    bytes_left -= bytes_written;
    buf_ptr += bytes_written;
  }

  return total_bytes_written;
}

static ssize_t read_all(int fd, void* buffer, size_t count) {
  char* buf_ptr = reinterpret_cast<char*>(buffer);
  size_t bytes_left = count;
  ssize_t total_bytes_read = 0;

  while (bytes_left > 0) {
    ssize_t bytes_read = read(fd, buf_ptr, bytes_left);
    if (bytes_read < 0) {
      if (errno == EINTR) continue;  // Retry if interrupted by signal
      throw std::runtime_error(
          fmt::format("Error reading from fd {}: {} ({})", fd, std::strerror(errno), errno));
    }
    if (bytes_read == 0) { throw std::runtime_error("Unexpected EOF while reading"); }
    total_bytes_read += bytes_read;
    bytes_left -= bytes_read;
    buf_ptr += bytes_read;
  }

  return total_bytes_read;
}

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

static std::vector<int> get_unused_network_ports_impl(uint32_t num_ports, uint32_t min_port,
                                                      uint32_t max_port,
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

  return unused_ports;
}

class Pipe {
 public:
  Pipe() {
    if (pipe(fd_) == -1) { throw std::runtime_error("Error creating pipe"); }
  }
  explicit Pipe(int fd0, int fd1) : fd_{fd0, fd1} {}

  ~Pipe() {
    if (fd_[0] != -1) { close(fd_[0]); }
    if (fd_[1] != -1) { close(fd_[1]); }
  }

  int read_fd() const { return fd_[0]; }
  int write_fd() const { return fd_[1]; }

 private:
  int fd_[2];
};

class PosixSpawnAttr {
 public:
  PosixSpawnAttr() {
    if (posix_spawnattr_init(&attr_) != 0) {
      throw std::runtime_error("Error initializing posix_spawnattr_t");
    }
  }

  ~PosixSpawnAttr() { posix_spawnattr_destroy(&attr_); }

  posix_spawnattr_t* get() { return &attr_; }

 private:
  posix_spawnattr_t attr_;
};

class PosixSpawnFileActions {
 public:
  PosixSpawnFileActions() {
    if (posix_spawn_file_actions_init(&actions_) != 0) {
      throw std::runtime_error("Error initializing posix_spawn_file_actions_t");
    }
  }

  ~PosixSpawnFileActions() { posix_spawn_file_actions_destroy(&actions_); }

  posix_spawn_file_actions_t* get() { return &actions_; }

 private:
  posix_spawn_file_actions_t actions_;
};

namespace {

/**
 * @brief Initializes the network port search process in a child process.
 *
 * The Initializer class is responsible for setting up and initializing
 * the process of searching for unused network ports. It will check if
 * it's running in a child process context, setup logging, read parameters
 * from a pipe, execute the search for unused network ports, and send the
 * results back to the parent process through a pipe.
 *
 * A static object of this class is created to ensure its constructor is
 * called before main() starts, and its destructor is called after main() ends.
 */
class Initializer {
 public:
  /**
   * @brief Constructor of the Initializer class.
   *
   * The constructor checks if it's running in a child process context by
   * checking environment variables. If it is in a child process context,
   * it sets up logging, reads parameters from the parent process through a
   * pipe, finds unused network ports, and sends the results back to the
   * parent process through a pipe.
   */
  Initializer() {
    // Check if this is the child process
    const char* is_child_process = std::getenv("HOLOSCAN_UNUSED_IP_CHILD_PROCESS");
    if (is_child_process == nullptr || std::strncmp(is_child_process, "1", 1) != 0) { return; }

    HOLOSCAN_LOG_DEBUG("Child process started to find unused network ports");

    // Enable logging
    holoscan::set_log_level(LogLevel::INFO);
    holoscan::set_log_pattern();

    try {
      // Get the pipe file descriptors from the environment variables
      char* parent_to_child_read_fd_str = std::getenv("HOLOSCAN_UNUSED_IP_PARENT_TO_CHILD_READ_FD");
      char* child_to_parent_write_fd_str =
          std::getenv("HOLOSCAN_UNUSED_IP_CHILD_TO_PARENT_WRITE_FD");
      if (parent_to_child_read_fd_str == nullptr || child_to_parent_write_fd_str == nullptr) {
        HOLOSCAN_LOG_ERROR("Error getting pipe file descriptors from environment variables");
        exit(1);
      }
      Pipe parent_to_child(std::stoi(parent_to_child_read_fd_str), -1);
      Pipe child_to_parent(-1, std::stoi(child_to_parent_write_fd_str));

      // Deserialize parameters from parent process

      uint32_t num_ports, min_port, max_port;
      size_t used_ports_size;
      std::vector<int> used_ports;
      size_t prefer_ports_size;
      std::vector<int> prefer_ports;

      // Read scalar values
      ssize_t bytes_read = read(parent_to_child.read_fd(), &num_ports, sizeof(num_ports));
      bytes_read += read(parent_to_child.read_fd(), &min_port, sizeof(min_port));
      bytes_read += read(parent_to_child.read_fd(), &max_port, sizeof(max_port));

      // Read vector values
      bytes_read += read(parent_to_child.read_fd(),
                         &used_ports_size,
                         sizeof(used_ports_size));  // Reading the size of the vector
      used_ports.resize(used_ports_size);
      bytes_read +=
          read_all(parent_to_child.read_fd(), used_ports.data(), used_ports_size * sizeof(int));

      bytes_read += read(parent_to_child.read_fd(),
                         &prefer_ports_size,
                         sizeof(prefer_ports_size));  // Reading the size of the vector
      prefer_ports.resize(prefer_ports_size);
      bytes_read +=
          read_all(parent_to_child.read_fd(), prefer_ports.data(), prefer_ports_size * sizeof(int));

      if (bytes_read !=
          static_cast<ssize_t>(sizeof(num_ports) + sizeof(min_port) + sizeof(max_port) +
                               sizeof(used_ports_size) + used_ports_size * sizeof(int) +
                               sizeof(prefer_ports_size) + prefer_ports_size * sizeof(int))) {
        throw std::runtime_error(
            fmt::format("Unable to read port parameters from pipe (fd={}): {} ({})",
                        parent_to_child.read_fd(),
                        std::strerror(errno),
                        errno));
      }

      HOLOSCAN_LOG_DEBUG("num_ports={}, min_port={}, max_port={}, used_ports={}, prefer_ports={}",
                         num_ports,
                         min_port,
                         max_port,
                         fmt::join(used_ports, ","),
                         fmt::join(prefer_ports, ","));

      std::vector<int> unused_ports =
          get_unused_network_ports_impl(num_ports, min_port, max_port, used_ports, prefer_ports);

      // Serialize results to the parent process using the child_to_parent pipe

      int unused_ports_size = unused_ports.size();
      ssize_t bytes_written = write(child_to_parent.write_fd(),
                                    &unused_ports_size,
                                    sizeof(unused_ports_size));  // Writing the size of the vector
      bytes_written += write_all(
          child_to_parent.write_fd(), unused_ports.data(), unused_ports_size * sizeof(int));

      if (bytes_written !=
          static_cast<ssize_t>(sizeof(unused_ports_size) + unused_ports_size * sizeof(int))) {
        throw std::runtime_error(
            fmt::format("Unable to write unused port info to pipe (fd={}): {} ({})",
                        child_to_parent.write_fd(),
                        std::strerror(errno),
                        errno));
      }

      // Exit with success
      exit(0);
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Error in child process: {}", e.what());
    }
    // Exit with error
    exit(1);
  }
};

/// Create a static object of the class
/// Its constructor will be called before main() starts,
/// and its destructor will be called after main() ends.
static Initializer initializer;

}  // namespace

std::vector<int> get_unused_network_ports(uint32_t num_ports, uint32_t min_port, uint32_t max_port,
                                          const std::vector<int>& used_ports,
                                          const std::vector<int>& prefer_ports) {
  // Note:: Since opening and closing sockets makes the open port unavailable for a while, we use a
  // child process to parallelize the process of finding unused ports. The child process writes
  // the unused ports to a pipe that the parent process reads from.

  // Create two pipes: one for parent-to-child communication, and one for child-to-parent
  // communication
  try {
    Pipe parent_to_child_pipe;
    Pipe child_to_parent_pipe;

    PosixSpawnAttr attr;
    PosixSpawnFileActions action;

    // Set the file descriptors to be closed in the child process
    posix_spawn_file_actions_addclose(action.get(), parent_to_child_pipe.write_fd());
    posix_spawn_file_actions_addclose(action.get(), child_to_parent_pipe.read_fd());

    // Copy argv() to args.
    char* args[argc() + 1]{};
    for (int i = 0; i < argc(); ++i) { args[i] = const_cast<char*>(argv()[i]); }
    args[argc()] = nullptr;  // the last element must be a nullptr

    // Get current environment variables
    char** env_vars = environ;

    // Determine the number of current environment variables
    int env_count = 0;
    while (env_vars[env_count] != nullptr) { env_count++; }

    // Create a vector to hold the current environment variables plus the new one
    std::vector<std::string> local_env_vars;
    local_env_vars.reserve(env_count + 5);
    for (int i = 0; i < env_count; i++) { local_env_vars.push_back(std::string(env_vars[i])); }

    // Add the new environment variable to the local vector
    local_env_vars.push_back("HOLOSCAN_UNUSED_IP_CHILD_PROCESS=1");

    // Update the environment variable setup to include the file descriptors for both pipes
    local_env_vars.push_back(fmt::format("HOLOSCAN_UNUSED_IP_PARENT_TO_CHILD_READ_FD={}",
                                         parent_to_child_pipe.read_fd()));
    local_env_vars.push_back(fmt::format("HOLOSCAN_UNUSED_IP_CHILD_TO_PARENT_WRITE_FD={}",
                                         child_to_parent_pipe.write_fd()));

    // Convert the local vector to a char** array
    std::vector<char*> local_env_ptrs;
    local_env_ptrs.reserve(local_env_vars.size() + 1);
    for (auto& str : local_env_vars) { local_env_ptrs.push_back(&str[0]); }
    local_env_ptrs.push_back(nullptr);  // Null-terminate the array

    // Now local_env_ptrs is a char** array with the original environment variables plus the new
    // ones
    char** new_env_vars = &local_env_ptrs[0];

    pid_t pid;
    int status = posix_spawnp(&pid, args[0], action.get(), attr.get(), args, new_env_vars);
    if (status != 0) {
      throw std::runtime_error(
          fmt::format("Unable to spawn child process: {} ({})", std::strerror(status), status));
    } else {
      // Parent process
      HOLOSCAN_LOG_DEBUG("Child process spawned with pid {}", pid);

      // Serialize parameters to send to child process

      size_t used_ports_size = used_ports.size();
      size_t prefer_ports_size = prefer_ports.size();

      // Write scalar values
      ssize_t bytes_written = write(parent_to_child_pipe.write_fd(), &num_ports, sizeof(num_ports));
      bytes_written += write(parent_to_child_pipe.write_fd(), &min_port, sizeof(min_port));
      bytes_written += write(parent_to_child_pipe.write_fd(), &max_port, sizeof(max_port));

      // Write vector values
      bytes_written += write(parent_to_child_pipe.write_fd(),
                             &used_ports_size,
                             sizeof(used_ports_size));  // Writing the size of the vector
      bytes_written += write_all(
          parent_to_child_pipe.write_fd(), used_ports.data(), used_ports_size * sizeof(int));

      bytes_written += write(parent_to_child_pipe.write_fd(),
                             &prefer_ports_size,
                             sizeof(prefer_ports_size));  // Writing the size of the vector
      bytes_written += write_all(
          parent_to_child_pipe.write_fd(), prefer_ports.data(), prefer_ports_size * sizeof(int));

      if (bytes_written !=
          static_cast<ssize_t>(sizeof(num_ports) + sizeof(min_port) + sizeof(max_port) +
                               sizeof(used_ports_size) + used_ports_size * sizeof(int) +
                               sizeof(prefer_ports_size) + prefer_ports_size * sizeof(int))) {
        throw std::runtime_error(
            fmt::format("Unable to write port parameters to pipe (fd={}): {} ({})",
                        parent_to_child_pipe.write_fd(),
                        std::strerror(errno),
                        errno));
      }

      // Read the unused ports from the child process

      int unused_ports_size = 0;

      ssize_t bytes_read =
          read(child_to_parent_pipe.read_fd(), &unused_ports_size, sizeof(unused_ports_size));

      std::vector<int> unused_ports(unused_ports_size);
      bytes_read += read_all(
          child_to_parent_pipe.read_fd(), unused_ports.data(), unused_ports_size * sizeof(int));
      if (bytes_read !=
          static_cast<ssize_t>(sizeof(unused_ports_size) + unused_ports_size * sizeof(int))) {
        throw std::runtime_error(
            fmt::format("Unable to read unused ports from pipe (fd={}): {} ({})",
                        child_to_parent_pipe.read_fd(),
                        std::strerror(errno),
                        errno));
      }

      HOLOSCAN_LOG_DEBUG(
          "unused_ports={} (size:{})", fmt::join(unused_ports, ","), unused_ports.size());

      int exit_code;
      waitpid(pid, &exit_code, 0);
      HOLOSCAN_LOG_DEBUG(
          "Child process exited with code {} ('{}')", exit_code, strerror(exit_code));

      return unused_ports;
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Error spawning child process: {}", e.what());
  }
  return {};
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
