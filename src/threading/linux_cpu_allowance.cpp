/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "linux_cpu_allowance.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace holoscan::detail {
namespace {

unexpected<Error> unproven(const std::string& detail) {
  return make_unexpected(
      Error{ErrorCode::kNotReady, "cannot prove permitted CPU affinity: " + detail});
}

bool has_token(std::string_view text, std::string_view token, char separator) {
  std::size_t begin = 0;
  while (begin < text.size()) {
    const auto end = text.find(separator, begin);
    if (text.substr(begin, end - begin) == token) {
      return true;
    }
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1;
  }
  return false;
}

bool absolute_cgroup_path(std::string_view path) {
  return !path.empty() && path.front() == '/' && !has_token(path, "..", '/') &&
         !has_token(path, ".", '/') && path.find("//") == std::string_view::npos;
}

std::string child_path(const std::string& parent, const std::string& child) {
  return parent == "/" ? "/" + child : parent + "/" + child;
}

std::optional<std::uint32_t> number(std::string_view text) {
  std::uint32_t value{};
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (error != std::errc() || end != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

expected<CPUAffinity, Error> cpu_list(std::string text) {
  // Kernel CPU lists may have a terminal newline, but no signs or internal whitespace.
  if (!text.empty() && text.back() == '\n') {
    text.pop_back();
  }
  if (text.empty() || text.back() == ',') {
    return unproven("empty or malformed CPU list");
  }
#ifdef __linux__
  constexpr std::uint32_t kMaskSize = CPU_SETSIZE;
#else
  constexpr std::uint32_t kMaskSize = 1024;
#endif
  std::bitset<kMaskSize> cpus;
  std::istringstream items(text);
  std::string item;
  while (std::getline(items, item, ',')) {
    const auto dash = item.find('-');
    const auto first = number(std::string_view(item).substr(0, dash));
    const auto last = dash == std::string::npos ? first : number(item.substr(dash + 1));
    // Match the fixed mask used by the Linux affinity apply/readback backend.
    if (!first || !last || *first > *last || *last >= kMaskSize) {
      return unproven("malformed or unsupported CPU list: " + item);
    }
    for (auto cpu = *first; cpu <= *last; ++cpu) {
      cpus.set(cpu);
    }
  }
  CPUAffinity result;
  for (std::uint32_t cpu = 0; cpu < kMaskSize; ++cpu) {
    if (cpus.test(cpu)) {
      result.cores.push_back(cpu);
    }
  }
  return result;
}

struct Membership {
  std::string path;
  bool unified;
};

expected<Membership, Error> membership(const std::string& text) {
  std::optional<Membership> v1;
  std::optional<Membership> v2;
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    const auto first = line.find(':');
    const auto second = line.find(':', first == std::string::npos ? 0 : first + 1);
    if (first == std::string::npos || second == std::string::npos ||
        !number(std::string_view(line).substr(0, first))) {
      return unproven("malformed thread cgroup membership");
    }
    const auto controllers = std::string_view(line).substr(first + 1, second - first - 1);
    const auto path = line.substr(second + 1);
    if (!absolute_cgroup_path(path)) {
      return unproven("invalid cgroup membership path");
    }
    const bool unified = line.substr(0, first) == "0" && controllers.empty();
    if (unified || has_token(controllers, "cpuset", ',')) {
      auto& selected = unified ? v2 : v1;
      if (selected) {
        return unproven("ambiguous thread cgroup membership");
      }
      selected = Membership{path, unified};
    }
  }
  // On hybrid hosts the cpuset controller can belong to the v1 hierarchy.
  if (v1) {
    return *v1;
  }
  if (v2) {
    return *v2;
  }
  return unproven("no cpuset or unified thread cgroup membership");
}

// Cgroup namespaces render inherited roots outside their boundary with leading
// parent components. Recognize that kernel spelling without resolving hidden ancestors.
bool namespace_external_root(std::string_view path) {
  if (path != "/.." && !path.starts_with("/../")) {
    return false;
  }
  while (path.starts_with("/../")) {
    path.remove_prefix(3);
  }
  return path == "/.." || absolute_cgroup_path(path);
}

expected<std::string, Error> mount_path(std::string_view encoded, bool is_root = false) {
  std::string decoded;
  for (std::size_t index = 0; index < encoded.size(); ++index) {
    if (encoded[index] != '\\') {
      decoded += encoded[index];
      continue;
    }
    const auto escape = encoded.substr(index, 4);
    if (escape == "\\040") {
      decoded += ' ';
    } else if (escape == "\\011") {
      decoded += '\t';
    } else if (escape == "\\012") {
      decoded += '\n';
    } else if (escape == "\\134") {
      decoded += '\\';
    } else {
      return unproven("invalid mountinfo path escape");
    }
    index += 3;
  }
  if (!absolute_cgroup_path(decoded) && !(is_root && namespace_external_root(decoded))) {
    return unproven("invalid cgroup mount path");
  }
  return decoded;
}

struct CgroupDirectory {
  std::string path;
  std::string mount;
};

bool contains_thread(const std::string& text, std::uint64_t thread_id) {
  std::istringstream ids(text);
  std::uint64_t id{};
  bool found = false;
  while (ids >> id) {
    found = found || id == thread_id;
  }
  return ids.eof() && found;
}

struct CgroupMount {
  std::string root;
  std::string path;
};

// Separate row syntax/filesystem selection from membership-to-directory resolution.
expected<std::optional<CgroupMount>, Error> cgroup_mount(const std::string& line, bool unified) {
  const auto separator = line.find(" - ");
  if (separator == std::string::npos) {
    return unproven("malformed mountinfo");
  }
  std::istringstream tail(line.substr(separator + 3));
  std::string type;
  std::string source;
  std::string options;
  if (!(tail >> type >> source >> options)) {
    return unproven("malformed mountinfo filesystem fields");
  }
  const bool matches_hierarchy =
      unified ? type == "cgroup2" : type == "cgroup" && has_token(options, "cpuset", ',');
  if (!matches_hierarchy) {
    return std::optional<CgroupMount>{};
  }
  std::istringstream fields(line.substr(0, separator));
  std::string ignored;
  std::string root_text;
  std::string mount_text;
  std::string mount_options;
  if (!(fields >> ignored >> ignored >> ignored >> root_text >> mount_text >> mount_options)) {
    return unproven("malformed mountinfo mount fields");
  }
  auto root = mount_path(root_text, true);
  auto mount = mount_path(mount_text);
  if (!root || !mount) {
    return unproven("invalid cgroup mount root or mountpoint");
  }
  // A valid namespace-external root is unusable, not a malformed observation.
  // Keep scanning for a visible mount whose directory independently verifies this TID.
  if (namespace_external_root(*root)) {
    return std::optional<CgroupMount>{};
  }
  return std::optional{CgroupMount{std::move(*root), std::move(*mount)}};
}

expected<CgroupDirectory, Error> cgroup_directory(const std::string& mounts,
                                                  const Membership& group,
                                                  const CPUAllowanceFileReader& read_file,
                                                  std::uint64_t thread_id) {
  std::istringstream lines(mounts);
  std::string line;
  while (std::getline(lines, line)) {
    const auto candidate = cgroup_mount(line, group.unified);
    if (!candidate) {
      return forward_error(candidate);
    }
    const auto& selected = *candidate;
    if (!selected) {
      continue;
    }
    const auto& root = selected->root;
    const auto& mount = selected->path;
    std::string relative = group.path;
    if (group.path == root) {
      relative = "/";
    } else if (root != "/" && group.path.starts_with(root + "/")) {
      relative = group.path.substr(root.size());
    }
    // Both proc views are relative to the reader's cgroup namespace. Path mapping
    // alone is not permission evidence: the candidate directory must contain this TID.
    const auto directory = relative == "/" ? mount : child_path(mount, relative.substr(1));
    const auto tids = read_file(child_path(directory, group.unified ? "cgroup.threads" : "tasks"));
    if (tids && contains_thread(*tids, thread_id)) {
      return CgroupDirectory{directory, mount};
    }
  }
  return unproven("no mounted cgroup directory verifies the calling thread");
}

expected<CPUAffinity, Error> effective_cpuset(CgroupDirectory directory, bool unified,
                                              const CPUAllowanceFileReader& read_file) {
  while (true) {
    if (unified && directory.path != directory.mount) {
      const auto controllers = read_file(child_path(directory.path, "cgroup.controllers"));
      if (!controllers) {
        return forward_error(controllers);
      }
      std::istringstream words(*controllers);
      std::string controller;
      bool cpuset_enabled = false;
      while (words >> controller) {
        cpuset_enabled = cpuset_enabled || controller == "cpuset";
      }
      if (!cpuset_enabled) {
        // A disabled controller inherits its nearest enabled ancestor. Stop at
        // the visible mount root; hidden ancestors cannot prove permission.
        directory.path = directory.path.substr(0, directory.path.rfind('/'));
        if (directory.path.empty()) {
          directory.path = "/";
        }
        continue;
      }
    }
    const auto path =
        child_path(directory.path, unified ? "cpuset.cpus.effective" : "cpuset.effective_cpus");
    const auto cpus = read_file(path);
    if (!cpus) {
      return forward_error(cpus);
    }
    return cpu_list(*cpus);
  }
}

#ifdef __linux__
expected<std::string, Error> read_file(const std::string& path) {
  std::ifstream stream(path);
  if (!stream) {
    return unproven("cannot read " + path);
  }
  constexpr std::size_t kMaximumBytes = 1024 * 1024;
  std::array<char, 4096> buffer{};
  std::string result;
  while (stream.read(buffer.data(), buffer.size()) || stream.gcount() > 0) {
    result.append(buffer.data(), static_cast<std::size_t>(stream.gcount()));
    if (result.size() > kMaximumBytes) {
      return unproven("observation exceeds size limit: " + path);
    }
  }
  if (!stream.eof()) {
    return unproven("incomplete read: " + path);
  }
  return result;
}
#endif

}  // namespace

expected<CPUAffinity, Error> read_linux_cpu_allowance(const CPUAllowanceFileReader& reader,
                                                      std::uint64_t thread_id) {
  const auto groups = reader("/proc/thread-self/cgroup");
  if (!groups) {
    return forward_error(groups);
  }
  const auto mounts = reader("/proc/thread-self/mountinfo");
  if (!mounts) {
    return forward_error(mounts);
  }
  const auto group = membership(*groups);
  if (!group) {
    return forward_error(group);
  }
  const auto directory = cgroup_directory(*mounts, *group, reader, thread_id);
  if (!directory) {
    return forward_error(directory);
  }
  const auto allowed = effective_cpuset(*directory, group->unified, reader);
  if (!allowed) {
    return forward_error(allowed);
  }
  const auto online_text = reader("/sys/devices/system/cpu/online");
  if (!online_text) {
    return forward_error(online_text);
  }
  const auto online = cpu_list(*online_text);
  if (!online) {
    return forward_error(online);
  }
  // Detect migration during discovery. Runtime still guards subsequent changes.
  const auto final_groups = reader("/proc/thread-self/cgroup");
  if (!final_groups) {
    return forward_error(final_groups);
  }
  if (*final_groups != *groups) {
    return unproven("thread cgroup membership changed during discovery");
  }
  CPUAffinity result;
  std::ranges::set_intersection(allowed->cores, online->cores, std::back_inserter(result.cores));
  return result;
}

expected<CPUAffinity, Error> read_linux_isolated_cpus(const CPUAllowanceFileReader& reader) {
  auto text = reader("/sys/devices/system/cpu/isolated");
  if (!text) {
    return CPUAffinity{};
  }
  while (!text->empty() && (text->back() == '\n' || text->back() == ' ')) {
    text->pop_back();
  }
  if (text->empty()) {
    return CPUAffinity{};
  }
  return cpu_list(*text);
}

expected<CPUAffinity, Error> read_linux_isolated_cpus() {
#ifdef __linux__
  return read_linux_isolated_cpus(read_file);
#else
  return CPUAffinity{};
#endif
}

expected<CPUAffinity, Error> read_linux_cpu_allowance() {
#ifdef __linux__
  return read_linux_cpu_allowance(read_file, static_cast<std::uint64_t>(syscall(SYS_gettid)));
#else
  return unproven("Linux CPU discovery is unsupported on this platform");
#endif
}

}  // namespace holoscan::detail
