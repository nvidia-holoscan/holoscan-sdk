/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "../support/optional_assertions.hpp"
#include "threading/linux_cpu_allowance.hpp"

namespace holoscan::detail {
namespace {

// REQ: THREAD-LINUX-011
class LinuxCPUAllowance : public ::testing::Test {
 protected:
  std::map<std::string, std::string> files{
      {"/proc/thread-self/cgroup", "0::/parent/leaf\n"},
      {"/proc/thread-self/mountinfo", "1 0 0:1 / /cg rw - cgroup2 cgroup rw\n"},
      {"/sys/devices/system/cpu/online", "0-7,12\n"},
      {"/cg/parent/leaf/cgroup.threads", "41\n42\n"},
      {"/cg/parent/leaf/cgroup.controllers", "cpuset memory\n"},
      {"/cg/parent/leaf/cpuset.cpus.effective", "1,3-5,12\n"},
  };

  expected<CPUAffinity, Error> resolve() {
    return read_linux_cpu_allowance(
        [&](const std::string& path) -> expected<std::string, Error> {
          const auto found = files.find(path);
          if (found == files.end()) {
            return make_unexpected(Error{ErrorCode::kFailure, "unreadable: " + path});
          }
          return found->second;
        },
        42);
  }
};

TEST_F(LinuxCPUAllowance, ReadsSparseV2EffectiveSetAndExcludesOfflineCPUs) {
  files["/sys/devices/system/cpu/online"] = "0-4,12\n";
  const auto result = resolve();
  ASSERT_TRUE(result) << result.error().message;
  EXPECT_EQ(result->cores, (std::vector<std::uint32_t>{1, 3, 4, 12}));
}

TEST_F(LinuxCPUAllowance, UsesEffectiveRatherThanConfiguredSet) {
  files["/cg/parent/leaf/cpuset.cpus"] = "";
  const auto result = resolve();
  ASSERT_TRUE(result);
  EXPECT_EQ(result->cores, (std::vector<std::uint32_t>{1, 3, 4, 5, 12}));
}

TEST_F(LinuxCPUAllowance, InheritsOnlyWhenControllerIsNotEnabled) {
  files.erase("/cg/parent/leaf/cpuset.cpus.effective");
  files["/cg/parent/leaf/cgroup.controllers"] = "memory\n";
  files["/cg/parent/cgroup.controllers"] = "cpuset memory\n";
  files["/cg/parent/cpuset.cpus.effective"] = "2-3\n";
  const auto result = resolve();
  ASSERT_TRUE(result);
  EXPECT_EQ(result->cores, (std::vector<std::uint32_t>{2, 3}));
}

TEST_F(LinuxCPUAllowance, DoesNotFallBackToAncestorForMissingEnabledControllerFile) {
  files.erase("/cg/parent/leaf/cpuset.cpus.effective");
  files["/cg/parent/cpuset.cpus.effective"] = "0-7\n";
  EXPECT_FALSE(resolve());
}

TEST_F(LinuxCPUAllowance, ResolvesNestedMountRoot) {
  files["/proc/thread-self/cgroup"] = "0::/host/container/parent/leaf\n";
  files["/proc/thread-self/mountinfo"] =
      "1 0 0:1 /host/container /cg rw shared:3 - cgroup2 cgroup rw\n";
  ASSERT_TRUE(resolve());
}

TEST_F(LinuxCPUAllowance, RequiresThreadVerificationForNonPrefixMountRoot) {
  // Synthetic path mapping, not qualification of an inherited namespace mount.
  files["/proc/thread-self/mountinfo"] = "1 0 0:1 /host/container /cg rw - cgroup2 cgroup rw\n";
  ASSERT_TRUE(resolve());
  files["/cg/parent/leaf/cgroup.threads"] = "41\n";
  EXPECT_FALSE(resolve());
}

TEST_F(LinuxCPUAllowance, RejectsInheritedAncestorMountRoot) {
  files["/proc/thread-self/mountinfo"] = "1 0 0:1 /.. /cg rw - cgroup2 cgroup rw\n";
  EXPECT_FALSE(resolve());
}

// REQ: THREAD-LINUX-011
TEST_F(LinuxCPUAllowance, SkipsNamespaceExternalRootsInEitherMountOrder) {
  const std::string usable = "1 0 0:1 / /cg rw shared:3 - cgroup2 cgroup rw\n";
  for (const auto* root : {"/..", "/../..", "/../sibling", "/../../other/leaf"}) {
    SCOPED_TRACE(root);
    const std::string external =
        std::string("2 0 0:1 ") + root + " /outside rw - cgroup2 cgroup rw\n";
    for (const auto& mounts : {external + usable, usable + external}) {
      SCOPED_TRACE(mounts);
      files["/proc/thread-self/mountinfo"] = mounts;
      const auto result = resolve();
      ASSERT_TRUE(result) << result.error().message;
      EXPECT_EQ(result->cores, (std::vector<std::uint32_t>{1, 3, 4, 5, 12}));
    }
    files["/proc/thread-self/mountinfo"] = external;
    // Even a granting synthetic directory cannot turn an external root into proof.
    files["/outside/parent/leaf/cgroup.threads"] = "42\n";
    files["/outside/parent/leaf/cgroup.controllers"] = "cpuset\n";
    files["/outside/parent/leaf/cpuset.cpus.effective"] = "0-7\n";
    EXPECT_FALSE(resolve());
    files["/proc/thread-self/mountinfo"] = external + usable;
    files["/cg/parent/leaf/cgroup.threads"] = "41\n";
    EXPECT_FALSE(resolve());
    files["/cg/parent/leaf/cgroup.threads"] = "41\n42\n";
  }
}

// REQ: THREAD-LINUX-011
TEST_F(LinuxCPUAllowance, RejectsMalformedExternalMountEvidence) {
  const std::string usable = "1 0 0:1 / /cg rw - cgroup2 cgroup rw\n";
  for (const auto* malformed : {"2 0 0:1 /.. /outside - cgroup2 cgroup rw\n",
                                "2 0 0:1 /.. /outside rw - cgroup2 cgroup\n",
                                "2 0 0:1 /.. /outside\\999 rw - cgroup2 cgroup rw\n",
                                "2 0 0:1 /../bad\\999 /outside rw - cgroup2 cgroup rw\n",
                                "2 0 0:1 /.././child /outside rw - cgroup2 cgroup rw\n",
                                "2 0 0:1 /..//child /outside rw - cgroup2 cgroup rw\n",
                                "2 0 0:1 /child/.. /outside rw - cgroup2 cgroup rw\n",
                                "2 0 0:1 /../child/.. /outside rw - cgroup2 cgroup rw\n",
                                "2 0 0:1 /.. /../outside rw - cgroup2 cgroup rw\n"}) {
    SCOPED_TRACE(malformed);
    files["/proc/thread-self/mountinfo"] = malformed + usable;
    EXPECT_FALSE(resolve());
  }
}

// REQ: THREAD-LINUX-011
TEST_F(LinuxCPUAllowance, PreservesDiscoveryReadErrors) {
  for (const auto* failed_path : {"/proc/thread-self/cgroup",
                                  "/proc/thread-self/mountinfo",
                                  "/cg/parent/leaf/cgroup.controllers",
                                  "/cg/parent/leaf/cpuset.cpus.effective",
                                  "/sys/devices/system/cpu/online"}) {
    SCOPED_TRACE(failed_path);
    const Error failure{
        ErrorCode::kPermissionDenied,
        std::string("cannot prove permitted CPU affinity: cannot read ") + failed_path,
        73};
    const auto result = read_linux_cpu_allowance(
        [&](const std::string& path) -> expected<std::string, Error> {
          if (path == failed_path) {
            return make_unexpected(failure);
          }
          return files.at(path);
        },
        42);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code, failure.code);
    EXPECT_EQ(result.error().message, failure.message);
    EXPECT_EQ(result.error().provenance, failure.provenance);
  }
}

// REQ: THREAD-LINUX-011
TEST_F(LinuxCPUAllowance, DistinguishesMembershipRereadFailureFromMigration) {
  for (const bool read_fails : {true, false}) {
    SCOPED_TRACE(read_fails);
    int membership_reads = 0;
    const Error failure{
        ErrorCode::kInterrupted,
        "cannot prove permitted CPU affinity: incomplete read: /proc/thread-self/cgroup",
        91};
    const auto result = read_linux_cpu_allowance(
        [&](const std::string& path) -> expected<std::string, Error> {
          if (path == "/proc/thread-self/cgroup" && ++membership_reads == 2) {
            if (read_fails) {
              return make_unexpected(failure);
            }
            return std::string("0::/changed\n");
          }
          return files.at(path);
        },
        42);
    ASSERT_FALSE(result);
    EXPECT_EQ(membership_reads, 2);
    if (read_fails) {
      EXPECT_EQ(result.error().code, failure.code);
      EXPECT_EQ(result.error().message, failure.message);
      EXPECT_EQ(result.error().provenance, failure.provenance);
    } else {
      EXPECT_EQ(result.error().code, ErrorCode::kNotReady);
      EXPECT_EQ(
          result.error().message,
          "cannot prove permitted CPU affinity: thread cgroup membership changed during discovery");
    }
  }
}

// REQ: THREAD-LINUX-011
TEST_F(LinuxCPUAllowance, SkipsUnverifiedCandidateWithoutGrantingPermission) {
  files["/proc/thread-self/mountinfo"] =
      "2 0 0:1 / /unreadable rw - cgroup2 cgroup rw\n" + files["/proc/thread-self/mountinfo"];
  ASSERT_TRUE(resolve());
  files.erase("/cg/parent/leaf/cgroup.threads");
  EXPECT_FALSE(resolve());
}

TEST_F(LinuxCPUAllowance, DecodesMountEscapes) {
  files["/proc/thread-self/mountinfo"] = "1 0 0:1 / /cgroup\\040mount rw - cgroup2 cgroup rw\n";
  files["/cgroup mount/parent/leaf/cgroup.threads"] = "42\n";
  files["/cgroup mount/parent/leaf/cgroup.controllers"] = "cpuset\n";
  files["/cgroup mount/parent/leaf/cpuset.cpus.effective"] = "3\n";
  const auto result = resolve();
  ASSERT_TRUE(result);
  EXPECT_EQ(result->cores, (std::vector<std::uint32_t>{3}));
}

TEST_F(LinuxCPUAllowance, ReadsV1EffectiveSetInHybridHierarchy) {
  files["/proc/thread-self/cgroup"] = "0::/unified\n5:cpu,cpuset:/parent/leaf\n";
  files["/proc/thread-self/mountinfo"] =
      "1 0 0:1 / /cg rw - cgroup cgroup rw,cpu,cpuset\n"
      "2 0 0:2 / /v2 rw - cgroup2 cgroup rw\n";
  files["/cg/parent/leaf/tasks"] = "42\n";
  files["/cg/parent/leaf/cpuset.effective_cpus"] = "2,6\n";
  const auto result = resolve();
  ASSERT_TRUE(result);
  EXPECT_EQ(result->cores, (std::vector<std::uint32_t>{2, 6}));
}

TEST_F(LinuxCPUAllowance, DoesNotGuessV1ConfiguredAllowance) {
  files["/proc/thread-self/cgroup"] = "5:cpuset:/parent/leaf\n";
  files["/proc/thread-self/mountinfo"] = "1 0 0:1 / /cg rw - cgroup cgroup rw,cpuset\n";
  files["/cg/parent/leaf/tasks"] = "42\n";
  files["/cg/parent/leaf/cpuset.cpus"] = "0-7\n";
  EXPECT_FALSE(resolve());
}

TEST_F(LinuxCPUAllowance, RejectsUnsupportedV1NoprefixAllowance) {
  files["/proc/thread-self/cgroup"] = "5:cpuset:/parent/leaf\n";
  files["/proc/thread-self/mountinfo"] = "1 0 0:1 / /cg rw - cgroup cgroup rw,cpuset,noprefix\n";
  files["/cg/parent/leaf/tasks"] = "42\n";
  files["/cg/parent/leaf/effective_cpus"] = "2,6\n";
  EXPECT_FALSE(resolve());
}

TEST_F(LinuxCPUAllowance, RejectsMalformedListsAndEmptyEffectiveSet) {
  for (const auto* value : {"", "1,", "-1", "3-1", "1--3", "1 2", "0-999999999", "1x"}) {
    SCOPED_TRACE(value);
    files["/cg/parent/leaf/cpuset.cpus.effective"] = value;
    EXPECT_FALSE(resolve());
  }
}

TEST_F(LinuxCPUAllowance, RejectsMountinfoWithoutMandatoryMountOptions) {
  files["/proc/thread-self/mountinfo"] = "1 0 0:1 / /cg - cgroup2 cgroup rw\n";
  EXPECT_FALSE(resolve());
}

TEST_F(LinuxCPUAllowance, RejectsMissingAndMalformedDiscovery) {
  const auto original = files;
  for (const auto* path : {"/proc/thread-self/cgroup",
                           "/proc/thread-self/mountinfo",
                           "/sys/devices/system/cpu/online",
                           "/cg/parent/leaf/cgroup.controllers",
                           "/cg/parent/leaf/cgroup.threads"}) {
    SCOPED_TRACE(path);
    files = original;
    files.erase(path);
    EXPECT_FALSE(resolve());
  }
  for (const auto* membership :
       {"", "broken", "0::/../escape\n", "0::relative\n", "0::/parent/leaf\n0::/other\n"}) {
    files = original;
    files["/proc/thread-self/cgroup"] = membership;
    EXPECT_FALSE(resolve());
  }
}

TEST_F(LinuxCPUAllowance, RequiresIndependentPermissionEvidenceAtMountRoot) {
  files["/proc/thread-self/cgroup"] = "0::/\n";
  files["/cg/cgroup.threads"] = "42\n";
  files["/cg/cgroup.controllers"] = "memory\n";
  EXPECT_FALSE(resolve());
  files["/cg/cpuset.cpus.effective"] = "2-3\n";
  ASSERT_TRUE(resolve());
}

// Unpinned helper threads avoid the kernel-isolated CPUs that sysfs lists; a missing or empty list
// means none are isolated.
// REQ: THREAD-LINUX-012
TEST(LinuxIsolatedCPUs, ReadsTheSysfsListAndTreatsMissingOrEmptyAsNone) {
  std::map<std::string, std::string> files;
  const auto read = [&files]() {
    return read_linux_isolated_cpus(
        [&files](const std::string& path) -> expected<std::string, Error> {
          const auto found = files.find(path);
          if (found == files.end()) {
            return make_unexpected(Error{ErrorCode::kFailure, "unreadable: " + path});
          }
          return found->second;
        });
  };

  auto isolated = read();
  HOLOSCAN_ASSERT_HAS_VALUE(isolated);
  EXPECT_TRUE(isolated->cores.empty());

  files["/sys/devices/system/cpu/isolated"] = "\n";
  isolated = read();
  HOLOSCAN_ASSERT_HAS_VALUE(isolated);
  EXPECT_TRUE(isolated->cores.empty());

  files["/sys/devices/system/cpu/isolated"] = "2-4,9\n";
  isolated = read();
  HOLOSCAN_ASSERT_HAS_VALUE(isolated);
  EXPECT_EQ(isolated->cores, (std::vector<std::uint32_t>{2U, 3U, 4U, 9U}));

  files["/sys/devices/system/cpu/isolated"] = "2-x\n";
  EXPECT_FALSE(read().has_value());
}

}  // namespace
}  // namespace holoscan::detail
