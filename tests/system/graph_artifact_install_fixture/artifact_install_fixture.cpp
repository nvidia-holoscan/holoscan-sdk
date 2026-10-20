// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>

#if !defined(_WIN32)
#include <sys/stat.h>
#endif

namespace {

constexpr std::array<std::string_view, 4> kCanonicalArtifactFiles{
    "artifact-manifest.json",
    "endpoint-catalog.json",
    "execution-plan.json",
    "profile-manifest.json",
};

[[nodiscard]] bool write_file(const std::filesystem::path& path) {
  std::ofstream stream{path};
  stream << "{}\n";
  return stream.good();
}

}  // namespace

int main(int argc, char** argv) {
  std::error_code error;
  const auto executable_directory = std::filesystem::weakly_canonical(
      std::filesystem::absolute(argv[0], error).parent_path(), error);
  const auto working_directory =
      std::filesystem::weakly_canonical(std::filesystem::current_path(error), error);
  if (error || executable_directory != working_directory) {
    std::cerr << "artifact generation did not run from the executable directory\n";
    return 1;
  }

  std::filesystem::path artifact_root;
  std::string_view deployment_id;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "--artifact-root" && index + 1 < argc) {
      artifact_root = argv[++index];
    } else if (argument == "--deployment-id" && index + 1 < argc) {
      deployment_id = argv[++index];
    }
  }
  if (artifact_root.empty() || deployment_id.empty()) {
    std::cerr << "artifact root and deployment ID are required\n";
    return 1;
  }

  const std::filesystem::path graph_directory = artifact_root / "fixture-graph-id";
  const std::filesystem::path bundle_directory = graph_directory / deployment_id;
  std::filesystem::create_directories(bundle_directory, error);
  if (error) {
    std::cerr << "failed to create canonical bundle\n";
    return 1;
  }
  for (const std::string_view filename : kCanonicalArtifactFiles) {
    if (!write_file(bundle_directory / filename)) {
      std::cerr << "failed to write canonical artifact\n";
      return 1;
    }
  }

  constexpr std::string_view kFixtureMode{HOLOSCAN_ARTIFACT_FIXTURE_MODE};
  if (kFixtureMode == "extra_deployment") {
    const std::filesystem::path extra_deployment = graph_directory / "unexpected-deployment";
    std::filesystem::create_directories(extra_deployment, error);
    if (error || !write_file(extra_deployment / "unexpected.json")) {
      std::cerr << "failed to write extra deployment fixture\n";
      return 1;
    }
  } else if (kFixtureMode == "nested_directory") {
    const std::filesystem::path nested_directory = bundle_directory / "unexpected-directory";
    std::filesystem::create_directories(nested_directory, error);
    if (error || !write_file(nested_directory / "unexpected.json")) {
      std::cerr << "failed to write nested directory fixture\n";
      return 1;
    }
  } else if (kFixtureMode == "non_regular_file") {
#if defined(_WIN32)
    std::cerr << "non-regular-file fixture is unsupported on Windows\n";
    return 1;
#else
    const std::filesystem::path non_regular_file = bundle_directory / kCanonicalArtifactFiles[0];
    if (!std::filesystem::remove(non_regular_file, error) || error ||
        ::mkfifo(non_regular_file.c_str(), 0600) != 0) {
      std::cerr << "failed to write non-regular artifact fixture\n";
      return 1;
    }
#endif
  }
  return 0;
}
