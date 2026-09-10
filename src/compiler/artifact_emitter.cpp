/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoscan/core/compile.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winternl.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#endif

#include <holoscan/core/parameter_schema.hpp>

#include "artifact_emitter_test_access.hpp"
#include "core/canonical_text_encoding.hpp"
#include "execution_plan_state.hpp"

namespace holoscan {
namespace {

constexpr std::string_view kArtifactManifestSchema{"holoscan.artifact_manifest.v1"};
constexpr std::size_t kFileReadChunkSize = 4096U;
constexpr std::size_t kMaximumTemporaryAttempts = 128U;
constexpr std::size_t kMaximumManifestTimestampCharacters =
    std::numeric_limits<std::int64_t>::digits10 + 2U;
constexpr std::int64_t kNoInjectedReplaceFailure = -1;
std::atomic<std::int64_t> replace_failure_countdown{kNoInjectedReplaceFailure};
std::atomic<bool> filesystem_exception_pending{};
std::atomic<bool> temporary_oversize_pending{};

struct PendingFile {
  std::filesystem::path destination;
  std::filesystem::path temporary;
  std::string bytes;
  bool changed{};
};

[[nodiscard]] ParamSchemaHash hash_bytes(std::string_view bytes) noexcept {
  detail::StableHashBuilder hash;
  for (const unsigned char byte : bytes) {
    hash.add_byte(byte);
  }
  return hash.finish();
}

enum class BoundedReadResult : std::uint8_t { kComplete, kOversized, kMissing, kFailure };
enum class ExclusiveWriteResult : std::uint8_t { kSuccess, kCollision, kFailure };

[[nodiscard]] bool valid_deployment_id(std::string_view value) {
  constexpr std::size_t kMaximumDeploymentIdLength = 128U;
  if (value.empty() || value.size() > kMaximumDeploymentIdLength || value == "." || value == "..") {
    return false;
  }
  const auto ascii_alphanumeric = [](unsigned char character) noexcept {
    return (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') ||
           (character >= '0' && character <= '9');
  };
  if (!ascii_alphanumeric(static_cast<unsigned char>(value.front())) ||
      !std::all_of(value.begin(),
                   value.end(),
                   [&](unsigned char character) {
                     return ascii_alphanumeric(character) || character == '-' || character == '_' ||
                            character == '.';
                   }) ||
      value.back() == '.') {
    return false;
  }
  const std::filesystem::path component{value};
  if (component.has_root_name() || component.has_root_directory() || component.has_parent_path() ||
      component.filename().generic_string() != value) {
    return false;
  }

  // Windows reserves these basenames even when they carry an extension. Reject
  // them on every platform so an artifact bundle has one portable location.
  const std::string_view basename = value.substr(0U, value.find('.'));
  std::string uppercase;
  uppercase.reserve(basename.size());
  std::transform(
      basename.begin(), basename.end(), std::back_inserter(uppercase), [](unsigned char character) {
        return static_cast<char>(character >= 'a' && character <= 'z' ? character - ('a' - 'A')
                                                                      : character);
      });
  if (uppercase == "CON" || uppercase == "PRN" || uppercase == "AUX" || uppercase == "NUL") {
    return false;
  }
  const bool reserved_numbered = uppercase.size() == 4U && uppercase[3] >= '1' &&
                                 uppercase[3] <= '9' &&
                                 (uppercase.starts_with("COM") || uppercase.starts_with("LPT"));
  return !reserved_numbered;
}

[[nodiscard]] std::filesystem::path artifact_root(const ArtifactEmitOptions& options) {
  if (!options.root.empty()) {
    return options.root;
  }
  if (const char* environment = std::getenv("HOLOSCAN_ARTIFACT_DIR");
      environment != nullptr && *environment != '\0') {
    return std::filesystem::path{environment};
  }
  return std::filesystem::current_path() / "holoscan-artifacts";
}

[[nodiscard]] std::optional<std::size_t> skip_json_string(std::string_view json,
                                                          std::size_t begin) noexcept {
  if (begin >= json.size() || json[begin] != '"') {
    return std::nullopt;
  }
  bool escaped = false;
  for (std::size_t index = begin + 1U; index < json.size(); ++index) {
    if (escaped) {
      escaped = false;
    } else if (json[index] == '\\') {
      escaped = true;
    } else if (json[index] == '"') {
      return index + 1U;
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<std::size_t> skip_json_value(std::string_view json, std::size_t begin) {
  if (begin >= json.size()) {
    return std::nullopt;
  }
  if (json[begin] == '"') {
    return skip_json_string(json, begin);
  }
  if (json[begin] != '{' && json[begin] != '[') {
    std::size_t end = begin;
    while (end < json.size() && json[end] != ',' && json[end] != '}' && json[end] != ']') {
      ++end;
    }
    return end == begin ? std::nullopt : std::optional<std::size_t>{end};
  }

  std::vector<char> closing;
  closing.reserve(8U);
  closing.push_back(json[begin] == '{' ? '}' : ']');
  for (std::size_t index = begin + 1U; index < json.size(); ++index) {
    if (json[index] == '"') {
      auto string_end = skip_json_string(json, index);
      if (!string_end) {
        return std::nullopt;
      }
      index = *string_end - 1U;
    } else if (json[index] == '{') {
      closing.push_back('}');
    } else if (json[index] == '[') {
      closing.push_back(']');
    } else if (!closing.empty() && json[index] == closing.back()) {
      closing.pop_back();
      if (closing.empty()) {
        return index + 1U;
      }
    }
  }
  return std::nullopt;
}

/// Extract one exact top-level object from the trusted canonical plan document.
///
/// This deliberately walks the root JSON members instead of searching for a
/// matching substring, because the same key may appear in nested plan records.
[[nodiscard]] std::optional<std::string> extract_top_level_object(std::string_view json,
                                                                  std::string_view key) {
  if (json.size() < 2U || json.front() != '{' || json.back() != '}') {
    return std::nullopt;
  }
  std::size_t position = 1U;
  while (position < json.size() - 1U) {
    const std::size_t key_begin = position;
    auto key_end = skip_json_string(json, key_begin);
    if (!key_end || *key_end >= json.size() || json[*key_end] != ':') {
      return std::nullopt;
    }
    const std::string_view member_key = json.substr(key_begin + 1U, *key_end - key_begin - 2U);
    const std::size_t value_begin = *key_end + 1U;
    auto value_end = skip_json_value(json, value_begin);
    if (!value_end) {
      return std::nullopt;
    }
    if (member_key == key) {
      if (json[value_begin] != '{') {
        return std::nullopt;
      }
      return std::string{json.substr(value_begin, *value_end - value_begin)};
    }
    position = *value_end;
    if (position < json.size() - 1U) {
      if (json[position] != ',') {
        return std::nullopt;
      }
      ++position;
    }
  }
  return std::nullopt;
}

[[nodiscard]] std::uint64_t process_identity() noexcept {
#if defined(_WIN32)
  return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
  return static_cast<std::uint64_t>(::getpid());
#endif
}

[[nodiscard]] std::filesystem::path temporary_candidate(const std::filesystem::path& destination) {
  static std::atomic<std::uint64_t> next_temporary{1U};
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  std::filesystem::path candidate = destination;
  candidate += ".tmp-" + std::to_string(process_identity()) + "-" + std::to_string(tick) + "-" +
               std::to_string(next_temporary.fetch_add(1U, std::memory_order_relaxed));
  return candidate;
}

class ArtifactDirectory {
 public:
  ArtifactDirectory(const ArtifactDirectory&) = delete;
  ArtifactDirectory& operator=(const ArtifactDirectory&) = delete;

  ArtifactDirectory(ArtifactDirectory&& other) noexcept : handle_(other.release()) {}

  ArtifactDirectory& operator=(ArtifactDirectory&& other) noexcept {
    if (this != &other) {
      close();
      handle_ = other.release();
    }
    return *this;
  }

  ~ArtifactDirectory() { close(); }

  [[nodiscard]] static expected<ArtifactDirectory, Error> open_or_create(
      const std::filesystem::path& directory) {
    std::error_code error;
    const std::filesystem::path absolute =
        std::filesystem::absolute(directory, error).lexically_normal();
    if (error || absolute.empty() || absolute.root_path().empty()) {
      return make_unexpected(Error{ErrorCode::kFailure, "ARTIFACT_DIRECTORY_CREATE_FAILED"});
    }
#if defined(_WIN32)
    HANDLE current = CreateFileW(absolute.root_path().c_str(),
                                 FILE_TRAVERSE | FILE_READ_ATTRIBUTES | SYNCHRONIZE,
                                 FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                 nullptr,
                                 OPEN_EXISTING,
                                 FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT,
                                 nullptr);
    if (!safe_native_handle(current, true)) {
      close_native(current);
      return make_unexpected(Error{ErrorCode::kFailure, "ARTIFACT_DIRECTORY_CREATE_FAILED"});
    }
    for (const auto& component : absolute.relative_path()) {
      if (component == ".") {
        continue;
      }
      if (component == "..") {
        close_native(current);
        return make_unexpected(Error{ErrorCode::kFailure, "ARTIFACT_DIRECTORY_CREATE_FAILED"});
      }
      NTSTATUS status{};
      HANDLE child = open_relative(
          current,
          component,
          FILE_TRAVERSE | FILE_READ_ATTRIBUTES | SYNCHRONIZE,
          FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
          FILE_OPEN_IF,
          FILE_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT | FILE_OPEN_REPARSE_POINT,
          status);
      if (!nt_success(status) || !safe_native_handle(child, true)) {
        close_native(child);
        close_native(current);
        return make_unexpected(Error{ErrorCode::kFailure, "ARTIFACT_DIRECTORY_CREATE_FAILED"});
      }
      close_native(current);
      current = child;
    }
#else
    int current =
        ::open(absolute.root_path().c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (current < 0) {
      return make_unexpected(Error{ErrorCode::kFailure, "ARTIFACT_DIRECTORY_CREATE_FAILED"});
    }
    for (const auto& component : absolute.relative_path()) {
      if (component == ".") {
        continue;
      }
      if (component == ".." ||
          (::mkdirat(current, component.c_str(), 0700) != 0 && errno != EEXIST)) {
        ::close(current);
        return make_unexpected(Error{ErrorCode::kFailure, "ARTIFACT_DIRECTORY_CREATE_FAILED"});
      }
      const int child =
          ::openat(current, component.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
      if (child < 0) {
        ::close(current);
        return make_unexpected(Error{ErrorCode::kFailure, "ARTIFACT_DIRECTORY_CREATE_FAILED"});
      }
      ::close(current);
      current = child;
    }
#endif
    return ArtifactDirectory{current};
  }

  // Read one byte beyond the trusted size so oversized files are bounded.
  [[nodiscard]] BoundedReadResult read(const std::filesystem::path& name, std::size_t expected_size,
                                       std::string& bytes) const noexcept {
    bytes.clear();
    if (expected_size == std::numeric_limits<std::size_t>::max()) {
      return BoundedReadResult::kFailure;
    }
#if defined(_WIN32)
    NTSTATUS status{};
    HANDLE file = open_relative(
        handle_,
        name,
        FILE_GENERIC_READ | SYNCHRONIZE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        FILE_OPEN,
        FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT | FILE_OPEN_REPARSE_POINT,
        status);
    if (!nt_success(status)) {
      return nt_missing(status) ? BoundedReadResult::kMissing : BoundedReadResult::kFailure;
    }
    if (!safe_native_handle(file, false)) {
      close_native(file);
      return BoundedReadResult::kFailure;
    }
    const std::size_t read_limit = expected_size + 1U;
    std::array<char, kFileReadChunkSize> chunk{};
    while (bytes.size() < read_limit) {
      const DWORD request = static_cast<DWORD>(std::min(chunk.size(), read_limit - bytes.size()));
      DWORD count{};
      if (ReadFile(file, chunk.data(), request, &count, nullptr) == 0) {
        close_native(file);
        return BoundedReadResult::kFailure;
      }
      if (count == 0U) {
        close_native(file);
        return BoundedReadResult::kComplete;
      }
      bytes.append(chunk.data(), count);
    }
    close_native(file);
#else
    const int file = ::openat(handle_, name.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (file < 0) {
      return errno == ENOENT ? BoundedReadResult::kMissing : BoundedReadResult::kFailure;
    }
    struct stat status{};
    if (::fstat(file, &status) != 0 || !S_ISREG(status.st_mode)) {
      ::close(file);
      return BoundedReadResult::kFailure;
    }
    const std::size_t read_limit = expected_size + 1U;
    std::array<char, kFileReadChunkSize> chunk{};
    while (bytes.size() < read_limit) {
      const std::size_t request = std::min(chunk.size(), read_limit - bytes.size());
      const ssize_t count = ::read(file, chunk.data(), request);
      if (count < 0 && errno == EINTR) {
        continue;
      }
      if (count < 0) {
        ::close(file);
        return BoundedReadResult::kFailure;
      }
      if (count == 0) {
        ::close(file);
        return BoundedReadResult::kComplete;
      }
      bytes.append(chunk.data(), static_cast<std::size_t>(count));
    }
    ::close(file);
#endif
    return BoundedReadResult::kOversized;
  }

  [[nodiscard]] ExclusiveWriteResult write_exclusive(const std::filesystem::path& name,
                                                     std::string_view bytes) const noexcept {
#if defined(_WIN32)
    NTSTATUS status{};
    HANDLE file = open_relative(
        handle_,
        name,
        FILE_GENERIC_WRITE | SYNCHRONIZE,
        0U,
        FILE_CREATE,
        FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT | FILE_OPEN_REPARSE_POINT,
        status);
    if (!nt_success(status)) {
      return nt_collision(status) ? ExclusiveWriteResult::kCollision
                                  : ExclusiveWriteResult::kFailure;
    }
    if (!safe_native_handle(file, false)) {
      close_native(file);
      return ExclusiveWriteResult::kFailure;
    }
    std::size_t offset{};
    bool succeeded = true;
    while (offset < bytes.size()) {
      const DWORD remaining = static_cast<DWORD>(
          std::min<std::size_t>(bytes.size() - offset, static_cast<std::size_t>(MAXDWORD)));
      DWORD written{};
      if (WriteFile(file, bytes.data() + offset, remaining, &written, nullptr) == 0 ||
          written == 0U) {
        succeeded = false;
        break;
      }
      offset += written;
    }
    succeeded = succeeded && FlushFileBuffers(file) != 0;
    succeeded = CloseHandle(file) != 0 && succeeded;
#else
    const int file =
        ::openat(handle_, name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (file < 0) {
      return errno == EEXIST ? ExclusiveWriteResult::kCollision : ExclusiveWriteResult::kFailure;
    }
    std::size_t offset{};
    bool succeeded = true;
    while (offset < bytes.size()) {
      const ssize_t written = ::write(file, bytes.data() + offset, bytes.size() - offset);
      if (written < 0 && errno == EINTR) {
        continue;
      }
      if (written <= 0) {
        succeeded = false;
        break;
      }
      offset += static_cast<std::size_t>(written);
    }
    succeeded = succeeded && ::fsync(file) == 0;
    succeeded = ::close(file) == 0 && succeeded;
#endif
    return succeeded ? ExclusiveWriteResult::kSuccess : ExclusiveWriteResult::kFailure;
  }

  [[nodiscard]] bool append_sentinel(const std::filesystem::path& name) const noexcept {
#if defined(_WIN32)
    NTSTATUS status{};
    HANDLE file = open_relative(
        handle_,
        name,
        FILE_APPEND_DATA | SYNCHRONIZE,
        0U,
        FILE_OPEN,
        FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT | FILE_OPEN_REPARSE_POINT,
        status);
    if (!nt_success(status) || !safe_native_handle(file, false)) {
      close_native(file);
      return false;
    }
    constexpr char kSentinel = '!';
    DWORD written{};
    const bool succeeded = WriteFile(file, &kSentinel, 1U, &written, nullptr) != 0 &&
                           written == 1U && FlushFileBuffers(file) != 0;
    return CloseHandle(file) != 0 && succeeded;
#else
    const int file = ::openat(handle_, name.c_str(), O_WRONLY | O_APPEND | O_NOFOLLOW | O_CLOEXEC);
    if (file < 0) {
      return false;
    }
    constexpr char kSentinel = '!';
    const bool succeeded = ::write(file, &kSentinel, 1U) == 1 && ::fsync(file) == 0;
    return ::close(file) == 0 && succeeded;
#endif
  }

  void remove(const std::filesystem::path& name) const noexcept {
#if defined(_WIN32)
    NTSTATUS status{};
    HANDLE file = open_relative(
        handle_,
        name,
        DELETE | FILE_READ_ATTRIBUTES | SYNCHRONIZE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        FILE_OPEN,
        FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT | FILE_OPEN_REPARSE_POINT,
        status);
    if (!nt_success(status) || !safe_native_handle(file, false)) {
      close_native(file);
      return;
    }
    FILE_DISPOSITION_INFO disposition{TRUE};
    static_cast<void>(
        SetFileInformationByHandle(file, FileDispositionInfo, &disposition, sizeof(disposition)));
    close_native(file);
#else
    static_cast<void>(::unlinkat(handle_, name.c_str(), 0));
#endif
  }

  [[nodiscard]] bool replace(const std::filesystem::path& source,
                             const std::filesystem::path& destination) const noexcept {
#if defined(_WIN32)
    NTSTATUS status{};
    HANDLE file = open_relative(
        handle_,
        source,
        DELETE | FILE_READ_ATTRIBUTES | SYNCHRONIZE,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        FILE_OPEN,
        FILE_NON_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT | FILE_OPEN_REPARSE_POINT,
        status);
    if (!nt_success(status) || !safe_native_handle(file, false)) {
      close_native(file);
      return false;
    }
    const std::wstring native = destination.native();
    const std::size_t name_bytes = native.size() * sizeof(wchar_t);
    if (name_bytes > std::numeric_limits<DWORD>::max()) {
      close_native(file);
      return false;
    }
    std::vector<std::byte> storage(offsetof(FILE_RENAME_INFO, FileName) + name_bytes);
    auto* rename = reinterpret_cast<FILE_RENAME_INFO*>(storage.data());
    rename->ReplaceIfExists = TRUE;
    rename->RootDirectory = handle_;
    rename->FileNameLength = static_cast<DWORD>(name_bytes);
    std::memcpy(rename->FileName, native.data(), name_bytes);
    const bool succeeded =
        SetFileInformationByHandle(
            file, FileRenameInfo, rename, static_cast<DWORD>(storage.size())) != 0;
    return CloseHandle(file) != 0 && succeeded;
#else
    return ::renameat(handle_, source.c_str(), handle_, destination.c_str()) == 0;
#endif
  }

 private:
#if defined(_WIN32)
  using NativeHandle = HANDLE;
  using NtCreateFileFunction = NTSTATUS(NTAPI*)(PHANDLE, ACCESS_MASK, POBJECT_ATTRIBUTES,
                                                PIO_STATUS_BLOCK, PLARGE_INTEGER, ULONG, ULONG,
                                                ULONG, ULONG, PVOID, ULONG);
  static constexpr NativeHandle kInvalidHandle = INVALID_HANDLE_VALUE;

  [[nodiscard]] static bool nt_success(NTSTATUS status) noexcept { return status >= 0; }

  [[nodiscard]] static bool nt_missing(NTSTATUS status) noexcept {
    constexpr NTSTATUS kObjectNameNotFound = static_cast<NTSTATUS>(0xC0000034UL);
    constexpr NTSTATUS kObjectPathNotFound = static_cast<NTSTATUS>(0xC000003AUL);
    return status == kObjectNameNotFound || status == kObjectPathNotFound;
  }

  [[nodiscard]] static bool nt_collision(NTSTATUS status) noexcept {
    constexpr NTSTATUS kObjectNameCollision = static_cast<NTSTATUS>(0xC0000035UL);
    return status == kObjectNameCollision;
  }

  [[nodiscard]] static NtCreateFileFunction nt_create_file() noexcept {
    static const auto function = reinterpret_cast<NtCreateFileFunction>(
        GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtCreateFile"));
    return function;
  }

  [[nodiscard]] static HANDLE open_relative(HANDLE root, const std::filesystem::path& name,
                                            ACCESS_MASK access, ULONG share, ULONG disposition,
                                            ULONG options, NTSTATUS& status) noexcept {
    const NtCreateFileFunction create_file = nt_create_file();
    const std::wstring native = name.native();
    if (create_file == nullptr || native.empty() ||
        native.size() > std::numeric_limits<USHORT>::max() / sizeof(wchar_t)) {
      status = static_cast<NTSTATUS>(0xC000000DUL);
      return INVALID_HANDLE_VALUE;
    }
    UNICODE_STRING unicode{
        .Length = static_cast<USHORT>(native.size() * sizeof(wchar_t)),
        .MaximumLength = static_cast<USHORT>(native.size() * sizeof(wchar_t)),
        .Buffer = const_cast<PWSTR>(native.data()),
    };
    OBJECT_ATTRIBUTES attributes;
    InitializeObjectAttributes(&attributes, &unicode, OBJ_CASE_INSENSITIVE, root, nullptr);
    IO_STATUS_BLOCK io_status{};
    HANDLE result = INVALID_HANDLE_VALUE;
    status = create_file(&result,
                         access,
                         &attributes,
                         &io_status,
                         nullptr,
                         FILE_ATTRIBUTE_NORMAL,
                         share,
                         disposition,
                         options,
                         nullptr,
                         0U);
    return result;
  }

  [[nodiscard]] static bool safe_native_handle(HANDLE handle, bool directory) noexcept {
    if (handle == nullptr || handle == INVALID_HANDLE_VALUE) {
      return false;
    }
    FILE_ATTRIBUTE_TAG_INFO attributes{};
    if (GetFileInformationByHandleEx(
            handle, FileAttributeTagInfo, &attributes, sizeof(attributes)) == 0 ||
        (attributes.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0U) {
      return false;
    }
    return ((attributes.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0U) == directory;
  }

  static void close_native(HANDLE handle) noexcept {
    if (handle != nullptr && handle != INVALID_HANDLE_VALUE) {
      static_cast<void>(CloseHandle(handle));
    }
  }
#else
  using NativeHandle = int;
  static constexpr NativeHandle kInvalidHandle = -1;
#endif

  explicit ArtifactDirectory(NativeHandle handle) noexcept : handle_(handle) {}

  [[nodiscard]] NativeHandle release() noexcept {
    const NativeHandle result = handle_;
    handle_ = kInvalidHandle;
    return result;
  }

  void close() noexcept {
#if defined(_WIN32)
    close_native(handle_);
#else
    if (handle_ >= 0) {
      static_cast<void>(::close(handle_));
    }
#endif
    handle_ = kInvalidHandle;
  }

  NativeHandle handle_{kInvalidHandle};
};

[[nodiscard]] expected<void, Error> prepare_file(const ArtifactDirectory& directory,
                                                 PendingFile& file) {
  std::string current;
  const BoundedReadResult destination =
      directory.read(file.destination.filename(), file.bytes.size(), current);
  if (destination == BoundedReadResult::kFailure) {
    return make_unexpected(Error{ErrorCode::kFailure, "ARTIFACT_DESTINATION_READ_FAILED"});
  }
  if (destination == BoundedReadResult::kComplete && current == file.bytes) {
    file.changed = false;
    return {};
  }

  file.changed = true;
  for (std::size_t attempt = 0U; attempt < kMaximumTemporaryAttempts; ++attempt) {
    file.temporary = temporary_candidate(file.destination.filename());
    const ExclusiveWriteResult write_result = directory.write_exclusive(file.temporary, file.bytes);
    if (write_result == ExclusiveWriteResult::kCollision) {
      file.temporary.clear();
      continue;
    }
    if (write_result == ExclusiveWriteResult::kFailure) {
      directory.remove(file.temporary);
      file.temporary.clear();
      return make_unexpected(Error{ErrorCode::kFailure, "ARTIFACT_TEMPORARY_WRITE_FAILED"});
    }
    const bool injection_failed =
        temporary_oversize_pending.exchange(false, std::memory_order_relaxed) &&
        !directory.append_sentinel(file.temporary);
    std::string verified;
    const BoundedReadResult verify_result =
        directory.read(file.temporary, file.bytes.size(), verified);
    if (injection_failed || verify_result != BoundedReadResult::kComplete ||
        verified != file.bytes) {
      directory.remove(file.temporary);
      file.temporary.clear();
      return make_unexpected(Error{ErrorCode::kFailure, "ARTIFACT_TEMPORARY_VERIFY_FAILED"});
    }
    return {};
  }
  return make_unexpected(Error{ErrorCode::kFailure, "ARTIFACT_TEMPORARY_COLLISION_LIMIT"});
}

[[nodiscard]] expected<void, Error> replace_file(const ArtifactDirectory& directory,
                                                 PendingFile& file) {
  if (!file.changed) {
    return {};
  }
  bool injected_failure = false;
  std::int64_t remaining = replace_failure_countdown.load(std::memory_order_relaxed);
  while (remaining >= 0) {
    const std::int64_t next = remaining == 0 ? kNoInjectedReplaceFailure : remaining - 1;
    if (replace_failure_countdown.compare_exchange_weak(
            remaining, next, std::memory_order_relaxed, std::memory_order_relaxed)) {
      injected_failure = remaining == 0;
      break;
    }
  }
  if (injected_failure || !directory.replace(file.temporary, file.destination.filename())) {
    directory.remove(file.temporary);
    file.temporary.clear();
    return make_unexpected(Error{ErrorCode::kFailure, "ARTIFACT_REPLACE_FAILED"});
  }
  file.temporary.clear();
  return {};
}

void discard_temporary_files(const ArtifactDirectory& directory,
                             std::span<PendingFile> files) noexcept {
  for (auto& file : files) {
    if (!file.temporary.empty()) {
      directory.remove(file.temporary);
      file.temporary.clear();
    }
  }
}

[[nodiscard]] std::string manifest_prefix(std::string_view catalog_digest) {
  return "{\"catalog_digest\":" + detail::json_string(catalog_digest) +
         ",\"content_updated_at_ns\":";
}

[[nodiscard]] std::string manifest_suffix(const std::vector<PendingFile>& files) {
  std::string output{",\"files\":["};
  bool first = true;
  for (const auto& file : files) {
    if (!first) {
      output.push_back(',');
    }
    first = false;
    output += "{\"name\":" + detail::json_string(file.destination.filename().generic_string());
    output += ",\"sha256\":" +
              detail::json_string(detail::lowercase_hex_text(hash_bytes(file.bytes).value,
                                                             detail::HexPrefix::kSha256)) +
              "}";
  }
  output += "],\"schema\":" + detail::json_string(kArtifactManifestSchema) + "}";
  return output;
}

[[nodiscard]] std::size_t maximum_manifest_size(const std::vector<PendingFile>& files,
                                                std::string_view catalog_digest) {
  return manifest_prefix(catalog_digest).size() + kMaximumManifestTimestampCharacters +
         manifest_suffix(files).size();
}

[[nodiscard]] std::string artifact_manifest(const std::vector<PendingFile>& files,
                                            std::string_view catalog_digest) {
  using ManifestNanoseconds = std::chrono::duration<std::int64_t, std::chrono::nanoseconds::period>;
  const std::int64_t update =
      std::chrono::time_point_cast<ManifestNanoseconds>(std::chrono::system_clock::now())
          .time_since_epoch()
          .count();
  return manifest_prefix(catalog_digest) + std::to_string(update) + manifest_suffix(files);
}

[[nodiscard]] bool manifest_is_valid(std::string_view manifest,
                                     const std::vector<PendingFile>& files,
                                     std::string_view catalog_digest) {
  const std::string prefix = manifest_prefix(catalog_digest);
  const std::string suffix = manifest_suffix(files);
  if (!manifest.starts_with(prefix) || !manifest.ends_with(suffix) ||
      manifest.size() <= prefix.size() + suffix.size()) {
    return false;
  }
  const std::string_view update =
      manifest.substr(prefix.size(), manifest.size() - prefix.size() - suffix.size());
  std::int64_t parsed_update{};
  const auto [end, error] =
      std::from_chars(update.data(), update.data() + update.size(), parsed_update);
  return error == std::errc() && end == update.data() + update.size() &&
         std::to_string(parsed_update) == update;
}

}  // namespace

namespace detail {

void ArtifactEmitterTestAccess::fail_next_replace() noexcept {
  fail_replace_after_successes(0U);
}

void ArtifactEmitterTestAccess::fail_replace_after_successes(
    std::uint32_t successful_replacements) noexcept {
  replace_failure_countdown.store(static_cast<std::int64_t>(successful_replacements),
                                  std::memory_order_relaxed);
}

void ArtifactEmitterTestAccess::throw_next_filesystem_error() noexcept {
  filesystem_exception_pending.store(true, std::memory_order_relaxed);
}

void ArtifactEmitterTestAccess::oversize_next_temporary() noexcept {
  temporary_oversize_pending.store(true, std::memory_order_relaxed);
}

std::size_t ArtifactEmitterTestAccess::maximum_manifest_bytes() {
  const std::vector<PendingFile> files{
      PendingFile{.destination = "endpoint-catalog.json"},
      PendingFile{.destination = "execution-plan.json"},
      PendingFile{.destination = "profile-manifest.json"},
  };
  const std::string digest = "sha256:" + std::string(64U, '0');
  return maximum_manifest_size(files, digest);
}

void ArtifactEmitterTestAccess::reset_temporary_injection() noexcept {
  temporary_oversize_pending.store(false, std::memory_order_relaxed);
}

}  // namespace detail

namespace {

expected<ArtifactEmitResult, Error> emit_artifacts_impl(const ExecutionPlan& plan,
                                                        const ArtifactEmitOptions& options) {
  if (filesystem_exception_pending.exchange(false, std::memory_order_relaxed)) {
    throw std::filesystem::filesystem_error{"injected artifact filesystem failure",
                                            std::make_error_code(std::errc::io_error)};
  }
  const detail::ExecutionPlanState* state = detail::ExecutionPlanAccess::state(plan);
  if (state == nullptr || !state->admitted || !valid_deployment_id(options.deployment_id)) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "ARTIFACT_PLAN_OR_DEPLOYMENT_INVALID"});
  }
  const bool generated_name_unadmitted =
      std::any_of(state->profile_manifest.generated_endpoint_names.begin(),
                  state->profile_manifest.generated_endpoint_names.end(),
                  [](const auto& record) {
                    return record.source == GeneratedEndpointNameAdmissionSource::kNone ||
                           record.disposition != ProfileFeatureDisposition::kAdmitted;
                  });
  if (generated_name_unadmitted) {
    return make_unexpected(
        Error{ErrorCode::kPermissionDenied, "ENDPOINT_NAME_ARTIFACT_UNADMITTED"});
  }

  const std::string graph_id =
      detail::lowercase_hex_text(state->digest.value, detail::HexPrefix::kNone);
  const std::filesystem::path directory = artifact_root(options) / graph_id / options.deployment_id;
  auto opened_directory = ArtifactDirectory::open_or_create(directory);
  if (!opened_directory) {
    return make_unexpected(std::move(opened_directory).error());
  }
  const ArtifactDirectory artifact_directory = std::move(opened_directory).value();

  const std::string catalog_json = state->endpoint_catalog.json();
  const std::string catalog_digest{state->endpoint_catalog.digest()};
  const auto profile = extract_top_level_object(state->json, "profile");
  if (!profile.has_value()) {
    return make_unexpected(Error{ErrorCode::kFailure, "ARTIFACT_PROFILE_SERIALIZATION_MISSING"});
  }

  std::vector<PendingFile> files{
      PendingFile{.destination = directory / "endpoint-catalog.json", .bytes = catalog_json},
      PendingFile{.destination = directory / "execution-plan.json", .bytes = state->json},
      PendingFile{.destination = directory / "profile-manifest.json", .bytes = *profile},
  };
  for (auto& file : files) {
    auto prepared = prepare_file(artifact_directory, file);
    if (!prepared) {
      discard_temporary_files(artifact_directory, files);
      return make_unexpected(std::move(prepared.error()));
    }
  }

  const bool content_changed =
      std::any_of(files.begin(), files.end(), [](const auto& file) { return file.changed; });
  PendingFile manifest{.destination = directory / "artifact-manifest.json",
                       .bytes = artifact_manifest(files, catalog_digest)};
  bool valid_existing_manifest = false;
  std::string existing_manifest;
  const BoundedReadResult manifest_read =
      artifact_directory.read(manifest.destination.filename(),
                              maximum_manifest_size(files, catalog_digest),
                              existing_manifest);
  if (manifest_read == BoundedReadResult::kFailure) {
    discard_temporary_files(artifact_directory, files);
    return make_unexpected(Error{ErrorCode::kFailure, "ARTIFACT_MANIFEST_READ_FAILED"});
  }
  valid_existing_manifest = manifest_read == BoundedReadResult::kComplete &&
                            manifest_is_valid(existing_manifest, files, catalog_digest);
  if (!content_changed && valid_existing_manifest) {
    return ArtifactEmitResult{.changed = false,
                              .catalog_path = files[0].destination,
                              .manifest_path = manifest.destination,
                              .catalog_digest = catalog_digest};
  }

  auto manifest_prepared = prepare_file(artifact_directory, manifest);
  if (!manifest_prepared) {
    discard_temporary_files(artifact_directory, files);
    return make_unexpected(std::move(manifest_prepared.error()));
  }

  // The manifest is the commit marker. Publishing it last means readers either
  // observe the previous valid bundle or reject a hash mismatch while a writer
  // replaces content files; they never accept a mixed generation as complete.
  for (auto& file : files) {
    auto replaced = replace_file(artifact_directory, file);
    if (!replaced) {
      discard_temporary_files(artifact_directory, files);
      const std::span<PendingFile> manifest_span{&manifest, 1U};
      discard_temporary_files(artifact_directory, manifest_span);
      return make_unexpected(std::move(replaced.error()));
    }
  }
  auto manifest_replaced = replace_file(artifact_directory, manifest);
  if (!manifest_replaced) {
    discard_temporary_files(artifact_directory, files);
    return make_unexpected(std::move(manifest_replaced.error()));
  }

  return ArtifactEmitResult{.changed = content_changed || manifest.changed,
                            .catalog_path = files[0].destination,
                            .manifest_path = manifest.destination,
                            .catalog_digest = catalog_digest};
}

}  // namespace

// The public value parameter owns one immutable publication snapshot; the
// private implementation only borrows that snapshot for this call.
// NOLINTBEGIN(performance-unnecessary-value-param)
expected<ArtifactEmitResult, Error> emit_artifacts(const ExecutionPlan& plan,
                                                   ArtifactEmitOptions options) {
  try {
    return emit_artifacts_impl(plan, options);
  } catch (const std::filesystem::filesystem_error&) {
    return make_unexpected(Error{ErrorCode::kFailure, "ARTIFACT_FILESYSTEM_EXCEPTION"});
  }
}
// NOLINTEND(performance-unnecessary-value-param)

}  // namespace holoscan
