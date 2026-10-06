/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <atomic>
#include <deque>
#include <mutex>
#include <string>
#include <string_view>

#define NVTX_NO_IMPL
#define NVTX_STATIC_INJECTION_IMPL
#include <nvtx3/nvToolsExt.h>

namespace holoscan::test::nvtx_emit_marker_capture {
namespace {

struct CaptureState {
  std::atomic<bool> initialized{};
  std::atomic<std::size_t> count{};
  std::atomic<nvtxStringHandle_t> expected_handle{};
  std::mutex strings_mutex;
  std::deque<std::string> strings;
  std::string expected_label;
};

CaptureState& state() {
  static CaptureState value;
  return value;
}

void NVTX_API capture_mark(nvtxDomainHandle_t, const nvtxEventAttributes_t* attributes) {
  if (attributes == nullptr || attributes->messageType != NVTX_MESSAGE_TYPE_REGISTERED ||
      attributes->message.registered == nullptr) {
    return;
  }
  auto& capture = state();
  if (attributes->message.registered == capture.expected_handle.load(std::memory_order_relaxed)) {
    capture.count.fetch_add(1U, std::memory_order_relaxed);
  }
}

nvtxStringHandle_t NVTX_API capture_string(nvtxDomainHandle_t, const char* value) {
  auto& capture = state();
  std::lock_guard lock{capture.strings_mutex};
  capture.strings.emplace_back(value);
  const auto handle = reinterpret_cast<nvtxStringHandle_t>(&capture.strings.back());
  if (capture.strings.back() == capture.expected_label) {
    capture.expected_handle.store(handle, std::memory_order_relaxed);
  }
  return handle;
}

int NVTX_API initialize(NvtxGetExportTableFunc_t get_export_table) {
  const auto* const callbacks =
      static_cast<const NvtxExportTableCallbacks*>(get_export_table(NVTX_ETID_CALLBACKS));
  NvtxFunctionTable table{};
  unsigned size{};
  if (callbacks == nullptr ||
      callbacks->GetModuleFunctionTable(NVTX_CB_MODULE_CORE2, &table, &size) == 0 ||
      size <= NVTX_CBID_CORE2_DomainRegisterStringA) {
    return 0;
  }
  *table[NVTX_CBID_CORE2_DomainMarkEx] = reinterpret_cast<NvtxFunctionPointer>(capture_mark);
  *table[NVTX_CBID_CORE2_DomainRegisterStringA] =
      reinterpret_cast<NvtxFunctionPointer>(capture_string);
  state().initialized.store(true, std::memory_order_release);
  return 1;
}

}  // namespace

bool reset(std::string_view label) {
  // Arm the capture before the session registers the expected label.
  auto& capture = state();
  if (!capture.initialized.load(std::memory_order_acquire)) {
    return false;
  }
  std::lock_guard lock{capture.strings_mutex};
  capture.expected_label = label;
  capture.expected_handle.store(nullptr, std::memory_order_relaxed);
  capture.count.store(0U, std::memory_order_relaxed);
  return true;
}

std::size_t emitted() noexcept {
  return state().count.load(std::memory_order_relaxed);
}

}  // namespace holoscan::test::nvtx_emit_marker_capture

extern "C" {
NvtxInitializeInjectionNvtxFunc_t InitializeInjectionNvtx2_fnptr =
    holoscan::test::nvtx_emit_marker_capture::initialize;
}
