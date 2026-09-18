/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "diagnostic_context.hpp"

#include <algorithm>
#include <exception>
#include <new>
#include <string>

#include <holoscan/core/execution_plan.hpp>

#include "canonical_text_encoding.hpp"

namespace holoscan::detail {
namespace {

#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
void capture_current(DiagnosticCapture& capture) noexcept;

void capture_standard(DiagnosticCapture& capture, ErrorCode code, const std::exception& exception,
                      bool strip_code) noexcept {
  const char* text = exception.what();
  if (text == nullptr) {
    text = "exception without a message";
  }
  std::size_t length = 0U;
  while (length < DiagnosticCapture::kTextCapacity && text[length] != '\0') {
    ++length;
  }
  capture.truncated |= length == DiagnosticCapture::kTextCapacity && text[length] != '\0';
  std::string_view message{text, length};
  if (strip_code) {
    const auto name = error_code_name(code);
    if (message == name) {
      message = {};
    } else if (message.starts_with(name) && message.substr(name.size(), 2U) == ": ") {
      message.remove_prefix(name.size() + 2U);
    }
  }
  capture.append(code, message);
  const auto* nested = dynamic_cast<const std::nested_exception*>(&exception);
  if (nested != nullptr && nested->nested_ptr()) {
    if (capture.count == DiagnosticCapture::kCauseCapacity) {
      capture.truncated = true;
      return;
    }
    try {
      nested->rethrow_nested();
    } catch (...) {
      capture_current(capture);
    }
  }
}

void capture_current(DiagnosticCapture& capture) noexcept {
  try {
    throw;
  } catch (const RuntimeError& error) {
    capture_standard(capture, error.code(), error, true);
  } catch (const std::bad_alloc& error) {
    capture_standard(capture, ErrorCode::kResourceExhausted, error, false);
  } catch (const std::exception& error) {
    capture_standard(capture, ErrorCode::kFailure, error, false);
  } catch (...) {
    capture.append(ErrorCode::kFailure, "non-standard exception");
  }
}
#endif

}  // namespace

void DiagnosticCapture::clear() noexcept {
  size = 0U;
  count = 0U;
  truncated = false;
}

void DiagnosticCapture::append(ErrorCode code, std::string_view message) noexcept {
  if (count == causes.size()) {
    truncated = true;
    return;
  }
  const std::size_t length = std::min(message.size(), text.size() - size);
  causes[count++] = Cause{code, size, length};
  std::copy_n(message.begin(), length, text.begin() + static_cast<std::ptrdiff_t>(size));
  size += length;
  truncated |= length != message.size();
}

void DiagnosticCapture::capture_exception() noexcept {
  clear();
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
  capture_current(*this);
#else
  append(ErrorCode::kFailure, "exception details unavailable");
#endif
}

DiagnosticDetail DiagnosticCapture::detail(DiagnosticOperation operation) const {
  DiagnosticDetail result{.operation = operation, .truncated = truncated};
  if (count != 0U) {
    const auto& first = causes[0];
    result.error = Error{first.code, std::string{text.data() + first.offset, first.length}};
    for (std::size_t index = 1U; index < count; ++index) {
      const auto& cause = causes[index];
      result.causes.push_back(
          Error{cause.code, std::string{text.data() + cause.offset, cause.length}});
    }
  }
  return result;
}

template <typename GraphViewT>
std::string operator_path_for(const GraphViewT& graph, VertexId vertex_id) {
  const auto vertices = graph.vertices();
  const auto vertex = std::ranges::find(vertices, vertex_id, &VertexRecord::id);
  if (vertex == vertices.end()) {
    return {};
  }
  std::string result;
  const auto graphs = graph.graphs();
  const auto owner = std::ranges::find(graphs, vertex->graph, &GraphRecord::id);
  if (owner != graphs.end()) {
    const auto& path = owner->path;
    if (path.has_value()) {
      for (const auto& segment : path->segments()) {
        append_percent_encoded_uri_path_segment(result, segment);
      }
    }
  }
  append_percent_encoded_uri_path_segment(result, vertex->name);
  return result;
}

std::string diagnostic_operator_path(const ExecutionGraphView& graph, VertexId vertex) {
  return operator_path_for(graph, vertex);
}

std::string diagnostic_operator_path(const GraphView& graph, VertexId vertex) {
  return operator_path_for(graph, vertex);
}

std::string diagnostic_display_text(std::string_view text, std::size_t limit) {
  std::string result;
  constexpr std::string_view hex{"0123456789abcdef"};
  const auto bounded = text.substr(0U, limit);
  for (const unsigned char byte : bounded) {
    if (byte < 0x20U || byte == 0x7fU) {
      result += "\\x";
      result.push_back(hex[byte >> 4U]);
      result.push_back(hex[byte & 0xfU]);
    } else {
      result.push_back(static_cast<char>(byte));
    }
  }
  if (bounded.size() != text.size()) {
    result += " [truncated]";
  }
  return result;
}

}  // namespace holoscan::detail
