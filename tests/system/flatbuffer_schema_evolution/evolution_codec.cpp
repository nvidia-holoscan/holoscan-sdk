/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include EVOLUTION_SCHEMA_HEADER

#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include <flatbuffers/flatbuffer_builder.h>
#include <flatbuffers/verifier.h>

namespace {

constexpr std::uint64_t kSequence = 42U;
constexpr std::int32_t kLevel = 9;
constexpr std::string_view kLabel = "frame";
constexpr std::string_view kNote = "optional-v2-field";

bool write_buffer(const std::string& path) {
  flatbuffers::FlatBufferBuilder builder;
  const auto label = builder.CreateString(kLabel);
#if EVOLUTION_HAS_NOTE
  const auto note = builder.CreateString(kNote);
#endif
  schema_evolution::FrameBuilder frame_builder(builder);
  frame_builder.add_sequence(kSequence);
  frame_builder.add_label(label);
  frame_builder.add_level(kLevel);
#if EVOLUTION_HAS_NOTE
  frame_builder.add_note(note);
#endif
  const auto frame = frame_builder.Finish();
  schema_evolution::FinishFrameBuffer(builder, frame);

  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(builder.GetBufferPointer()),
               static_cast<std::streamsize>(builder.GetSize()));
  return output.good();
}

bool read_buffer(const std::string& path, std::string_view writer_revision) {
  std::ifstream input(path, std::ios::binary);
  const std::vector<std::uint8_t> bytes{std::istreambuf_iterator<char>{input},
                                        std::istreambuf_iterator<char>{}};
  if (bytes.empty()) {
    return false;
  }

  flatbuffers::Verifier verifier(bytes.data(), bytes.size());
  if (!schema_evolution::VerifyFrameBuffer(verifier)) {
    return false;
  }
  const schema_evolution::Frame* frame = schema_evolution::GetFrame(bytes.data());
  if (frame == nullptr || frame->sequence() != kSequence || frame->level() != kLevel ||
      frame->label() == nullptr || frame->label()->string_view() != kLabel) {
    return false;
  }
#if EVOLUTION_HAS_NOTE
  if (writer_revision == "v1") {
    return frame->note() == nullptr;
  }
  return frame->note() != nullptr && frame->note()->string_view() == kNote;
#else
  static_cast<void>(writer_revision);
  return true;
#endif
}

}  // namespace

// REQ: SCHEMA-002
int main(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::cerr << "usage: evolution_codec write <path> | read <path> <writer-revision>\n";
    return 2;
  }
  const std::string_view operation{argv[1]};
  if (operation == "write" && argc == 3) {
    return write_buffer(argv[2]) ? 0 : 1;
  }
  if (operation == "read" && argc == 4) {
    return read_buffer(argv[2], argv[3]) ? 0 : 1;
  }
  return 2;
}
