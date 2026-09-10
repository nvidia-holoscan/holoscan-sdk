/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <holoviz/rhi/slang_compiler.hpp>

#include <slang-com-ptr.h>
#include <slang.h>

#include <cstddef>
#include <cstring>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <holoviz/logger.hpp>

namespace holoviz::rhi {

struct SlangCompilerInternalState {
  slang::IGlobalSession* global_session = nullptr;
  slang::ISession* spirv_session = nullptr;
};

/// The process-wide Slang sessions, created exactly once (magic static: concurrent
/// first callers block until the initializer finishes) and immortal — releasing a
/// Slang session mid-process is not reliable (its Linkage teardown has crashed in
/// practice), so the sessions deliberately live for the remainder of the process.
/// A failed creation is sticky — every later caller re-sees the stored error.
/// Known cost of the immortal session: it caches loaded modules by name for its
/// lifetime, so repeated compiles grow session memory (bounded in practice by
/// setup-time compiles; a per-compile session is not an option, see above).
static expected<SlangCompilerInternalState> slang_state_init() {
  SlangCompilerInternalState state;
  if (SLANG_FAILED(slang::createGlobalSession(&state.global_session))) {
    return make_unexpected(Error{ErrorCode::kFailure, "failed to create Slang global session"});
  }

  slang::TargetDesc target{};
  target.format = SLANG_SPIRV;
  target.profile = state.global_session->findProfile("spirv_1_5");

  slang::CompilerOptionEntry options[] = {
      {slang::CompilerOptionName::EmitSpirvDirectly,
       {slang::CompilerOptionValueKind::Int, 1, 0, nullptr, nullptr}},
  };

  slang::SessionDesc session_desc{};
  session_desc.targets = &target;
  session_desc.targetCount = 1;
  session_desc.compilerOptionEntries = options;
  session_desc.compilerOptionEntryCount = 1;

  if (SLANG_FAILED(state.global_session->createSession(session_desc, &state.spirv_session))) {
    state.global_session->release();
    state.global_session = nullptr;
    return make_unexpected(Error{ErrorCode::kFailure, "failed to create Slang session"});
  }
  return state;
}

static expected<slang::ISession*> slang_spirv_session() {
  static const expected<SlangCompilerInternalState> state = slang_state_init();
  if (!state) {
    return make_unexpected(state.error());
  }
  return state->spirv_session;
}

/// Serializes compile_slang_shader() bodies: Slang sessions (including the global
/// session's createSession) are not thread-safe. Compiles happen at setup time,
/// so a coarse process-wide lock costs nothing on the frame path.
static std::mutex& slang_compile_mutex() {
  static std::mutex m;
  return m;
}

namespace {

// Internal linkage on purpose: these take raw Slang reflection pointers and are only
// valid under slang_compile_mutex() — nothing outside this translation unit may call
// them, keeping "all Slang access is serialized by the mutex" enforceable.
expected<BindingKind> to_binding_kind(slang::TypeLayoutReflection* tl) {
  // Arrays of resources: classify by the element type.
  if (tl->getKind() == slang::TypeReflection::Kind::Array)
    tl = tl->getElementTypeLayout();

  switch (tl->getKind()) {
      // case slang::TypeReflection::Kind::ConstantBuffer:
      //     return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    case slang::TypeReflection::Kind::SamplerState:
      return BindingKind::kSampler;

    case slang::TypeReflection::Kind::Resource: {
      auto shape = tl->getResourceShape();
      auto access = tl->getResourceAccess();

      if (access != SLANG_RESOURCE_ACCESS_READ) {
        return make_unexpected(
            Error{ErrorCode::kFailure, "Only read-only resources are supported for now"});
      }

      switch (shape & SLANG_RESOURCE_BASE_SHAPE_MASK) {
        case SLANG_STRUCTURED_BUFFER:
        case SLANG_BYTE_ADDRESS_BUFFER:
          return BindingKind::kStorageBuffer;

        case SLANG_TEXTURE_2D:
          // The public Image API creates plain 2D images only (ImageDesc takes an
          // Extent2D), so arrayed/multisampled/combined/shadow/feedback variants
          // (Texture2DArray, Texture2DMS, Sampler2D, ...) reflect an interface no
          // RHI image could satisfy.
          if (shape & SLANG_RESOURCE_EXT_SHAPE_MASK) {
            return make_unexpected(
                Error{ErrorCode::kFailure,
                      "arrayed/multisampled/combined texture bindings are not supported; "
                      "only plain Texture2D"});
          }
          return BindingKind::kSampledImage;

        case SLANG_TEXTURE_1D:
        case SLANG_TEXTURE_3D:
        case SLANG_TEXTURE_CUBE:
          return make_unexpected(Error{ErrorCode::kFailure,
                                       "only 2D sampled images are supported (RHI images are 2D)"});

        default:
          break;
      }
      break;
    }

    default:
      break;
  }

  return make_unexpected(
      Error{ErrorCode::kFailure, "Slang type is not a recognized resource type"});
}

expected<std::pair<Format, uint32_t>> to_format_size_pair(slang::TypeReflection* type) {
  const unsigned count = type->getElementCount() ? type->getElementCount() : 1;
  if (type->getScalarType() == slang::TypeReflection::Float32) {
    uint32_t byte_size = 4 * count;
    constexpr Format formats[] = {Format::kR32Sfloat,
                                  Format::kR32G32Sfloat,
                                  Format::kR32G32B32Sfloat,
                                  Format::kR32G32B32A32Sfloat};
    if (count >= 1 && count <= 4)
      return std::pair<Format, uint32_t>{formats[count - 1], byte_size};
  }
  if (type->getScalarType() == slang::TypeReflection::UInt32 && count == 1) {
    // Scalar uint attribute, e.g. a packed RGBA8 color unpacked in the shader.
    return std::pair<Format, uint32_t>{Format::kR32Uint, 4};
  }
  // extend: Int32, uint2..4, float16, unorm8 color attrs, ...

  return make_unexpected(
      Error{ErrorCode::kFailure, "Slang type is not a recognized vertex attribute type"});
}

// Logs a diagnostics blob left behind by a *successful* Slang operation (warnings)
// and clears the slot so a later failure cannot pick up stale text. Each Slang
// out-diagnostic is caller-owned; the ComPtr releases it on setNull().
void log_and_clear_slang_diagnostics(Slang::ComPtr<slang::IBlob>& diagnostics) {
  if (!diagnostics)
    return;
  // The blob is not guaranteed NUL-terminated; bound the read by getBufferSize().
  HOLOVIZ_LOG_WARN("Slang diagnostics: {}",
                   std::string_view(static_cast<const char*>(diagnostics->getBufferPointer()),
                                    diagnostics->getBufferSize()));
  diagnostics.setNull();
}

}  // namespace

expected<ShaderBlob> compile_slang_shader(Backend backend, const char* vertex_entry_point,
                                          const char* fragment_entry_point, const char* source) {
  if (vertex_entry_point == nullptr || fragment_entry_point == nullptr || source == nullptr) {
    return make_unexpected(
        Error{ErrorCode::kInvalidArgument, "Null pointer passed to compile_slang_shader"});
  }

  const std::lock_guard<std::mutex> lock(slang_compile_mutex());

  VIZ_UNWRAP(slang::ISession*, spirv_session, slang_spirv_session());

  if (backend != Backend::kVulkan && backend != Backend::kAuto) {
    return make_unexpected(
        Error{ErrorCode::kNotImplemented, "Slang compiler only supports Vulkan backend for now"});
  }

  slang::IModule* module = nullptr;
  // Every Slang out-diagnostic is caller-owned, and an operation that emits nothing
  // leaves the slot untouched. writeRef() releases and nulls the held blob before
  // each out-call, so each operation's diagnostic is isolated: no overwrite leak,
  // and fail() never reports stale text from an earlier operation.
  Slang::ComPtr<slang::IBlob> diagnostics;
  slang::IEntryPoint* vertex_entry_point_ptr = nullptr;
  slang::IEntryPoint* fragment_entry_point_ptr = nullptr;
  slang::IComponentType* program = nullptr;
  slang::IComponentType* linked = nullptr;
  slang::IBlob* vertex_spirv = nullptr;
  slang::IBlob* fragment_spirv = nullptr;

  auto cleanup = [&]() {
    if (vertex_spirv)
      vertex_spirv->release();
    if (fragment_spirv)
      fragment_spirv->release();
    if (linked)
      linked->release();
    if (program)
      program->release();
    if (vertex_entry_point_ptr)
      vertex_entry_point_ptr->release();
    if (fragment_entry_point_ptr)
      fragment_entry_point_ptr->release();
    if (module)
      module->release();
  };

  auto fail = [&](const std::string& fallback) -> expected<ShaderBlob> {
    std::string message = fallback;
    if (diagnostics) {
      message.assign(static_cast<const char*>(diagnostics->getBufferPointer()),
                     diagnostics->getBufferSize());
    }
    cleanup();
    return make_unexpected(Error{ErrorCode::kFailure, message});
  };

  // The session caches modules by name and rejects a reused name with different source
  // ("module already loaded with different source"), so every compile gets a fresh name.
  // Plain (non-atomic) because the whole function body runs under slang_compile_mutex().
  // The cache therefore grows one entry per compile for the life of the process — accepted:
  // compiles happen at setup time, bounded by the declared pipeline set.
  static uint64_t module_counter = 0;
  const std::string module_name = "module_" + std::to_string(module_counter++);
  module = spirv_session->loadModuleFromSourceString(
      module_name.c_str(), (module_name + ".slang").c_str(), source, diagnostics.writeRef());
  if (!module) {
    return fail("Slang: failed to load module");
  }
  log_and_clear_slang_diagnostics(diagnostics);

  if (SLANG_FAILED(module->findEntryPointByName(vertex_entry_point, &vertex_entry_point_ptr))) {
    return fail("Slang: vertex entry point not found: " + std::string(vertex_entry_point));
  }
  if (SLANG_FAILED(module->findEntryPointByName(fragment_entry_point, &fragment_entry_point_ptr))) {
    return fail("Slang: fragment entry point not found: " + std::string(fragment_entry_point));
  }

  // 3. Compose and link. Both results are caller-owned.
  slang::IComponentType* components[] = {module, vertex_entry_point_ptr, fragment_entry_point_ptr};
  if (SLANG_FAILED(spirv_session->createCompositeComponentType(
          components, 3, &program, diagnostics.writeRef()))) {
    return fail("Slang: composition failed");
  }
  log_and_clear_slang_diagnostics(diagnostics);
  if (SLANG_FAILED(program->link(&linked, diagnostics.writeRef()))) {
    return fail("Slang: link failed");
  }
  log_and_clear_slang_diagnostics(diagnostics);

  // 4. Code generation. The blob is caller-owned; copy out, then release.
  if (SLANG_FAILED(linked->getEntryPointCode(0, 0, &vertex_spirv, diagnostics.writeRef()))) {
    return fail("Slang: vertex SPIR-V generation failed");
  }
  log_and_clear_slang_diagnostics(diagnostics);
  if (SLANG_FAILED(linked->getEntryPointCode(1, 0, &fragment_spirv, diagnostics.writeRef()))) {
    return fail("Slang: fragment SPIR-V generation failed");
  }
  log_and_clear_slang_diagnostics(diagnostics);

  slang::ProgramLayout* layout = linked->getLayout(0, diagnostics.writeRef());
  if (layout == nullptr) {
    return fail("Slang: failed to get program layout");
  }
  log_and_clear_slang_diagnostics(diagnostics);

  const auto* vertex_words = static_cast<const uint32_t*>(vertex_spirv->getBufferPointer());

  std::vector<std::byte> vertex_bytes(vertex_spirv->getBufferSize());
  std::memcpy(vertex_bytes.data(), vertex_words, vertex_spirv->getBufferSize());

  const auto* fragment_words = static_cast<const uint32_t*>(fragment_spirv->getBufferPointer());

  std::vector<std::byte> fragment_bytes(fragment_spirv->getBufferSize());
  std::memcpy(fragment_bytes.data(), fragment_words, fragment_spirv->getBufferSize());

  std::vector<AttributeDesc> attributes;
  std::vector<BindingDesc> bindings;
  uint32_t push_constant_size = 0;

  slang::EntryPointReflection* vs = layout->getEntryPointByIndex(0);
  if (vs == nullptr) {
    return fail("Slang: failed to get vertex entry point reflection");
  }
  for (unsigned i = 0; i < vs->getParameterCount(); i++) {
    slang::VariableLayoutReflection* param = vs->getParameterByIndex(i);
    slang::TypeLayoutReflection* tl = param->getTypeLayout();
    if (tl->getKind() != slang::TypeReflection::Kind::Struct) {
      // System-value inputs (e.g. `uint vid : SV_VertexID`) are legal direct
      // parameters and contribute no vertex attribute.
      const char* sem = param->getSemanticName();
      if (sem && strncmp(sem, "SV_", 3) == 0)
        continue;
      // Any other direct varying input would need its own location/format
      // reflection; the supported grammar is a single input struct instead.
      if (param->getCategory() == slang::ParameterCategory::VaryingInput) {
        return fail("Slang: direct vertex input '" + std::string(param->getName()) +
                    "' is not supported; wrap user vertex inputs in a struct parameter");
      }
      continue;
    }

    for (unsigned f = 0; f < tl->getFieldCount(); f++) {
      slang::VariableLayoutReflection* field = tl->getFieldByIndex(f);
      const char* sem = field->getSemanticName();
      if (sem && strncmp(sem, "SV_", 3) == 0)
        continue;

      auto format_result = to_format_size_pair(field->getType());
      if (!format_result) {
        return fail("Slang: failed to convert vertex attribute type to Format and size");
      }

      attributes.push_back({
          .name = std::string(field->getVariable()->getName()),
          .location = field->getBindingIndex(),
          .format = format_result.value().first,
          .byte_size = format_result.value().second,
      });
    }
  }

  for (unsigned i = 0; i < layout->getParameterCount(); i++) {
    slang::VariableLayoutReflection* param = layout->getParameterByIndex(i);
    slang::ParameterCategory category = param->getCategory();

    if (category == slang::ParameterCategory::PushConstantBuffer) {
      auto* element = param->getTypeLayout()->getElementTypeLayout();
      push_constant_size =
          static_cast<uint32_t>(element->getSize(slang::ParameterCategory::Uniform));
      HOLOVIZ_LOG_DEBUG("Slang: reflected push constant size: {} bytes", push_constant_size);
      continue;
    }

    if (category != slang::ParameterCategory::DescriptorTableSlot)
      continue;

    // Descriptor arrays (e.g. `Texture2D images[2]`, sized or unsized) are rejected:
    // the Descriptor API only has single-resource bind operations, so a layout with
    // count > 1 could never be populated. Checked before to_binding_kind() so an
    // array of an unsupported type still reports this clearer error.
    if (param->getTypeLayout()->getKind() == slang::TypeReflection::Kind::Array) {
      return fail("Slang: array resource bindings are not supported yet (binding '" +
                  std::string(param->getName()) + "')");
    }

    auto binding_kind_result = to_binding_kind(param->getTypeLayout());
    if (!binding_kind_result) {
      return fail("Slang: failed to convert resource type to BindingKind (binding '" +
                  std::string(param->getName()) + "'): " + binding_kind_result.error().message);
    }

    if (param->getBindingSpace() != 0) {
      return fail("Slang: binding space is not 0; only space 0 is supported for now");
    }

    bindings.push_back({
        .name = std::string(param->getName()),
        .index = static_cast<uint32_t>(param->getBindingIndex()),
        .kind = binding_kind_result.value(),
        .count = 1u,
    });
  }

  cleanup();

  return ShaderBlob{
      .ir = ShaderIR::kSpirv,
      .vertex_bytes = std::move(vertex_bytes),
      .fragment_bytes = std::move(fragment_bytes),
      .vertex_entry_point = vertex_entry_point,
      .fragment_entry_point = fragment_entry_point,
      .attributes = std::move(attributes),
      .bindings = std::move(bindings),
      .push_constant_size = push_constant_size,
  };
}

}  // namespace holoviz::rhi
