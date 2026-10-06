/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.hpp"

#include "module_name.hpp"

#include <fmt/format.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <holoscan/config/document.hpp>
#include <holoscan/config/io.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/parameter_provenance.hpp>
#include <holoscan/params/config.hpp>
#include <holoscan/params/error.hpp>

#include "graph.hpp"
#include "python_path.hpp"

namespace holoscan {
namespace {

ErrorCode core_error_code(config::ConfigErrorCode config_code) noexcept {
  ErrorCode code = ErrorCode::kInvalidArgument;
  switch (config_code) {
    case config::ConfigErrorCode::kPathNotFound:
      code = ErrorCode::kNotFound;
      break;
    case config::ConfigErrorCode::kUnsupportedFormat:
    case config::ConfigErrorCode::kUnsupportedValue:
    case config::ConfigErrorCode::kAtomicReplaceUnsupported:
    case config::ConfigErrorCode::kUnsupportedVersion:
      code = ErrorCode::kNotSupported;
      break;
    case config::ConfigErrorCode::kPermissionDenied:
      code = ErrorCode::kPermissionDenied;
      break;
    case config::ConfigErrorCode::kGuardBusy:
      code = ErrorCode::kTimeout;
      break;
    case config::ConfigErrorCode::kLimitExceeded:
      code = ErrorCode::kResourceExhausted;
      break;
    case config::ConfigErrorCode::kIoError:
      code = ErrorCode::kFailure;
      break;
    default:
      break;
  }
  return code;
}

/// Carry a complete configuration failure to the Python exception translator.
class ConfigBindingError final : public std::exception {
 public:
  explicit ConfigBindingError(config::ConfigError error)
      : core_code_(core_error_code(error.code())),
        error_(std::move(error)),
        message_(config::format_error(error_)) {}

  [[nodiscard]] ErrorCode core_code() const noexcept { return core_code_; }
  [[nodiscard]] const config::ConfigError& error() const noexcept { return error_; }
  [[nodiscard]] const char* what() const noexcept override { return message_.c_str(); }

 private:
  ErrorCode core_code_;
  config::ConfigError error_;
  std::string message_;
};

enum class ConfigExceptionKind : std::size_t {
  kInvalidArgument,
  kNotFound,
  kNotSupported,
  kPermissionDenied,
  kTimeout,
  kResourceExhausted,
  kFailure,
  kCount,
};

constexpr std::size_t kConfigExceptionKindCount =
    static_cast<std::size_t>(ConfigExceptionKind::kCount);

ConfigExceptionKind config_exception_kind(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::kNotFound:
      return ConfigExceptionKind::kNotFound;
    case ErrorCode::kNotSupported:
      return ConfigExceptionKind::kNotSupported;
    case ErrorCode::kPermissionDenied:
      return ConfigExceptionKind::kPermissionDenied;
    case ErrorCode::kTimeout:
      return ConfigExceptionKind::kTimeout;
    case ErrorCode::kResourceExhausted:
      return ConfigExceptionKind::kResourceExhausted;
    case ErrorCode::kFailure:
      return ConfigExceptionKind::kFailure;
    default:
      return ConfigExceptionKind::kInvalidArgument;
  }
}

struct ConfigExceptionState {
  std::array<nb::object, kConfigExceptionKindCount> exception_types{};
  std::array<nb::object, kConfigExceptionKindCount> error_codes{};
};

void translate_config_binding_error(const std::exception_ptr& exception, void* payload) {
  try {
    std::rethrow_exception(exception);
  } catch (const ConfigBindingError& error) {
    auto& state = *static_cast<ConfigExceptionState*>(payload);
    const std::size_t index = static_cast<std::size_t>(config_exception_kind(error.core_code()));
    PyObject* type = state.exception_types[index].ptr();
    PyObject* instance = PyObject_CallFunction(type, "s", error.what());
    if (instance == nullptr) {
      return;
    }
    const std::string_view token = error.error().token();
    if (token.size() > static_cast<std::size_t>(PY_SSIZE_T_MAX)) {
      Py_DECREF(instance);
      PyErr_SetString(PyExc_RuntimeError, "Configuration exception token is too large");
      return;
    }
    PyObject* token_object =
        PyUnicode_FromStringAndSize(token.data(), static_cast<Py_ssize_t>(token.size()));
    if (token_object == nullptr) {
      Py_DECREF(instance);
      return;
    }
    const nb::object details = nb::cast(error.error());
    if (PyObject_SetAttrString(instance, "code", state.error_codes[index].ptr()) != 0 ||
        PyObject_SetAttrString(instance, "token", token_object) != 0 ||
        PyObject_SetAttrString(instance, "config_error", details.ptr()) != 0) {
      Py_DECREF(token_object);
      Py_DECREF(instance);
      return;
    }
    Py_DECREF(token_object);
    PyErr_SetObject(type, instance);
    Py_DECREF(instance);
  }
}

[[noreturn]] void throw_config_error(const config::ConfigError& error) {
  throw ConfigBindingError{error};
}

[[noreturn]] void throw_param_error(const params::ParamToolError& error) {
  ErrorCode code = ErrorCode::kInvalidArgument;
  switch (error.code()) {
    case params::ParamToolErrorCode::kTargetMismatch:
      code = ErrorCode::kIncompatible;
      break;
    case params::ParamToolErrorCode::kPermissionDenied:
      code = ErrorCode::kPermissionDenied;
      break;
    case params::ParamToolErrorCode::kNotSupported:
      code = ErrorCode::kNotSupported;
      break;
    case params::ParamToolErrorCode::kClosed:
      code = ErrorCode::kClosed;
      break;
    case params::ParamToolErrorCode::kTimeout:
      code = ErrorCode::kTimeout;
      break;
    case params::ParamToolErrorCode::kProtocol:
      code = ErrorCode::kProtocolError;
      break;
    case params::ParamToolErrorCode::kResourceExhausted:
      code = ErrorCode::kResourceExhausted;
      break;
    case params::ParamToolErrorCode::kPersistence:
      code = ErrorCode::kFailure;
      break;
    default:
      break;
  }
  throw RuntimeError{Error{code, params::format_error(error)}};
}

config_io::Format infer_format(const std::filesystem::path& path) {
  std::string extension = path.extension().string();
  std::transform(
      extension.begin(), extension.end(), extension.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
      });
  if (extension == ".json") {
    return config_io::Format::kJson;
  }
  if (extension == ".yaml" || extension == ".yml") {
    return config_io::Format::kYaml;
  }
  throw RuntimeError{Error{ErrorCode::kInvalidArgument, "CONFIG_FORMAT_REQUIRED"}};
}

config::Document load_document(nb::handle path, std::optional<config_io::Format> format) {
  const std::filesystem::path source = python_filesystem_path(path);
  const config_io::Format selected_format = format.has_value() ? *format : infer_format(source);
  auto loaded = [&source, selected_format]() {
    const nb::gil_scoped_release release;
    return config_io::try_load(source, selected_format);
  }();
  if (!loaded) {
    throw_config_error(loaded.error());
  }
  return std::move(*loaded);
}

std::string config_source_bytes(nb::handle source) {
  if (nb::isinstance<nb::bytes>(source)) {
    char* bytes = nullptr;
    Py_ssize_t size = 0;
    if (PyBytes_AsStringAndSize(source.ptr(), &bytes, &size) != 0) {
      throw nb::python_error();
    }
    return std::string{bytes, static_cast<std::size_t>(size)};
  }
  if (nb::isinstance<nb::str>(source)) {
    Py_ssize_t size = 0;
    const char* text = PyUnicode_AsUTF8AndSize(source.ptr(), &size);
    if (text == nullptr) {
      throw nb::python_error();
    }
    return std::string{text, static_cast<std::size_t>(size)};
  }
  throw nb::type_error("CONFIG_SOURCE_STRING_OR_BYTES_REQUIRED");
}

config::Document parse_document(nb::handle source, config_io::Format format) {
  const std::string bytes = config_source_bytes(source);
  auto parsed = [&bytes, format]() {
    const nb::gil_scoped_release release;
    return config_io::try_parse(bytes, format);
  }();
  if (!parsed) {
    throw_config_error(parsed.error());
  }
  return std::move(*parsed);
}

config::Path parse_path(const std::string& value) {
  auto parsed = config::Path::try_parse(value);
  if (!parsed) {
    throw_config_error(parsed.error());
  }
  return std::move(*parsed);
}

std::string hex_digest(const config::Sha256Digest& digest) {
  constexpr std::string_view kHex{"0123456789abcdef"};
  std::string result(digest.size() * 2U, '0');
  for (std::size_t index = 0U; index < digest.size(); ++index) {
    const auto byte = std::to_integer<unsigned int>(digest[index]);
    result[index * 2U] = kHex[byte >> 4U];
    result[index * 2U + 1U] = kHex[byte & 0x0FU];
  }
  return result;
}

std::string prefixed_digest(const config::Sha256Digest& digest) {
  return "sha256:" + hex_digest(digest);
}

// Retains one validated operator-to-config binding and its display metadata for Python.
class PyParamConfigBinding {
 public:
  PyParamConfigBinding(nb::handle op, config::Path path, params::ParamBindingMode mode)
      : operator_path_(python_operator_name(op)),
        config_path_(path.to_string()),
        binding_(python_operator_config_binding(op, std::move(path), mode)) {}

  [[nodiscard]] const params::ParamConfigBinding& binding() const noexcept { return binding_; }
  [[nodiscard]] std::string repr() const {
    return fmt::format("<params.Binding '{}' <- '{}'>", operator_path_, config_path_);
  }

 private:
  std::string operator_path_;
  std::string config_path_;
  params::ParamConfigBinding binding_;
};

nb::tuple immutable_config_fields(const std::vector<ParamConfigFieldProvenance>& config_fields) {
  nb::list result;
  for (const auto& field : config_fields) {
    result.append(nb::cast(field));
  }
  PyObject* tuple = PyList_AsTuple(result.ptr());
  if (tuple == nullptr) {
    throw nb::python_error();
  }
  return nb::steal<nb::tuple>(tuple);
}

// Owns one parameter-rebind result and exposes its immutable provenance to Python.
class PyParamRebindProvenance {
 public:
  explicit PyParamRebindProvenance(ParamRebindProvenance provenance)
      : provenance_(std::move(provenance)),
        config_fields_(immutable_config_fields(provenance_.config_fields)) {}

  [[nodiscard]] ParamRebindMode mode() const noexcept { return provenance_.mode; }
  [[nodiscard]] nb::tuple config_fields() const { return config_fields_; }
  [[nodiscard]] std::size_t config_field_count() const noexcept {
    return provenance_.config_fields.size();
  }

 private:
  ParamRebindProvenance provenance_;
  nb::tuple config_fields_;
};

PyParamRebindProvenance set_graph_params(nb::handle graph, const config::Document& document,
                                         nb::handle values,
                                         params::ParamConfigDecodeOptions options) {
  Graph& native_graph = python_graph(graph);
  std::vector<params::ParamConfigBinding> core_bindings;
  nb::iterable bindings;
  try {
    bindings = nb::cast<nb::iterable>(values);
  } catch (const nb::cast_error&) {
    throw nb::type_error("PARAM_CONFIG_BINDINGS_ITERABLE_REQUIRED");
  }
  std::size_t index = 0U;
  for (const nb::handle value : bindings) {
    try {
      core_bindings.push_back(nb::cast<const PyParamConfigBinding&>(value).binding());
    } catch (const nb::cast_error&) {
      const std::string message =
          fmt::format("PARAM_CONFIG_BINDING_ELEMENT_REQUIRED index={}", index);
      throw nb::type_error(message.c_str());
    }
    ++index;
  }
  auto attached = params::set_params(native_graph, document, core_bindings, options);
  if (!attached) {
    throw_param_error(attached.error());
  }
  return PyParamRebindProvenance{std::move(*attached)};
}

std::optional<config::RawSourceDigest> config_source(
    const ParamConfigDocumentProvenance& provenance, bool base) {
  const bool present = base ? provenance.has_base_raw_source : provenance.has_raw_source;
  if (!present) {
    return std::nullopt;
  }
  if (base) {
    return config::RawSourceDigest{.algorithm_version = provenance.base_raw_algorithm_version,
                                   .sha256 = provenance.base_raw_digest,
                                   .byte_size = provenance.base_raw_byte_size};
  }
  return config::RawSourceDigest{.algorithm_version = provenance.raw_algorithm_version,
                                 .sha256 = provenance.raw_digest,
                                 .byte_size = provenance.raw_byte_size};
}

std::string raw_source_repr(const config::RawSourceDigest& source) {
  return fmt::format("<config.RawSourceDigest digest='{}' byte_size={}>",
                     prefixed_digest(source.sha256),
                     source.byte_size);
}

std::string field_provenance_repr(const ParamConfigFieldProvenance& field) {
  return fmt::format(
      "<params.ConfigFieldProvenance '{}' -> '{}'>", field.source_path, field.target_operator_path);
}

std::string set_params_provenance_repr(const PyParamRebindProvenance& provenance) {
  return fmt::format("<params.SetParamsProvenance config_fields={}>",
                     provenance.config_field_count());
}

std::optional<config::RawSourceDigest> document_source(const config::Document& document,
                                                       bool base) {
  const config::ConfigDigest& digest = document.digest();
  if (base) {
    return digest.base_source;
  }
  return digest.source;
}

std::string schema_hash(const ParamSchemaHash& hash) {
  return prefixed_digest(hash.value);
}

std::string config_semantic_digest(const ParamConfigDocumentProvenance& provenance) {
  return prefixed_digest(provenance.semantic_digest);
}

std::string document_semantic_digest(const config::Document& document) {
  return prefixed_digest(document.digest().semantic.sha256);
}

std::string document_repr(const config::Document& document) {
  return fmt::format("<config.Document semantic_digest='{}'>", document_semantic_digest(document));
}

std::string config_error_cause_repr(const config::ConfigErrorCause& error) {
  return fmt::format("<config.ConfigErrorCause token='{}' path='{}'>", error.token(), error.path());
}

nb::tuple config_error_causes(const config::ConfigError& error) {
  nb::list result;
  for (std::size_t index = 0U; index < error.cause_count(); ++index) {
    result.append(nb::cast(*error.cause(index)));
  }
  PyObject* tuple = PyList_AsTuple(result.ptr());
  if (tuple == nullptr) {
    throw nb::python_error();
  }
  return nb::steal<nb::tuple>(tuple);
}

std::string config_error_repr(const config::ConfigError& error) {
  return fmt::format(
      "<config.ConfigErrorDetails token='{}' path='{}'>", error.token(), error.path());
}

void bind_config_error_types(nb::module_& module) {
  python::bind_enum_in_module<config::ConfigErrorCode>(module, "holoscan.config", "ConfigErrorCode")
      .value("INVALID_ARGUMENT", config::ConfigErrorCode::kInvalidArgument)
      .value("PARSE_ERROR", config::ConfigErrorCode::kParseError)
      .value("UNSUPPORTED_FORMAT", config::ConfigErrorCode::kUnsupportedFormat)
      .value("UNSUPPORTED_VALUE", config::ConfigErrorCode::kUnsupportedValue)
      .value("INVALID_UTF8", config::ConfigErrorCode::kInvalidUtf8)
      .value("DUPLICATE_KEY", config::ConfigErrorCode::kDuplicateKey)
      .value("PATH_SYNTAX", config::ConfigErrorCode::kPathSyntax)
      .value("PATH_NOT_FOUND", config::ConfigErrorCode::kPathNotFound)
      .value("PATH_CONFLICT", config::ConfigErrorCode::kPathConflict)
      .value("TYPE_MISMATCH", config::ConfigErrorCode::kTypeMismatch)
      .value("OUT_OF_RANGE", config::ConfigErrorCode::kOutOfRange)
      .value("LIMIT_EXCEEDED", config::ConfigErrorCode::kLimitExceeded)
      .value("LOSSLESS_PRESERVATION_UNAVAILABLE",
             config::ConfigErrorCode::kLosslessPreservationUnavailable)
      .value("SOURCE_CHANGED", config::ConfigErrorCode::kSourceChanged)
      .value("GUARD_BUSY", config::ConfigErrorCode::kGuardBusy)
      .value("CODEC_MISSING", config::ConfigErrorCode::kCodecMissing)
      .value("CODEC_FAILURE", config::ConfigErrorCode::kCodecFailure)
      .value("IO_ERROR", config::ConfigErrorCode::kIoError)
      .value("PERMISSION_DENIED", config::ConfigErrorCode::kPermissionDenied)
      .value("ATOMIC_REPLACE_UNSUPPORTED", config::ConfigErrorCode::kAtomicReplaceUnsupported)
      .value("DIGEST_MISMATCH", config::ConfigErrorCode::kDigestMismatch)
      .value("UNSUPPORTED_VERSION", config::ConfigErrorCode::kUnsupportedVersion);
  python::bind_enum_in_module<config::ConfigOperationStage>(
      module, "holoscan.config", "ConfigOperationStage")
      .value("NONE", config::ConfigOperationStage::kNone)
      .value("PARSE", config::ConfigOperationStage::kParse)
      .value("VALIDATE", config::ConfigOperationStage::kValidate)
      .value("SERIALIZE", config::ConfigOperationStage::kSerialize)
      .value("WRITE_TEMPORARY", config::ConfigOperationStage::kWriteTemporary)
      .value("ACQUIRE_GUARD", config::ConfigOperationStage::kAcquireGuard)
      .value("REPLACE_DESTINATION", config::ConfigOperationStage::kReplaceDestination)
      .value("FLUSH_DESTINATION_METADATA", config::ConfigOperationStage::kFlushDestinationMetadata);
  python::bind_enum_in_module<config::ConfigDestinationState>(
      module, "holoscan.config", "ConfigDestinationState")
      .value("NOT_APPLICABLE", config::ConfigDestinationState::kNotApplicable)
      .value("UNCHANGED", config::ConfigDestinationState::kUnchanged)
      .value("REPLACED", config::ConfigDestinationState::kReplaced);
  python::bind_class_in_module<config::ConfigErrorCause>(
      module, "holoscan.config", "ConfigErrorCause")
      .def_prop_ro("code", &config::ConfigErrorCause::code)
      .def_prop_ro("token",
                   [](const config::ConfigErrorCause& error) { return std::string{error.token()}; })
      .def_prop_ro(
          "message",
          [](const config::ConfigErrorCause& error) { return std::string{error.message()}; })
      .def_prop_ro("path",
                   [](const config::ConfigErrorCause& error) { return std::string{error.path()}; })
      .def_prop_ro(
          "source",
          [](const config::ConfigErrorCause& error) { return std::string{error.source()}; })
      .def_prop_ro("message_truncated", &config::ConfigErrorCause::message_truncated)
      .def_prop_ro("path_truncated", &config::ConfigErrorCause::path_truncated)
      .def_prop_ro("source_truncated", &config::ConfigErrorCause::source_truncated)
      .def_prop_ro("excerpt_truncated", &config::ConfigErrorCause::excerpt_truncated)
      .def_prop_ro("cause_overflow", &config::ConfigErrorCause::cause_overflow)
      .def_prop_ro("operation_stage", &config::ConfigErrorCause::operation_stage)
      .def_prop_ro("destination_state", &config::ConfigErrorCause::destination_state)
      .def("__repr__", &config_error_cause_repr);
  python::bind_class_in_module<config::ConfigError>(module, "holoscan.config", "ConfigErrorDetails")
      .def_prop_ro("code", &config::ConfigError::code)
      .def_prop_ro("token",
                   [](const config::ConfigError& error) { return std::string{error.token()}; })
      .def_prop_ro("message",
                   [](const config::ConfigError& error) { return std::string{error.message()}; })
      .def_prop_ro("path",
                   [](const config::ConfigError& error) { return std::string{error.path()}; })
      .def_prop_ro("source",
                   [](const config::ConfigError& error) { return std::string{error.source()}; })
      .def_prop_ro("message_truncated", &config::ConfigError::message_truncated)
      .def_prop_ro("path_truncated", &config::ConfigError::path_truncated)
      .def_prop_ro("source_truncated", &config::ConfigError::source_truncated)
      .def_prop_ro("excerpt_truncated", &config::ConfigError::excerpt_truncated)
      .def_prop_ro("cause_overflow", &config::ConfigError::cause_overflow)
      .def_prop_ro("causes",
                   &config_error_causes,
                   nb::for_getter(nb::sig(
                       "def causes(self, /) -> tuple[holoscan.config.ConfigErrorCause, ...]")))
      .def_prop_ro("operation_stage", &config::ConfigError::operation_stage)
      .def_prop_ro("destination_state", &config::ConfigError::destination_state)
      .def("__repr__", &config_error_repr);
}

void register_config_exception_translator(nb::module_& module) {
  // Match the process-lifetime translator ownership used by the general Holoscan error bindings.
  auto* state = new ConfigExceptionState;
  constexpr std::array entries{
      std::tuple{ConfigExceptionKind::kInvalidArgument, "InvalidArgumentError", "INVALID_ARGUMENT"},
      std::tuple{ConfigExceptionKind::kNotFound, "NotFoundError", "NOT_FOUND"},
      std::tuple{ConfigExceptionKind::kNotSupported, "NotSupportedError", "NOT_SUPPORTED"},
      std::tuple{
          ConfigExceptionKind::kPermissionDenied, "PermissionDeniedError", "PERMISSION_DENIED"},
      std::tuple{ConfigExceptionKind::kTimeout, "HoloscanTimeoutError", "TIMEOUT"},
      std::tuple{
          ConfigExceptionKind::kResourceExhausted, "ResourceExhaustedError", "RESOURCE_EXHAUSTED"},
      std::tuple{ConfigExceptionKind::kFailure, "FailureError", "FAILURE"},
  };
  for (const auto& [kind, exception_name, code_name] : entries) {
    const std::size_t index = static_cast<std::size_t>(kind);
    state->exception_types[index] = module.attr(exception_name);
    state->error_codes[index] = module.attr("ErrorCode").attr(code_name);
  }
  nb::register_exception_translator(&translate_config_binding_error, state);
}

std::string provenance_document_repr(const ParamConfigDocumentProvenance& provenance) {
  return fmt::format("<params.ConfigDocumentProvenance semantic_digest='{}'>",
                     config_semantic_digest(provenance));
}

std::string binding_path(nb::handle path) {
  try {
    return nb::cast<std::string>(path);
  } catch (const nb::cast_error&) {
    throw RuntimeError{Error{ErrorCode::kInvalidArgument, "PARAM_CONFIG_PATH_STRING_REQUIRED"}};
  }
}

config::Path binding_config_path(nb::handle path) {
  if (nb::isinstance<config::Path>(path)) {
    return nb::cast<config::Path>(path);
  }
  return parse_path(binding_path(path));
}

PyParamConfigBinding make_binding(nb::handle op, nb::handle path, params::ParamBindingMode mode) {
  return PyParamConfigBinding{op, binding_config_path(path), mode};
}

void bind_config_document_types(nb::module_& module) {
  python::bind_class_in_module<config::RawSourceDigest>(
      module, "holoscan.config", "RawSourceDigest")
      .def_ro("algorithm_version", &config::RawSourceDigest::algorithm_version)
      .def_prop_ro(
          "digest",
          [](const config::RawSourceDigest& source) { return prefixed_digest(source.sha256); })
      .def_ro("byte_size", &config::RawSourceDigest::byte_size)
      .def("__repr__", &raw_source_repr);
  python::bind_class_in_module<config::Document>(
      module, "holoscan.config", "Document", "An immutable configuration document.")
      .def_prop_ro("semantic_digest", &document_semantic_digest)
      .def_prop_ro(
          "source",
          [](const config::Document& document) { return document_source(document, false); })
      .def_prop_ro("base_source",
                   [](const config::Document& document) { return document_source(document, true); })
      .def("__repr__", &document_repr);
}

void bind_config_provenance_types(nb::module_& module) {
  python::bind_enum_in_module<ParamFieldRebindOrigin>(module, "holoscan.params", "FieldOrigin")
      .value("CAPTURED", ParamFieldRebindOrigin::kCaptured)
      .value("ALIAS", ParamFieldRebindOrigin::kAlias)
      .value("MIGRATION", ParamFieldRebindOrigin::kMigration)
      .value("BASELINE", ParamFieldRebindOrigin::kBaseline)
      .value("DEFAULT", ParamFieldRebindOrigin::kDefault);
  python::bind_enum_in_module<ParamRebindMode>(module, "holoscan.params", "RebindMode")
      .value("SPARSE_CONTRIBUTIONS", ParamRebindMode::kSparseContributions)
      .value("COMPLETE_OBJECT", ParamRebindMode::kCompleteObject);
  python::bind_class_in_module<ParamConfigDocumentProvenance>(
      module, "holoscan.params", "ConfigDocumentProvenance")
      .def_ro("semantic_version", &ParamConfigDocumentProvenance::semantic_version)
      .def_prop_ro("semantic_digest", &config_semantic_digest)
      .def_prop_ro("source",
                   [](const ParamConfigDocumentProvenance& provenance) {
                     return config_source(provenance, false);
                   })
      .def_prop_ro("base_source",
                   [](const ParamConfigDocumentProvenance& provenance) {
                     return config_source(provenance, true);
                   })
      .def("__repr__", &provenance_document_repr);
  python::bind_class_in_module<ParamConfigFieldProvenance>(
      module, "holoscan.params", "ConfigFieldProvenance")
      .def_ro("document", &ParamConfigFieldProvenance::document)
      .def_ro("source_path", &ParamConfigFieldProvenance::source_path)
      .def_ro("source_ordinal", &ParamConfigFieldProvenance::source_ordinal)
      .def_ro("target_operator_path", &ParamConfigFieldProvenance::target_operator_path)
      .def_ro("target_schema_id", &ParamConfigFieldProvenance::target_schema_id)
      .def_ro("target_schema_version", &ParamConfigFieldProvenance::target_schema_version)
      .def_ro("target_field_id", &ParamConfigFieldProvenance::target_field_id)
      .def_prop_ro(
          "target_full_id",
          [](const ParamConfigFieldProvenance& field) { return schema_hash(field.target_full_id); })
      .def_ro("origin", &ParamConfigFieldProvenance::origin)
      .def("__repr__", &field_provenance_repr);
  python::bind_class_in_module<PyParamRebindProvenance>(
      module, "holoscan.params", "SetParamsProvenance")
      .def_prop_ro("mode", &PyParamRebindProvenance::mode)
      .def_prop_ro("config_fields", &PyParamRebindProvenance::config_fields)
      .def_prop_ro("config_field_count", &PyParamRebindProvenance::config_field_count)
      .def("__repr__", &set_params_provenance_repr);
}

}  // namespace

void init_config(nb::module_& module) {
  bind_config_error_types(module);
  register_config_exception_translator(module);
  python::bind_class_in_module<config::Path>(
      module, "holoscan.config", "Path", "An owning RFC 6901 JSON Pointer.")
      .def_static("parse", &parse_path, nb::arg("value"))
      .def_prop_ro("tokens",
                   [](const config::Path& path) {
                     std::vector<std::string> result;
                     result.reserve(path.token_count());
                     for (std::size_t index = 0U; index < path.token_count(); ++index) {
                       result.emplace_back(path.token(index));
                     }
                     return result;
                   })
      .def_prop_ro("is_root", &config::Path::is_root)
      .def("__str__", &config::Path::to_string)
      .def("__repr__", [](const config::Path& path) {
        return fmt::format("<config.Path '{}'>", path.to_string());
      });
  bind_config_document_types(module);

  python::bind_enum_in_module<config_io::Format>(module, "holoscan.config", "Format")
      .value("JSON", config_io::Format::kJson)
      .value("YAML", config_io::Format::kYaml);
  python::bind_enum_in_module<config::MissingValuePolicy>(
      module, "holoscan.config", "MissingValuePolicy")
      .value("REJECT", config::MissingValuePolicy::kReject)
      .value("USE_TYPE_DEFAULTS", config::MissingValuePolicy::kUseTypeDefaults);
  python::bind_enum_in_module<config::UnknownFieldPolicy>(
      module, "holoscan.config", "UnknownFieldPolicy")
      .value("REJECT", config::UnknownFieldPolicy::kReject)
      .value("IGNORE", config::UnknownFieldPolicy::kIgnore);
  python::bind_enum_in_module<config::NumericConversionPolicy>(
      module, "holoscan.config", "NumericConversionPolicy")
      .value("LOSSLESS", config::NumericConversionPolicy::kLossless)
      .value("ALLOW_CHECKED_LOSSY", config::NumericConversionPolicy::kAllowCheckedLossy);
  python::bind_class_in_module<config::DecodeOptions>(
      module, "holoscan.config", "ValueDecodeOptions")
      .def(nb::init<config::MissingValuePolicy,
                    config::UnknownFieldPolicy,
                    config::NumericConversionPolicy>(),
           nb::arg("missing_fields") = config::MissingValuePolicy::kReject,
           nb::arg("unknown_fields") = config::UnknownFieldPolicy::kReject,
           nb::arg("numeric_conversion") = config::NumericConversionPolicy::kLossless)
      .def_prop_ro("missing_fields", &config::DecodeOptions::missing_fields)
      .def_prop_ro("unknown_fields", &config::DecodeOptions::unknown_fields)
      .def_prop_ro("numeric_conversion", &config::DecodeOptions::numeric_conversion);
  python::bind_functions_in_module(module, "holoscan.config", [&] {
    module.def("load", &load_document, nb::arg("path"), nb::arg("format").none() = nb::none());
    module.def("parse", &parse_document, nb::arg("source"), nb::arg("format"));
  });

  python::bind_enum_in_module<params::ParamBindingMode>(module, "holoscan.params", "BindingMode")
      .value("PATCH", params::ParamBindingMode::kPatch);
  python::bind_class_in_module<params::ParamConfigDecodeOptions>(
      module, "holoscan.params", "DecodeOptions")
      .def(nb::init<>())
      .def_rw("values", &params::ParamConfigDecodeOptions::values)
      .def_rw("max_bindings", &params::ParamConfigDecodeOptions::max_bindings)
      .def_rw("max_fields", &params::ParamConfigDecodeOptions::max_fields);
  python::bind_class_in_module<PyParamConfigBinding>(module, "holoscan.params", "Binding")
      .def("__repr__", &PyParamConfigBinding::repr);
  bind_config_provenance_types(module);
  python::bind_functions_in_module(module, "holoscan.params", [&] {
    module.def("bind",
               &make_binding,
               nb::arg("operator"),
               nb::arg("path"),
               nb::arg("mode") = params::ParamBindingMode::kPatch);
    module.def("set_params",
               &set_graph_params,
               nb::arg("graph"),
               nb::arg("document"),
               nb::arg("bindings"),
               nb::arg("options") = params::ParamConfigDecodeOptions{});
  });
}

}  // namespace holoscan
