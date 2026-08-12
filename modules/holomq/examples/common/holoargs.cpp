/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "holoargs.hpp"

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace holomq_examples {
namespace args {

using ParsedOptions = std::unordered_map<int, std::optional<std::string>>;

TypedValue<bool>::Pointer bool_switch() {
  return std::make_shared<TypedValue<bool>>();
}

OptionDescription::OptionDescription(std::string name, int key, std::string doc,
                                     std::shared_ptr<const ValueSemantic> value_semantic)
    : name_(std::move(name)),
      key_(key),
      doc_(std::move(doc)),
      value_semantic_(std::move(value_semantic)) {}

OptionsDescriptionEasyInit::OptionsDescriptionEasyInit(OptionsDescription& options_description)
    : options_description_(options_description) {}

OptionsDescriptionEasyInit& OptionsDescriptionEasyInit::operator()(const std::string& name,
                                                                   const std::string& doc) {
  options_description_.add(
      OptionDescription(name, options_description_.long_form_only_index_++, doc));
  return *this;
}

OptionsDescriptionEasyInit& OptionsDescriptionEasyInit::operator()(
    const std::string& name, std::shared_ptr<const ValueSemantic> value_semantic,
    const std::string& doc) {
  options_description_.add(OptionDescription(
      name, options_description_.long_form_only_index_++, doc, std::move(value_semantic)));
  return *this;
}

OptionsDescription::OptionsDescription(std::string caption)
    : caption_(std::move(caption)), long_form_only_index_(256) {}

void OptionsDescription::add(const OptionDescription& option_description) {
  options_.emplace(option_description.key_, option_description);
}

OptionsDescriptionEasyInit OptionsDescription::add_options() {
  return OptionsDescriptionEasyInit(*this);
}

VariableValue::VariableValue(std::any value) : value_(std::move(value)) {}

std::any& VariableValue::value() {
  return value_;
}

const std::any& VariableValue::value() const {
  return value_;
}

bool VariableValue::empty() const {
  return !value_.has_value();
}

struct Input {
  const OptionsDescription& options_description_;
  ParsedOptions& parsed_option_;
};

// PARSER. Field 2 in ARGP.
error_t Parser::parse_option(int key, char* arg, ::argp_state* state) {
  auto input = reinterpret_cast<Input*>(state->input);
  const OptionsDescription& options_description(input->options_description_);
  ParsedOptions& parsed_options(input->parsed_option_);

  auto iter = options_description.options_.find(key);
  if (iter == options_description.options_.end())
    return ARGP_ERR_UNKNOWN;
  if (arg != nullptr)
    parsed_options.emplace(key, std::optional<std::string>(arg));
  else
    parsed_options.emplace(key, std::optional<std::string>());
  return 0;
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
VariablesMap Parser::parse_command_line(int argc,
                                        char* argv[],  // NOLINT(cppcoreguidelines-avoid-c-arrays)
                                        const OptionsDescription& options_description) const {
  //  OPTIONS.  Field 1 in ARGP.
  std::vector<::argp_option> options;
  std::vector<std::string> arg_name_storage;
  options.reserve(options_description.options_.size() + 1);
  arg_name_storage.reserve(options_description.options_.size());
  for (auto& [key, option_description] : options_description.options_) {
    const auto& value_semantic = option_description.value_semantic_;
    const char* arg = nullptr;
    int flags{};
    if (value_semantic != nullptr) {
      if (value_semantic->value_type() != typeid(bool)) {
        const std::string name = value_semantic->name();
        if (name.empty()) {
          arg = "VALUE";
        } else {
          arg_name_storage.push_back(name);
          arg = arg_name_storage.back().c_str();
        }
      } else {
        flags |= OPTION_ARG_OPTIONAL;
      }
    }

    options.emplace_back(::argp_option{option_description.name_.c_str(),
                                       option_description.key_,
                                       arg,
                                       flags,
                                       option_description.doc_.c_str()});
  }
  options.emplace_back(::argp_option{});

  // PARSER. Field 2 in ARGP.
  // Parser::parse_option

  // ARGS_DOC. Field 3 in ARGP.
  const std::string args_doc;

  // DOC.  Field 4 in ARGP.
  const std::string& doc = options_description.caption_;

  // ARGP
  const ::argp argp = {&options.front(), &Parser::parse_option, args_doc.c_str(), doc.c_str()};

  ParsedOptions parsed_options;
  Input input{options_description, parsed_options};

  ::argp_parse(&argp, argc, argv, 0, 0, &input);

  // Store options in the VariablesMap
  VariablesMap variables_map;
  variables_map.reserve(parsed_options.size());
  for (auto& [key, optional_token] : parsed_options) {
    auto& option_description = options_description.options_.at(key);
    std::any value;
    const auto& value_semantic = option_description.value_semantic_;
    if (optional_token.has_value() && value_semantic != nullptr) {
      value_semantic->parse(value, optional_token.value());
    } else if (value_semantic == nullptr || value_semantic->value_type() == typeid(bool)) {
      value = true;
    }
    variables_map[option_description.name_] = VariableValue(value);
  }

  // Set Default Values and verify that Required Options have values
  for (const auto& [key, option_description] : options_description.options_) {
    if (variables_map.find(option_description.name_) != variables_map.end())
      continue;
    if (option_description.value_semantic_ == nullptr)
      continue;
    auto& value_semantic = *option_description.value_semantic_;
    std::any value;
    if (value_semantic.apply_default(value))
      variables_map[option_description.name_] = VariableValue(value);
    else if (value_semantic.is_required())
      throw RequiredOption(option_description.name_);
  }
  return variables_map;
}

}  // namespace args
}  // namespace holomq_examples
