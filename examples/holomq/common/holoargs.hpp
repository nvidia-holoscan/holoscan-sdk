/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * The Holoargs work was inspired by the Boost.Program_options library.
 * Although it was implemented from scratch, an effort was made to keep the
 * API similar to Boost.Program_options' for whose those who are familiar with.
 */
#pragma once

#include <argp.h>

#include <any>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace holomq_examples {
namespace args {

class OptionsDescription;
class Parser;

class RequiredOption : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class ValueSemantic : public std::enable_shared_from_this<ValueSemantic> {
 public:
  using Pointer = std::shared_ptr<ValueSemantic>;
  virtual ~ValueSemantic() = default;
  virtual bool is_required() const = 0;
  virtual std::string name() const = 0;
  virtual bool apply_default(std::any& value) const = 0;
  virtual void parse(std::any& value_store, const std::string& token) const = 0;
  virtual const std::type_info& value_type() const = 0;
};

template <typename T>
class TypedValue : public ValueSemantic {
 public:
  using Pointer = std::shared_ptr<TypedValue<T>>;

  virtual ~TypedValue() = default;

  Pointer required() {
    required_ = true;
    return std::static_pointer_cast<TypedValue<T>>(shared_from_this());
  }

  Pointer value_name(const std::string& name) {
    name_ = name;
    return std::static_pointer_cast<TypedValue<T>>(shared_from_this());
  }

  Pointer default_value(const T& default_value) {
    default_value_ = default_value;
    return std::static_pointer_cast<TypedValue<T>>(shared_from_this());
  }

  bool is_required() const override { return required_; }

  std::string name() const override { return name_; }

  bool apply_default(std::any& value) const override {
    if (default_value_.has_value())
      value = default_value_;
    return default_value_.has_value();
  }

  void parse(std::any& value_store, const std::string& token) const override {
    T value{};
    std::istringstream iss(token);
    if (!(iss >> value) || (iss.peek() != std::char_traits<char>::eof())) {
      throw std::invalid_argument("invalid option value: " + token);
    }
    value_store = value;
  }

  const std::type_info& value_type() const override { return typeid(T); }

 private:
  bool required_ = false;
  std::string name_;
  std::any default_value_;
};

template <typename T>
typename TypedValue<T>::Pointer value() {
  return std::make_shared<TypedValue<T>>();
}

TypedValue<bool>::Pointer bool_switch();

class OptionDescription {
 public:
  OptionDescription(std::string name, int key, std::string doc,
                    std::shared_ptr<const ValueSemantic> value_semantic = nullptr);

 private:
  std::string name_;
  int key_;
  std::string doc_;
  std::shared_ptr<const ValueSemantic> value_semantic_;

  friend class OptionsDescription;
  friend class Parser;
};

class OptionsDescriptionEasyInit {
 public:
  explicit OptionsDescriptionEasyInit(OptionsDescription& options_description);

  OptionsDescriptionEasyInit& operator()(const std::string& name, const std::string& doc);
  OptionsDescriptionEasyInit& operator()(const std::string& name,
                                         std::shared_ptr<const ValueSemantic> value_semantic,
                                         const std::string& doc);

 private:
  OptionsDescription& options_description_;
};

class OptionsDescription {
 public:
  explicit OptionsDescription(std::string caption = std::string());
  void add(const OptionDescription& option_description);
  OptionsDescriptionEasyInit add_options();

 private:
  std::string caption_;
  int long_form_only_index_;
  std::unordered_map<int, OptionDescription> options_;

  friend class OptionsDescriptionEasyInit;
  friend class Parser;
};

class VariableValue {
 public:
  VariableValue() = default;
  explicit VariableValue(std::any value);
  std::any& value();
  const std::any& value() const;
  bool empty() const;

  template <typename T>
  T& as() {
    return std::any_cast<T&>(value_);
  }

  template <typename T>
  const T& as() const {
    return std::any_cast<const T&>(value_);
  }

 private:
  std::any value_;
};

using VariablesMap = std::unordered_map<std::string, VariableValue>;

class Parser {
 public:
  VariablesMap parse_command_line(int argc, char** argv,
                                  const OptionsDescription& options_description) const;

 private:
  static error_t parse_option(int key, char* arg, ::argp_state* state);
};

}  // namespace args
}  // namespace holomq_examples
