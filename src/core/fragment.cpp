/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "holoscan/core/fragment.hpp"

#include <yaml-cpp/yaml.h>

#include <iterator>  // for std::back_inserter
#include <set>
#include <string>
#include <typeinfo>
#include <utility>
#include <variant>

#include "holoscan/core/arg.hpp"
#include "holoscan/core/config.hpp"
#include "holoscan/core/executors/gxf/gxf_executor.hpp"
#include "holoscan/core/graphs/flow_graph.hpp"
#include "holoscan/core/operator.hpp"

namespace holoscan {

Fragment& Fragment::name(const std::string& name) & {
  name_ = name;
  return *this;
}

Fragment&& Fragment::name(const std::string& name) && {
  name_ = name;
  return std::move(*this);
}

const std::string& Fragment::name() const {
  return name_;
}

Fragment& Fragment::application(Application* app) {
  app_ = app;
  return *this;
}

void Fragment::config(const std::string& config_file, const std::string& prefix) {
  (void)prefix;  // prefix is used for from_config() method.
  if (config_) { HOLOSCAN_LOG_WARN("Config object was already created. Overwriting..."); }
  config_ = make_config<Config>(config_file, prefix);
}

Config& Fragment::config() {
  if (!config_) { config_ = make_config<Config>(); }
  return *config_;
}

Graph& Fragment::graph() {
  if (!graph_) { graph_ = make_graph<FlowGraph>(); }
  return *graph_;
}

Executor& Fragment::executor() {
  if (!executor_) { executor_ = make_executor<gxf::GXFExecutor>(); }
  return *executor_;
}
ArgList Fragment::from_config(const std::string& key) {
  (void)key;
  auto& yaml_nodes = config().yaml_nodes();
  ArgList args;

  std::vector<std::string> key_parts;

  size_t pos = 0;
  while (pos != std::string::npos) {
    size_t next_pos = key.find_first_of('.', pos);
    if (next_pos == std::string::npos) { break; }
    key_parts.push_back(key.substr(pos, next_pos - pos));
    pos = next_pos + 1;
  }
  key_parts.push_back(key.substr(pos));

  size_t key_parts_size = key_parts.size();

  for (const auto& yaml_node : yaml_nodes) {
    if (yaml_node.IsMap()) {
      auto yaml_map = yaml_node.as<YAML::Node>();
      size_t key_index = 0;
      for (const auto& key_part : key_parts) {
        (void)key_part;
        if (yaml_map.IsMap()) {
          yaml_map.reset(yaml_map[key_part]);
          ++key_index;
        } else {
          break;
        }
      }
      if (!yaml_map || key_index < key_parts_size) {
        HOLOSCAN_LOG_ERROR("Unable to find the parameter item/map with key '{}'", key);
        continue;
      }

      const auto& parameters = yaml_map;

      if (parameters.IsScalar()) {
        const std::string& param_key = key;
        auto& value = parameters;
        args.add(Arg(param_key) = value);
        continue;
      }

      for (const auto& p : parameters) {
        const std::string param_key = p.first.as<std::string>();
        auto& value = p.second;
        args.add(Arg(param_key) = value);
      }
    }
  }

  return args;
}

void Fragment::add_operator(const std::shared_ptr<Operator>& op) {
  graph().add_operator(op);
}

void Fragment::add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Operator>& downstream_op) {
  add_flow(upstream_op, downstream_op, {});
}

void Fragment::add_flow(const std::shared_ptr<Operator>& upstream_op,
                        const std::shared_ptr<Operator>& downstream_op,
                        std::set<std::pair<std::string, std::string>> port_pairs) {
  auto port_map = std::make_shared<Graph::EdgeDataElementType>();

  auto upstream_op_spec = upstream_op->spec();
  if (upstream_op_spec == nullptr) {
    HOLOSCAN_LOG_ERROR("upstream_op_spec is nullptr");
    return;
  }
  auto downstream_op_spec = downstream_op->spec();
  if (downstream_op_spec == nullptr) {
    HOLOSCAN_LOG_ERROR("downstream_op_spec is nullptr");
    return;
  }

  auto& op_outputs = upstream_op_spec->outputs();
  auto& op_inputs = downstream_op_spec->inputs();
  if (port_pairs.empty()) {
    if (op_outputs.size() > 1) {
      std::vector<std::string> output_labels;
      for (const auto& [key, _] : op_outputs) { output_labels.push_back(key); }

      HOLOSCAN_LOG_ERROR(
          "The upstream operator has more than one output port ({}) so mapping should be specified "
          "explicitly!",
          fmt::join(output_labels, ", "));
      return;
    }
    if (op_inputs.size() > 1) {
      std::vector<std::string> input_labels;
      for (const auto& [key, _] : op_inputs) { input_labels.push_back(key); }
      HOLOSCAN_LOG_ERROR(
          "The downstream operator has more than one input port ({}) so mapping should be "
          "specified "
          "explicitly!",
          fmt::join(input_labels, ", "));
      return;
    }
    port_pairs.emplace("", "");
  }

  std::vector<std::string> output_labels;
  output_labels.reserve(port_pairs.size());

  // Convert port pairs to port map (set<pair<string, string>> -> map<string, set<string>>)
  for (auto& [key, value] : port_pairs) {
    if (port_map->find(key) == port_map->end()) {
      (*port_map)[key] = std::set<std::string, std::less<>>();
      output_labels.push_back(key);  // maintain the order of output labels
    }
    (*port_map)[key].insert(value);
  }

  // Verify that the upstream & downstream operators have the input and output ports specified by
  // the port_map
  if (op_outputs.size() == 1 && output_labels.size() != 1) {
    HOLOSCAN_LOG_ERROR(
        "The upstream operator({}) has only one port with label '{}' but port_map "
        "specifies {} labels({}) to the upstream operator's output port",
        upstream_op->name(),
        (*op_outputs.begin()).first,
        output_labels.size(),
        fmt::join(output_labels, ", "));
    return;
  }

  for (const auto& output_label : output_labels) {
    if (op_outputs.find(output_label) == op_outputs.end()) {
      if (op_outputs.size() == 1 && output_labels.size() == 1 && output_label == "") {
        // Set the default output port label
        (*port_map)[op_outputs.begin()->first] = std::move((*port_map)[output_label]);
        port_map->erase(output_label);
        // Update the output label
        output_labels[0] = op_outputs.begin()->first;
        break;
      }

      auto msg_buf = fmt::memory_buffer();
      for (const auto& [label, _] : op_outputs) {
        if (&label == &(op_outputs.begin()->first)) {
          fmt::format_to(std::back_inserter(msg_buf), "{}", label);
        } else {
          fmt::format_to(std::back_inserter(msg_buf), ", {}", label);
        }
      }
      HOLOSCAN_LOG_ERROR(
          "The upstream operator({}) does not have an output port with label '{}'. It should be "
          "one of ({:.{}})",
          upstream_op->name(),
          output_label,
          msg_buf.data(),
          msg_buf.size());
      return;
    }
  }

  for (const auto& output_label : output_labels) {
    auto& input_labels = (*port_map)[output_label];
    if (op_inputs.size() == 1 && input_labels.size() != 1) {
      HOLOSCAN_LOG_ERROR(
          "The downstream operator({}) has only one port with label '{}' but port_map "
          "specifies {} labels({}) to the downstream operator's input port",
          downstream_op->name(),
          (*op_inputs.begin()).first,
          input_labels.size(),
          fmt::join(input_labels, ", "));
      return;
    }

    for (auto& input_label : input_labels) {
      if (op_inputs.find(input_label) == op_inputs.end()) {
        if (op_inputs.size() == 1 && input_labels.size() == 1 && input_label == "") {
          // Set the default input port label.
          input_labels.clear();
          input_labels.insert(op_inputs.begin()->first);
          break;
        }

        // Support for the case where the destination input port label points to the
        // parameter name of the downstream operator, and the parameter type is
        // 'std::vector<holoscan::IOSpec*>'.
        // If we cannot find the input port with the specified label (e.g., 'receivers'),
        // then we need to find a downstream operator's parameter
        // (with 'std::vector<holoscan::IOSpec*>' type) and create a new input port
        // with a specific label ('<parameter name>:<index>'. e.g, 'receivers:0').
        auto& downstream_op_params = downstream_op_spec->params();
        if (downstream_op_params.find(input_label) != downstream_op_params.end()) {
          auto& downstream_op_param = downstream_op_params.at(input_label);
          if (std::type_index(downstream_op_param.type()) ==
              std::type_index(typeid(std::vector<holoscan::IOSpec*>))) {
            const std::any& any_value = downstream_op_param.value();
            auto& param = *std::any_cast<Parameter<std::vector<holoscan::IOSpec*>>*>(any_value);
            param.set_default_value();

            std::vector<holoscan::IOSpec*>& iospec_vector = param.get();

            // Create a new input port for this receivers parameter
            bool succeed = executor().add_receivers(downstream_op, input_label,
                                                    input_labels, iospec_vector);
            if (!succeed) {
              HOLOSCAN_LOG_ERROR(
                  "Failed to add receivers to the downstream operator({}) with label '{}'",
                  downstream_op->name(),
                  input_label);
              return;
            }
            continue;
          }
        }

        auto msg_buf = fmt::memory_buffer();
        for (const auto& [label, _] : op_inputs) {
          if (&label == &(op_inputs.begin()->first)) {
            fmt::format_to(std::back_inserter(msg_buf), "{}", label);
          } else {
            fmt::format_to(std::back_inserter(msg_buf), ", {}", label);
          }
        }
        HOLOSCAN_LOG_ERROR(
            "The downstream operator({}) does not have an input port with label '{}'. It should be "
            "one of ({:.{}})",
            downstream_op->name(),
            input_label,
            msg_buf.data(),
            msg_buf.size());
        return;
      }
    }
  }

  graph().add_flow(upstream_op, downstream_op, port_map);
}

void Fragment::compose() {}

void Fragment::run() {
  executor().run(graph());
}

}  // namespace holoscan
