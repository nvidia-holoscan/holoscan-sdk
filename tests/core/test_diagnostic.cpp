/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>

namespace holoscan {
namespace {

template <typename ContextT>
concept DiagnosticFormatContext = requires(const Diagnostic& diagnostic, const ContextT& context) {
  format_diagnostic(diagnostic, context);
};

static_assert(DiagnosticFormatContext<ExecutionGraphView>);
static_assert(!DiagnosticFormatContext<GraphView>);

class DiagnosticFormatOperator final : public Operator<> {
 public:
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

ExecutionGraphView compiled_graph(std::string_view vertex_name) {
  Graph graph{"diagnostic-format"};
  graph.op<DiagnosticFormatOperator>(vertex_name);
  const ExecutionPlan plan = compile(graph);
  if (!plan.ok()) {
    ADD_FAILURE() << plan.json();
  }
  return plan.graph();
}

TEST(DiagnosticFormat, RendersKnownVertexAndEveryPublicDetail) {
  const ExecutionGraphView graph = compiled_graph("sink");
  ASSERT_EQ(graph.vertices().size(), 1U);
  const Diagnostic diagnostic{
      .id = DiagnosticId{11U},
      .severity = DiagnosticSeverity::kError,
      .token = "RUNTIME_OPERATOR_CALLBACK_FAILED",
      .source = DiagnosticSource{.graph_generation = 3U,
                                 .graph = GraphId{2U},
                                 .vertex = graph.vertices().front().id,
                                 .port = PortId{5U}},
      .service = "operator_compute",
      .endpoint = "command",
      .json_pointer = "/operators/sink",
      .required = "success",
      .provided = "kFailure",
      .cause_id = DiagnosticId{4U},
  };

  const std::string expected =
      "error RUNTIME_OPERATOR_CALLBACK_FAILED [id=11, graph_generation=3, graph=2, "
      "vertex=sink, vertex_id=" +
      std::to_string(graph.vertices().front().id.value) +
      ", port=5, service=operator_compute, endpoint=command, "
      "json_pointer=/operators/sink, required=success, provided=kFailure, cause_id=4]";
  EXPECT_EQ(format_diagnostic(diagnostic, graph), expected);
}

TEST(DiagnosticFormat, RendersUnknownVertexWithoutInventingAName) {
  const Diagnostic diagnostic{
      .severity = DiagnosticSeverity::kWarning,
      .token = "DEADLINE_MISS",
      .source = DiagnosticSource{.vertex = VertexId{99U}},
  };

  EXPECT_EQ(format_diagnostic(diagnostic, ExecutionGraphView{}),
            "warning DEADLINE_MISS [vertex=unknown, vertex_id=99]");
}

TEST(DiagnosticFormat, RendersDiagnosticWithoutLocationOrDetails) {
  const Diagnostic diagnostic{
      .severity = DiagnosticSeverity::kError,
      .token = "RUNTIME_STARTUP_FAILED",
  };

  EXPECT_EQ(format_diagnostic(diagnostic, ExecutionGraphView{}), "error RUNTIME_STARTUP_FAILED");
}

}  // namespace
}  // namespace holoscan
