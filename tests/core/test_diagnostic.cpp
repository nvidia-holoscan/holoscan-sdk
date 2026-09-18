/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/execution_plan.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/time/manual_clock.hpp>

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

class DiagnosticFormatSource final : public Operator<> {
 public:
  explicit DiagnosticFormatSource(std::string port_name) : port_name_(std::move(port_name)) {}
  void setup(OperatorSpec& spec) override { spec.output(output, port_name_); }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Output<std::uint32_t> output;

 private:
  std::string port_name_;
};

class DiagnosticFormatSink final : public Operator<> {
 public:
  explicit DiagnosticFormatSink(std::string port_name) : port_name_(std::move(port_name)) {}
  void setup(OperatorSpec& spec) override { spec.input(input, port_name_); }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
  Input<std::uint32_t> input;

 private:
  std::string port_name_;
};

class DiagnosticFormatGraph final : public Graph {
 public:
  DiagnosticFormatGraph(std::string output_name, std::string input_name)
      : output_name_(std::move(output_name)), input_name_(std::move(input_name)) {}

 protected:
  void compose() override {
    set_default_clock(add_clock<ManualClock>("timeline", std::int64_t{0}));
    const auto source = op<DiagnosticFormatSource>("source", output_name_);
    const auto sink = op<DiagnosticFormatSink>("sink", input_name_);
    (void)add_flow(source->output, sink->input);
  }

 private:
  std::string output_name_;
  std::string input_name_;
};

ExecutionGraphView compiled_endpoints(std::string output_name = "output",
                                      std::string input_name = "input") {
  Graph graph{"diagnostic-endpoints"};
  (void)graph.embed<DiagnosticFormatGraph>(
      "pipeline", std::move(output_name), std::move(input_name));
  const ExecutionPlan plan = compile(graph);
  if (!plan.ok()) {
    ADD_FAILURE() << plan.json();
  }
  return plan.graph();
}

DiagnosticSource endpoint_source(const ExecutionGraphView& graph, std::string_view name) {
  const auto vertices = graph.vertices();
  const auto vertex = std::ranges::find(vertices, name, &VertexRecord::name);
  if (vertex == vertices.end()) {
    ADD_FAILURE() << "Missing fixture vertex " << name;
    return {};
  }
  const auto ports = graph.ports();
  const auto port = std::ranges::find(ports, vertex->id, &PortRecord::owner);
  if (port == ports.end()) {
    ADD_FAILURE() << "Missing fixture port for " << name;
    return {};
  }
  return {.graph_generation = graph.generation(),
          .graph = vertex->graph,
          .vertex = vertex->id,
          .port = PortId{port->id.value}};
}

Diagnostic endpoint_diagnostic(const ExecutionGraphView& graph) {
  return {.token = "PORT_CONTEXT",
          .source = endpoint_source(graph, "source"),
          .detail = DiagnosticDetail{.error = Error{ErrorCode::kSuccess},
                                     .related_source = endpoint_source(graph, "sink")}};
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

// REQ: EXPECTED-030
TEST(DiagnosticFormat, RendersStructuralRelatedEndpointWithoutGraph) {
  const Diagnostic diagnostic{.token = "RELATED_ENDPOINT",
                              .detail = DiagnosticDetail{.error = Error{ErrorCode::kSuccess},
                                                         .related_source = {.graph_generation = 3U,
                                                                            .graph = GraphId{2U},
                                                                            .vertex = VertexId{7U},
                                                                            .port = PortId{5U}}}};
  EXPECT_EQ(format_diagnostic(diagnostic, ExecutionGraphView{}),
            "error RELATED_ENDPOINT [related_graph_generation=3, related_graph=2, "
            "related_vertex_id=7, related_port_id=5, related_operator=unknown]");
}

// REQ: EXPECTED-030
TEST(DiagnosticFormat, ResolvesBothStructuralEndpointsInCompiledGraph) {
  const auto graph = compiled_endpoints();
  const auto diagnostic = endpoint_diagnostic(graph);
  ASSERT_TRUE(diagnostic.source.port);
  if (!diagnostic.detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  const auto& detail = *diagnostic.detail;
  const auto& related = detail.related_source;
  ASSERT_TRUE(related.port);
  ASSERT_NE(diagnostic.source.vertex, related.vertex);
  ASSERT_NE(diagnostic.source.port, related.port);
  const auto report = format_diagnostic(diagnostic, graph);
  EXPECT_NE(report.find("vertex=pipeline/source, vertex_id=" +
                        std::to_string(diagnostic.source.vertex.value)),
            std::string::npos);
  EXPECT_NE(
      report.find("port=" + std::to_string(diagnostic.source.port.value()) + ", port_name=output"),
      std::string::npos);
  EXPECT_NE(report.find("related_graph_generation=" + std::to_string(related.graph_generation)),
            std::string::npos);
  EXPECT_NE(report.find("related_graph=" + std::to_string(related.graph.value)), std::string::npos);
  EXPECT_NE(report.find("related_vertex_id=" + std::to_string(related.vertex.value)),
            std::string::npos);
  EXPECT_NE(report.find("related_port_id=" + std::to_string(related.port.value())),
            std::string::npos);
  EXPECT_NE(report.find("related_operator=pipeline/sink, related_port=input"), std::string::npos);
}

// REQ: EXPECTED-030
TEST(DiagnosticFormat, RendersPartialRelatedIdentitiesIndependently) {
  const std::vector<std::pair<DiagnosticSource, std::string>> cases{
      {{}, ""},
      {{.graph_generation = 3U}, " [related_graph_generation=3]"},
      {{.graph = GraphId{2U}}, " [related_graph=2]"},
      {{.vertex = VertexId{99U}}, " [related_vertex_id=99, related_operator=unknown]"},
      {{.port = PortId{99U}}, " [related_port_id=99]"},
  };
  for (const auto& graph : {ExecutionGraphView{}, compiled_endpoints()}) {
    for (const auto& [source, fields] : cases) {
      SCOPED_TRACE(fields);
      const Diagnostic diagnostic{.token = "RELATED_ENDPOINT",
                                  .detail = DiagnosticDetail{.error = Error{ErrorCode::kSuccess},
                                                             .related_source = source}};
      EXPECT_EQ(format_diagnostic(diagnostic, graph), "error RELATED_ENDPOINT" + fields);
    }
  }
  const auto graph = compiled_endpoints();
  const auto sink = endpoint_source(graph, "sink");
  Diagnostic diagnostic{.token = "RELATED_ENDPOINT",
                        .detail = DiagnosticDetail{.error = Error{ErrorCode::kSuccess},
                                                   .related_source = {.port = sink.port}}};
  EXPECT_EQ(format_diagnostic(diagnostic, graph),
            "error RELATED_ENDPOINT [related_port_id=" + std::to_string(sink.port.value()) +
                ", related_port=input]");
  diagnostic.detail->related_source = {.vertex = sink.vertex};
  EXPECT_EQ(format_diagnostic(diagnostic, graph),
            "error RELATED_ENDPOINT [related_vertex_id=" + std::to_string(sink.vertex.value) +
                ", related_operator=pipeline/sink]");
}

// REQ: EXPECTED-030
TEST(DiagnosticFormat, DoesNotRepeatEqualPrimaryPortNames) {
  const auto graph = compiled_endpoints("output", "output");
  auto diagnostic = endpoint_diagnostic(graph);
  if (!diagnostic.detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  auto& detail = *diagnostic.detail;
  detail.port_name = "output";
  detail.related_port_name = "output";
  const auto report = format_diagnostic(diagnostic, graph);
  const auto first = report.find(", port_name=output");
  ASSERT_NE(first, std::string::npos);
  EXPECT_EQ(first, report.rfind(", port_name=output"));
  EXPECT_EQ(report.find("captured_port_name="), std::string::npos);
  EXPECT_NE(report.find("related_port=output"), std::string::npos);
}

// REQ: EXPECTED-030
TEST(DiagnosticFormat, PreservesOwnedEndpointNamesWithoutGraph) {
  Diagnostic diagnostic{
      .token = "OWNED_ENDPOINTS",
      .source = {.vertex = VertexId{1U}, .port = PortId{2U}},
      .detail = DiagnosticDetail{.error = Error{ErrorCode::kSuccess},
                                 .operator_path = "retained/source",
                                 .port_name = "output",
                                 .related_source = {.vertex = VertexId{3U}, .port = PortId{4U}},
                                 .related_operator_path = "retained/sink",
                                 .related_port_name = "input"}};
  auto copied = diagnostic;
  diagnostic.detail.reset();
  const auto report = format_diagnostic(copied, ExecutionGraphView{});
  EXPECT_NE(report.find("vertex=retained/source"), std::string::npos);
  EXPECT_NE(report.find("port_name=output"), std::string::npos);
  EXPECT_NE(report.find("related_vertex_id=3, related_port_id=4"), std::string::npos);
  EXPECT_NE(report.find("related_operator=retained/sink, related_port=input"), std::string::npos);
  ASSERT_TRUE(copied.detail);
  copied.detail->related_operator_path.clear();
  copied.detail->related_source = {};
  EXPECT_NE(format_diagnostic(copied, ExecutionGraphView{}).find("related_port=input"),
            std::string::npos);
}

// REQ: EXPECTED-030
TEST(DiagnosticFormat, KeepsDistinctCapturedPortNamesAfterEscaping) {
  // A literal escape spelling and an actual newline render alike but are different evidence.
  const auto graph = compiled_endpoints("output\\x0a");
  auto diagnostic = endpoint_diagnostic(graph);
  if (!diagnostic.detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  auto& detail = *diagnostic.detail;
  detail.port_name = "output\n";
  const auto report = format_diagnostic(diagnostic, graph);
  EXPECT_NE(report.find("port_name=output\\x0a"), std::string::npos);
  EXPECT_NE(report.find("captured_port_name=output\\x0a"), std::string::npos);
  EXPECT_EQ(report.find('\n'), std::string::npos);
  detail.port_name = "captured-output";
  EXPECT_NE(format_diagnostic(diagnostic, graph).find("captured_port_name=captured-output"),
            std::string::npos);
  EXPECT_NE(format_diagnostic(diagnostic, ExecutionGraphView{}).find("port_name=captured-output"),
            std::string::npos);

  // Keep the compiled name within the authoring limit while the captured name is truncated.
  const std::string prefix(255U, 'x');
  const auto long_name_graph = compiled_endpoints(prefix);
  auto long_name_diagnostic = endpoint_diagnostic(long_name_graph);
  if (!long_name_diagnostic.detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  auto& long_name_detail = *long_name_diagnostic.detail;
  long_name_detail.port_name = prefix + std::string(5000U, 'y');
  const auto long_name_report = format_diagnostic(long_name_diagnostic, long_name_graph);
  EXPECT_NE(long_name_report.find("port_name=" + prefix + ","), std::string::npos);
  EXPECT_NE(long_name_report.find("captured_port_name=" + prefix + "y"), std::string::npos);
  EXPECT_NE(long_name_report.find("[truncated]"), std::string::npos);
}

// REQ: EXPECTED-030
TEST(DiagnosticFormat, KeepsOwnedRelatedNamesAlongsideStructuralIdentity) {
  const auto graph = compiled_endpoints();
  auto diagnostic = endpoint_diagnostic(graph);
  if (!diagnostic.detail.has_value()) {
    FAIL() << "Expected diagnostic detail";
  }
  auto& detail = *diagnostic.detail;
  detail.related_operator_path = "captured/sink";
  detail.related_port_name = "captured-input";
  const auto report = format_diagnostic(diagnostic, graph);
  EXPECT_NE(report.find("related_vertex_id=" + std::to_string(detail.related_source.vertex.value)),
            std::string::npos);
  EXPECT_NE(report.find("related_operator=captured/sink, related_port=captured-input"),
            std::string::npos);
  detail.related_port_name.clear();
  EXPECT_NE(format_diagnostic(diagnostic, graph)
                .find("related_operator=captured/sink, related_port=input"),
            std::string::npos);
  detail.related_operator_path.clear();
  detail.related_port_name = "captured-input";
  EXPECT_NE(format_diagnostic(diagnostic, graph)
                .find("related_operator=pipeline/sink, related_port=captured-input"),
            std::string::npos);
}

// REQ: EXPECTED-030
TEST(DiagnosticFormat, EscapesAndBoundsRelatedEndpointNamesInReports) {
  const Diagnostic diagnostic{
      .token = "RELATED_ENDPOINT",
      .detail = DiagnosticDetail{.error = Error{ErrorCode::kSuccess},
                                 .related_source = {.vertex = VertexId{7U}, .port = PortId{5U}},
                                 .related_operator_path = "retained\n\x1b[31m",
                                 .related_port_name = std::string(5000U, 'x')}};
  const auto line = format_diagnostic(diagnostic, ExecutionGraphView{});
  EXPECT_EQ(line.find('\n'), std::string::npos);
  EXPECT_EQ(line.find('\x1b'), std::string::npos);
  EXPECT_NE(line.find("retained\\x0a\\x1b[31m"), std::string::npos);
  EXPECT_NE(line.find("[truncated]"), std::string::npos);
  const std::vector<Diagnostic> diagnostics(8U, diagnostic);
  const auto report = format_diagnostics(diagnostics, ExecutionGraphView{});
  EXPECT_LE(report.size(), 16384U);
  EXPECT_NE(report.find("related_vertex_id=7"), std::string::npos);
  EXPECT_NE(report.find("diagnostic report truncated"), std::string::npos);
}

// REQ: EXPECTED-030
TEST(DiagnosticFormat, OwnsAndRendersCallbackCauseWithoutGraph) {
  Diagnostic diagnostic{
      .token = "RUNTIME_LIFECYCLE_CALLBACK_FAILED",
      .detail =
          DiagnosticDetail{
              .operation = DiagnosticOperation::kLifecycle,
              .error = Error{ErrorCode::kInvalidArgument, "RGB unsupported; requires RGBA"},
              .stage = LifecycleStage::kStart,
              .phase = LifecycleTransitionPhase::kStartupForward,
              .operator_path = "pipeline/display",
              .causes = {Error{ErrorCode::kFailure, "backend detail\n\x1b[31m"}},
          },
  };
  const auto copied = diagnostic;
  diagnostic.detail.reset();
  const auto report = format_diagnostic(copied, ExecutionGraphView{});
  EXPECT_NE(report.find("pipeline/display"), std::string::npos);
  EXPECT_NE(report.find("kStart"), std::string::npos);
  EXPECT_NE(report.find("kStartupForward"), std::string::npos);
  EXPECT_NE(report.find("kInvalidArgument: RGB unsupported; requires RGBA"), std::string::npos);
  EXPECT_EQ(report.find('\n'), std::string::npos);
  EXPECT_EQ(report.find('\x1b'), std::string::npos);
  EXPECT_NE(report.find("backend detail"), std::string::npos);
}

TEST(DiagnosticFormat, RendersTeardownContextWithoutInventingACause) {
  const Diagnostic diagnostic{.token = "RUNTIME_LANE_JOIN_FAILED",
                              .detail = DiagnosticDetail{.operation = DiagnosticOperation::kRuntime,
                                                         .error = Error{ErrorCode::kSuccess},
                                                         .role = DiagnosticFailureRole::kTeardown}};
  const auto report = format_diagnostic(diagnostic, ExecutionGraphView{});
  EXPECT_NE(report.find("role=kTeardown"), std::string::npos);
  EXPECT_EQ(report.find("cause="), std::string::npos);
}

TEST(DiagnosticFormat, BoundsReportAndMarksTruncation) {
  std::vector<Diagnostic> diagnostics(64U);
  for (auto& diagnostic : diagnostics) {
    diagnostic.token = "RUNTIME_OPERATOR_CALLBACK_FAILED";
    diagnostic.detail =
        DiagnosticDetail{.error = Error{ErrorCode::kFailure, std::string(4096, 'x')}};
  }
  const auto report = format_diagnostics(diagnostics, ExecutionGraphView{});
  EXPECT_LE(report.size(), 16384U);
  EXPECT_NE(report.find("diagnostic report truncated"), std::string::npos);
}

// REQ: EXPECTED-030
TEST(DiagnosticFormat, KeepsOriginalIndicesVisibleWhenErrorsPrecedeWarnings) {
  // A warning arrives first, then the initiating failure and its linked cleanup failure.
  const std::vector<Diagnostic> diagnostics{
      Diagnostic{.severity = DiagnosticSeverity::kWarning, .token = "WARNING"},
      Diagnostic{.token = "PRIMARY"},
      Diagnostic{.token = "CLEANUP",
                 .detail = DiagnosticDetail{.role = DiagnosticFailureRole::kTeardown,
                                            .primary_diagnostic_index = 1U}},
  };
  const auto report = format_diagnostics(diagnostics, ExecutionGraphView{});
  const auto first_end = report.find('\n');
  ASSERT_NE(first_end, std::string::npos);
  const auto second_end = report.find('\n', first_end + 1U);
  ASSERT_NE(second_end, std::string::npos);
  const auto primary = report.substr(0U, first_end);
  const auto cleanup = report.substr(first_end + 1U, second_end - first_end - 1U);
  const auto warning = report.substr(second_end + 1U);
  EXPECT_TRUE(primary.starts_with("error PRIMARY"));
  EXPECT_NE(primary.find("diagnostic_index=1"), std::string::npos);
  EXPECT_NE(cleanup.find("primary_diagnostic_index=1"), std::string::npos);
  EXPECT_NE(cleanup.find("diagnostic_index=2"), std::string::npos);
  EXPECT_TRUE(warning.starts_with("warning WARNING"));
  EXPECT_NE(warning.find("diagnostic_index=0"), std::string::npos);
}

TEST(DiagnosticFormat, TerminalCausePrecedesVerboseWarnings) {
  std::vector<Diagnostic> diagnostics(14U);
  for (auto& diagnostic : diagnostics) {
    diagnostic.severity = DiagnosticSeverity::kWarning;
    diagnostic.token = "DEADLINE_MISS";
    diagnostic.provided = std::string(4096U, 'x');
  }
  diagnostics.push_back(Diagnostic{
      .token = "RUNTIME_OPERATOR_CALLBACK_FAILED",
      .detail = DiagnosticDetail{.error = Error{ErrorCode::kFailure, "original terminal cause"}},
  });
  const auto report = format_diagnostics(diagnostics, ExecutionGraphView{});
  EXPECT_TRUE(report.starts_with("error RUNTIME_OPERATOR_CALLBACK_FAILED"));
  EXPECT_NE(report.find("original terminal cause"), std::string::npos);
  EXPECT_NE(report.find("truncated"), std::string::npos);
  EXPECT_LE(report.size(), 16384U);
}

}  // namespace
}  // namespace holoscan
