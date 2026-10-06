/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include <holoscan/core/graph.hpp>
#include <holoscan/time/manual_clock.hpp>

#include "core/graph_compiler_access.hpp"

namespace holoscan {
namespace {

// EmbeddingOp: minimal operator with one typed input, one typed output, and a readable source;
// used as the vertex type in every graph and subgraph topology test.
class EmbeddingOp final : public Operator<> {
 public:
  explicit EmbeddingOp(std::string label = {}) : label_(std::move(label)) {}

  void setup(OperatorSpec& spec) override {
    spec.input(input, "input");
    spec.output(output, "output");
    spec.readable_source(readable, "readable");
  }

  expected<void, Error> compute(ExecutionContext&) override { return {}; }

  Input<std::uint32_t> input;
  Output<std::uint32_t> output;
  ReadableSource readable;
  std::string label_;
};

// ReusableGraph: two-node linear pipeline (first → second) with exported input/output ports,
// a child region, two clocks with a conversion, lifecycle ordering, and topic import/export.
// Embedded repeatedly to test independent factory instantiation and canonical identity.
class ReusableGraph final : public Graph {
 public:
  explicit ReusableGraph(std::string label = "reusable") : label_(std::move(label)) {}

  [[nodiscard]] const OperatorHandle<EmbeddingOp>& first() const {
    // Core completes compose before publishing the owning GraphHandle.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return first_.value();
  }

  [[nodiscard]] const OperatorHandle<EmbeddingOp>& second() const {
    // Core completes compose before publishing the owning GraphHandle.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return second_.value();
  }

  Input<std::uint32_t> input{"input"};
  Output<std::uint32_t> output{"output"};

 protected:
  void compose() override {
    first_ = op<EmbeddingOp>("first", label_ + "-first");
    second_ = op<EmbeddingOp>("second", label_ + "-second");
    (void)add_flow(first_->operator->()->output, second_->operator->()->input);
    export_input(input, first_->operator->()->input);
    export_output(output, second_->operator->()->output);
    region("child_region")
        .add(*first_)
        .constrain(RegionConstraint::bounded_memory())
        .profile(Profile::kRealTime);
    lifecycle()
        .before(*first_, *second_, LifecycleStage::kStart)
        .allow_restart_request(*first_, *second_);
    const ClockKey source_clock = add_clock<ManualClock>("source_clock", std::int64_t{1});
    const ClockKey target_clock = add_clock<ManualClock>("target_clock", std::int64_t{2});
    set_default_clock(source_clock);
    add_clock_conversion(source_clock, target_clock, AffineClockConversion{.offset_ns = 3});
    (void)import_topic<std::uint32_t>("holomq://embedded-import");
    (void)export_topic<std::uint32_t>("holomq://embedded-export");
  }

 private:
  std::string label_;
  std::optional<OperatorHandle<EmbeddingOp>> first_;
  std::optional<OperatorHandle<EmbeddingOp>> second_;
};

// AuthoredOrderLeafGraph: single-node graph labelled at construction; used as a lightweight
// sibling child to verify that authored record order survives embedding unchanged.
class AuthoredOrderLeafGraph final : public Graph {
 public:
  explicit AuthoredOrderLeafGraph(std::string label) : label_(std::move(label)) {}

 protected:
  void compose() override { (void)op<EmbeddingOp>("node", label_); }

 private:
  std::string label_;
};

class CaptureEligibleEmbeddingOp final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override { spec.cuda_graph_capture_eligible(); }
  expected<void, Error> compute(ExecutionContext&) override { return {}; }
};

class CaptureEligibleGraph final : public Graph {
 protected:
  void compose() override { (void)op<CaptureEligibleEmbeddingOp>("capture"); }
};

class MoveOnlyGraph final : public Graph {
 public:
  explicit MoveOnlyGraph(std::unique_ptr<int> value) : value_(std::move(value)) {}

  [[nodiscard]] int value() const noexcept { return *value_; }

 protected:
  void compose() override {}

 private:
  std::unique_ptr<int> value_;
};

class AuthoringHandleGraph final : public Graph {
 public:
  explicit AuthoringHandleGraph(const OperatorHandle<EmbeddingOp>&) {}

 protected:
  void compose() override {}
};

class ImportedClockGraph final : public Graph {
 public:
  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }

 protected:
  void compose() override {
    clock_import_ = import_clock("timeline");
    set_default_clock(clock_import_);
  }

 private:
  ClockKey clock_import_;
};

class ForwardedClockGraph final : public Graph {
 public:
  [[nodiscard]] ClockKey clock_import() const noexcept { return clock_import_; }

 protected:
  void compose() override {
    clock_import_ = import_clock("outer_timeline");
    set_default_clock(clock_import_);
    auto leaf = embed<ImportedClockGraph>("leaf");
    bind_clock(leaf, leaf->clock_import(), clock_import_);
  }

 private:
  ClockKey clock_import_;
};

// NonLexicalAuthoredGraph: authors all record families (vertices, edges, subgraphs, regions,
// partitions, clocks, topics, lifecycle) in reverse-alphabetical order to detect
// canonical-sort regressions.
// REQ: GRAPH-022 — authored position must survive embedding unchanged.
class NonLexicalAuthoredGraph final : public Graph {
 public:
  Output<std::uint32_t> z_output{"z_output"};
  Output<std::uint32_t> a_output{"a_output"};

 protected:
  void compose() override {
    auto z = op<EmbeddingOp>("z");
    auto a = op<EmbeddingOp>("a");
    (void)add_flow(z->output, a->input);
    export_output(z_output, z->output);
    export_output(a_output, a->output);
    (void)embed<AuthoredOrderLeafGraph>("z_child", "z");
    (void)embed<AuthoredOrderLeafGraph>("a_child", "a");
    region("z_region").add(z);
    region("a_region").add(a);
    region("mixed_region").add(z).add(a);
    partition("serial").add(z).add(a);
    const ClockKey z_clock = add_clock<ManualClock>("z_clock", std::int64_t{1});
    const ClockKey a_clock = add_clock<ManualClock>("a_clock", std::int64_t{2});
    add_clock_conversion(z_clock, a_clock, AffineClockConversion{});
    lifecycle()
        .before(z, a, LifecycleStage::kStop)
        .before(z, a, LifecycleStage::kStart)
        .allow_restart_request(z, a)
        .allow_restart_request(a, z);
    (void)import_topic<std::uint32_t>("holomq://z-topic");
    (void)export_topic<std::uint32_t>("holomq://a-topic");
  }
};

// NestedGraph: one-level wrapper around ReusableGraph; embeds "leaf" and re-exports its
// input/output ports, producing a two-level-deep path (e.g. "left/leaf").
class NestedGraph final : public Graph {
 public:
  [[nodiscard]] const GraphHandle<ReusableGraph>& leaf() const {
    // Core completes compose before publishing the owning GraphHandle.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return leaf_.value();
  }

  Input<std::uint32_t> input{"input"};
  Output<std::uint32_t> output{"output"};

 protected:
  void compose() override {
    leaf_ = embed<ReusableGraph>("leaf", "nested");
    export_input(input, leaf_->operator->()->input);
    export_output(output, leaf_->operator->()->output);
  }

 private:
  std::optional<GraphHandle<ReusableGraph>> leaf_;
};

// OpaqueNestedGraph: wraps ReusableGraph without re-exporting its ports; the inner
// interface is opaque to the parent, verifying unexported port rejection at the parent level.
class OpaqueNestedGraph final : public Graph {
 public:
  [[nodiscard]] const GraphHandle<ReusableGraph>& leaf() const {
    // Core completes compose before publishing the owning GraphHandle.
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    return leaf_.value();
  }

 protected:
  void compose() override { leaf_ = embed<ReusableGraph>("leaf", "opaque"); }

 private:
  std::optional<GraphHandle<ReusableGraph>> leaf_;
};

// PartiallyExportedGraph: has two public C++ input ports but only one registered export;
// confirms that public C++ visibility alone does not grant parent wiring authority.
class PartiallyExportedGraph final : public Graph {
 public:
  Input<std::uint32_t> exported_input{"exported_input"};
  Input<std::uint32_t> unexported_input{"unexported_input"};
  Output<std::uint32_t> unexported_output{"unexported_output"};

 protected:
  void compose() override {
    auto node = op<EmbeddingOp>("node");
    export_input(exported_input, node->input);
  }
};

// ConstructorFailureGraph: throws in its constructor before compose() runs; both
// std::runtime_error and unknown-exception paths must roll back parent state atomically.
class ConstructorFailureGraph final : public Graph {
 public:
  explicit ConstructorFailureGraph(bool unknown_exception) {
    if (unknown_exception)
      throw 7;
    throw std::runtime_error{"embedded constructor failure"};
  }
};

// ComposeFailureGraph: creates a partial draft vertex then throws inside compose(); both
// std::runtime_error and unknown-exception paths must leave parent state unchanged.
class ComposeFailureGraph final : public Graph {
 public:
  explicit ComposeFailureGraph(bool unknown_exception) : unknown_exception_(unknown_exception) {}

 protected:
  void compose() override {
    (void)op<EmbeddingOp>("discarded");
    if (unknown_exception_)
      throw 7;
    throw std::runtime_error{"embedded compose failure"};
  }

 private:
  bool unknown_exception_{};
};

// ConflictingEndpointGraph: valid standalone child whose endpoint_name conflicts with the
// parent's; exercises the post-composition registration validation path (REQ: GRAPH-020).
class ConflictingEndpointGraph final : public Graph {
 protected:
  void compose() override {
    TopicOptions options;
    options.endpoint_name = "shared_endpoint";
    (void)import_topic<std::uint32_t>("holomq://child-topic", options);
  }
};

// DirectRecursiveGraph: embeds itself; verifies that direct self-recursion is detected and
// rejected with GRAPH_EMBED_CYCLE before any parent state is published.
class DirectRecursiveGraph final : public Graph {
 protected:
  void compose() override { (void)embed<DirectRecursiveGraph>("recursive"); }
};

class IndirectRecursiveGraphB;

// IndirectRecursiveGraphA / IndirectRecursiveGraphB: mutual recursion pair (A embeds B,
// B embeds A); tests indirect cycle detection that walks the full pending-embed stack.
class IndirectRecursiveGraphA final : public Graph {
 protected:
  void compose() override;
};

class IndirectRecursiveGraphB final : public Graph {
 protected:
  void compose() override;
};

void IndirectRecursiveGraphA::compose() {
  (void)embed<IndirectRecursiveGraphB>("b");
}

void IndirectRecursiveGraphB::compose() {
  (void)embed<IndirectRecursiveGraphA>("a");
}

struct HostedGraphTypeIdentity {
  const void* value{};
};

// LanguageHostedGraph models one C++ adapter hosting multiple logical graph types. Its explicit
// identity lets the recursion guard distinguish valid cross-type nesting without admitting a
// logical self-cycle.
// Representative test: SubgraphEmbedding.HostedTypeIdentityPreservesRecursionChecks.
class LanguageHostedGraph final : public Graph {
 public:
  LanguageHostedGraph(const HostedGraphTypeIdentity& identity,
                      const std::vector<HostedGraphTypeIdentity>& descendants)
      : identity_(identity), descendants_(descendants) {}

 protected:
  [[nodiscard]] const void* embedding_type_identity() const noexcept override {
    return identity_.value;
  }

  void compose() override {
    if (!descendants_.empty()) {
      const std::vector<HostedGraphTypeIdentity> remaining(descendants_.begin() + 1U,
                                                           descendants_.end());
      (void)embed<LanguageHostedGraph>("child", descendants_.front(), remaining);
    }
  }

 private:
  HostedGraphTypeIdentity identity_;
  std::vector<HostedGraphTypeIdentity> descendants_;
};

// Executes callable(), catches the first RuntimeError, and returns it as an Error value.
// If no exception is thrown the test fails immediately via ADD_FAILURE().
template <typename Callable>
Error capture_graph_error(Callable&& callable) {
  try {
    std::forward<Callable>(callable)();
  } catch (const RuntimeError& error) {
    return make_error(error);
  }
  ADD_FAILURE() << "expected RuntimeError";
  return Error{ErrorCode::kSuccess};
}

// Asserts that every size counter in a GraphView matches the before-snapshot; used as the
// atomicity invariant after a failed embed or lifecycle mutation attempt.
void expect_same_authoring_state(const GraphView& before, const GraphView& after) {
  EXPECT_EQ(after.generation(), before.generation());
  EXPECT_EQ(after.identity(), before.identity());
  EXPECT_EQ(after.graphs().size(), before.graphs().size());
  EXPECT_EQ(after.vertices().size(), before.vertices().size());
  EXPECT_EQ(after.ports().size(), before.ports().size());
  EXPECT_EQ(after.edges().size(), before.edges().size());
  EXPECT_EQ(after.interface_bindings().size(), before.interface_bindings().size());
  EXPECT_EQ(after.external_topics().size(), before.external_topics().size());
  EXPECT_EQ(after.readiness_sources().size(), before.readiness_sources().size());
  EXPECT_EQ(after.regions().size(), before.regions().size());
  EXPECT_EQ(after.partitions().size(), before.partitions().size());
  EXPECT_EQ(after.clocks().size(), before.clocks().size());
  EXPECT_EQ(after.clock_imports().size(), before.clock_imports().size());
  EXPECT_EQ(after.clock_bindings().size(), before.clock_bindings().size());
  EXPECT_EQ(after.clock_conversions().size(), before.clock_conversions().size());
  EXPECT_EQ(after.lifecycle_dependencies().size(), before.lifecycle_dependencies().size());
  EXPECT_EQ(after.lifecycle_restart_permissions().size(),
            before.lifecycle_restart_permissions().size());
}

// Converts a GraphRecord's (or optional GraphInstancePath's) path segments into a plain
// string vector for readable EXPECT_EQ comparisons.
std::vector<std::string> decoded_path(const GraphRecord& graph) {
  if (!graph.path)
    return {};
  return {graph.path->segments().begin(), graph.path->segments().end()};
}

std::vector<std::string> decoded_path(const std::optional<GraphInstancePath>& path) {
  if (!path)
    return {};
  return {path->segments().begin(), path->segments().end()};
}

// Returns the GraphId of the embedded sub-graph at the given path segments, or fails the
// test immediately if no such graph exists in the view.
GraphId graph_id_at_path(const GraphView& view, const std::vector<std::string>& path) {
  const auto found = std::find_if(view.graphs().begin(),
                                  view.graphs().end(),
                                  [&](const auto& graph) { return decoded_path(graph) == path; });
  if (found == view.graphs().end()) {
    ADD_FAILURE() << "embedded Graph path was not found";
    return {};
  }
  return found->id;
}

// Looks up the VertexId of a named operator vertex within a specific graph scope, or fails
// the test if no such vertex is found.
VertexId vertex_id_by_name(const GraphView& view, GraphId graph, std::string_view name) {
  const auto found =
      std::find_if(view.vertices().begin(), view.vertices().end(), [&](const auto& vertex) {
        return vertex.graph == graph && vertex.name == name;
      });
  if (found == view.vertices().end()) {
    ADD_FAILURE() << "embedded Operator vertex was not found";
    return {};
  }
  return found->id;
}

// Formats a GraphRecord's path as a display string: "_root" for the root graph, or
// "/seg1/seg2/..." for embedded graphs. Used as a key in graph-path → label maps.
std::string decoded_path_text(const GraphRecord& graph) {
  if (!graph.path)
    return "_root";
  std::string result;
  for (const auto& segment : graph.path->segments()) {
    result.push_back('/');
    result.append(segment);
  }
  return result;
}

// Returns "<default>" for the default ClockKey or the key's name as a std::string.
std::string clock_key_text(const ClockKey& key) {
  return key.is_default() ? "<default>" : std::string(key.name());
}

// Casts an enum to its underlying integer type for use in ostringstream record building.
template <typename Enum>
constexpr auto enum_value(Enum value) {
  return static_cast<std::underlying_type_t<Enum>>(value);
}

// Serializes every record in a GraphSnapshotData (graphs, vertices, ports, edges, bindings,
// topics, readiness, regions, partitions, clocks, conversions, lifecycle) into a string vector
// using
// canonical ID ordering. Compared pairwise to detect insertion-order regressions.
// The authoring hash covers every field in every canonical record. This companion
// signature makes ordering and all inter-record identity remaps independently
// visible, so a hash comparison cannot hide an insertion-order regression.
std::vector<std::string> canonical_record_order(const detail::GraphSnapshotData& snapshot) {
  std::vector<std::string> result;
  std::ostringstream record;
  auto append = [&](std::string_view kind) {
    result.push_back(std::string(kind) + '|' + record.str());
    record.str({});
    record.clear();
  };
  for (const auto& graph : snapshot.graphs) {
    record << graph.id.value << '|' << (graph.parent ? graph.parent->value : 0U) << '|'
           << graph.name << '|' << decoded_path_text(graph);
    append("graph");
  }
  for (const auto& vertex : snapshot.vertices) {
    record << vertex.id.value << '|' << vertex.graph.value << '|'
           << static_cast<unsigned>(enum_value(vertex.kind)) << '|' << vertex.name;
    append("vertex");
  }
  for (const auto& port : snapshot.ports) {
    record << port.id.value << '|' << port.owner.value << '|'
           << static_cast<unsigned>(enum_value(port.owner_kind)) << '|'
           << static_cast<unsigned>(enum_value(port.direction)) << '|' << port.name;
    append("port");
  }
  for (const auto& edge : snapshot.edges) {
    record << edge.id.value << '|' << edge.source.value << '|' << edge.target.value;
    append("edge");
  }
  for (const auto& binding : snapshot.interface_bindings) {
    record << binding.public_port.value << '|' << binding.internal_port.value << '|'
           << static_cast<unsigned>(enum_value(binding.direction));
    append("binding");
  }
  for (const auto& topic : snapshot.external_topics) {
    record << topic.id.value << '|' << topic.vertex.value << '|' << topic.graph_port.value << '|'
           << topic.imported << '|' << topic.declaration_ordinal << '|' << topic.uri;
    append("topic");
  }
  for (const auto& source : snapshot.readiness_sources) {
    record << source.owner.value << '|' << source.declaration_ordinal << '|'
           << static_cast<unsigned>(enum_value(source.kind)) << '|' << source.name;
    append("readiness");
  }
  for (const auto& region : snapshot.regions) {
    record << region.id.value << '|' << region.graph.value << '|' << region.name;
    for (const VertexId member : region.members)
      record << "|v" << member.value;
    for (const GraphId member : region.graph_members)
      record << "|g" << member.value;
    append("region");
  }
  for (const auto& partition : snapshot.partitions) {
    record << partition.id.value << '|' << partition.graph.value << '|' << partition.name;
    for (const VertexId member : partition.members)
      record << "|v" << member.value;
    append("partition");
  }
  for (const auto& clock : snapshot.clocks) {
    record << clock.graph.value << '|' << clock_key_text(clock.key) << '|' << clock.shared << '|'
           << clock.is_default;
    append("clock");
  }
  for (const auto& conversion : snapshot.clock_conversions) {
    record << conversion.graph.value << '|' << clock_key_text(conversion.from) << '|'
           << clock_key_text(conversion.to);
    append("conversion");
  }
  for (const auto& dependency : snapshot.lifecycle_dependencies) {
    record << dependency.before.value << '|' << dependency.after.value << '|'
           << static_cast<unsigned>(enum_value(dependency.stage));
    append("lifecycle-dependency");
  }
  for (const auto& permission : snapshot.lifecycle_restart_permissions) {
    record << static_cast<unsigned>(enum_value(permission.origin)) << '|'
           << (permission.requester ? permission.requester->value : 0U) << '|'
           << permission.target.value;
    append("lifecycle-permission");
  }
  return result;
}

// Same serialization as canonical_record_order but over a live GraphView with graph-ID and
// path-prefix normalization; lets an embedded child's records compare directly against the
// standalone child's authored order.
// Normalize only the outer Graph scope so direct and parent-projected authored records compare.
std::vector<std::string> authored_projection_order(const GraphView& view, std::size_t graph_skip,
                                                   std::uint64_t graph_id_offset,
                                                   std::size_t path_prefix_segments) {
  std::vector<std::string> result;
  std::ostringstream record;
  auto append = [&](std::string_view kind) {
    result.push_back(std::string(kind) + '|' + record.str());
    record.str({});
    record.clear();
  };
  const auto graph_id = [graph_id_offset](GraphId id) { return id.value - graph_id_offset; };
  for (std::size_t index = graph_skip; index < view.graphs().size(); ++index) {
    const auto& graph = view.graphs()[index];
    const bool root = index == graph_skip;
    record << graph_id(graph.id) << '|' << (root ? 0U : graph_id(graph.parent.value())) << '|'
           << (root ? "<root>" : graph.name) << '|';
    if (!graph.path || graph.path->segments().size() == path_prefix_segments) {
      record << "_root";
    } else {
      for (std::size_t segment = path_prefix_segments; segment < graph.path->segments().size();
           ++segment) {
        record << '/' << graph.path->segments()[segment];
      }
    }
    append("graph");
  }
  for (const auto& vertex : view.vertices()) {
    record << vertex.id.value << '|' << graph_id(vertex.graph) << '|'
           << static_cast<unsigned>(enum_value(vertex.kind)) << '|' << vertex.name;
    append("vertex");
  }
  for (const auto& port : view.ports()) {
    record << port.id.value << '|' << port.owner.value << '|'
           << static_cast<unsigned>(enum_value(port.owner_kind)) << '|'
           << static_cast<unsigned>(enum_value(port.direction)) << '|' << port.name;
    append("port");
  }
  for (const auto& edge : view.edges()) {
    record << edge.id.value << '|' << edge.source.value << '|' << edge.target.value;
    append("edge");
  }
  for (const auto& binding : view.interface_bindings()) {
    record << binding.public_port.value << '|' << binding.internal_port.value << '|'
           << static_cast<unsigned>(enum_value(binding.direction));
    append("binding");
  }
  for (const auto& topic : view.external_topics()) {
    record << topic.id.value << '|' << topic.vertex.value << '|' << topic.graph_port.value << '|'
           << topic.imported << '|' << topic.declaration_ordinal << '|' << topic.uri;
    append("topic");
  }
  for (const auto& source : view.readiness_sources()) {
    record << source.owner.value << '|' << source.declaration_ordinal << '|'
           << static_cast<unsigned>(enum_value(source.kind)) << '|' << source.name;
    append("readiness");
  }
  for (const auto& region : view.regions()) {
    record << region.id.value << '|' << graph_id(region.graph) << '|' << region.name;
    for (const VertexId member : region.members)
      record << "|v" << member.value;
    for (const GraphId member : region.graph_members)
      record << "|g" << graph_id(member);
    append("region");
  }
  for (const auto& partition : view.partitions()) {
    record << partition.id.value << '|' << graph_id(partition.graph) << '|' << partition.name;
    for (const VertexId member : partition.members)
      record << "|v" << member.value;
    append("partition");
  }
  for (const auto& clock : view.clocks()) {
    record << graph_id(clock.graph) << '|' << clock_key_text(clock.key) << '|' << clock.shared
           << '|' << clock.is_default;
    append("clock");
  }
  for (const auto& conversion : view.clock_conversions()) {
    record << graph_id(conversion.graph) << '|' << clock_key_text(conversion.from) << '|'
           << clock_key_text(conversion.to);
    append("conversion");
  }
  for (const auto& dependency : view.lifecycle_dependencies()) {
    record << dependency.before.value << '|' << dependency.after.value << '|'
           << static_cast<unsigned>(enum_value(dependency.stage));
    append("lifecycle-dependency");
  }
  for (const auto& permission : view.lifecycle_restart_permissions()) {
    record << static_cast<unsigned>(enum_value(permission.origin)) << '|'
           << (permission.requester ? permission.requester->value : 0U) << '|'
           << permission.target.value;
    append("lifecycle-permission");
  }
  return result;
}

// Builds a three-vertex, two-embedded-subgraph parent in two authoring orders (forward or
// reversed) for use in canonical identity tests; same structure, different insertion order.
void author_permuted_parent(Graph& graph, bool reverse) {
  std::optional<OperatorHandle<EmbeddingOp>> alpha;
  std::optional<OperatorHandle<EmbeddingOp>> beta;
  std::optional<OperatorHandle<EmbeddingOp>> gamma;
  if (reverse) {
    gamma = graph.op<EmbeddingOp>("gamma");
    beta = graph.op<EmbeddingOp>("beta");
    alpha = graph.op<EmbeddingOp>("alpha");
    (void)graph.embed<ReusableGraph>("right", "right");
    (void)graph.embed<ReusableGraph>("left", "left");
    (void)graph.add_flow(alpha->operator->()->output, gamma->operator->()->input);
    (void)graph.add_flow(alpha->operator->()->output, beta->operator->()->input);
  } else {
    alpha = graph.op<EmbeddingOp>("alpha");
    beta = graph.op<EmbeddingOp>("beta");
    gamma = graph.op<EmbeddingOp>("gamma");
    (void)graph.embed<ReusableGraph>("left", "left");
    (void)graph.embed<ReusableGraph>("right", "right");
    (void)graph.add_flow(alpha->operator->()->output, beta->operator->()->input);
    (void)graph.add_flow(alpha->operator->()->output, gamma->operator->()->input);
  }
}

static_assert(!std::default_initializable<GraphInstancePath>);
static_assert(!std::copy_constructible<Graph>);
static_assert(!std::move_constructible<Graph>);
static_assert(std::same_as<decltype(std::declval<const GraphHandle<ReusableGraph>&>().operator->()),
                           const ReusableGraph*>);
static_assert(std::same_as<decltype(std::declval<const GraphHandle<ReusableGraph>&>().path()),
                           std::optional<GraphInstancePath>>);
static_assert(!detail::ShallowCapture<GraphHandle<ReusableGraph>>);

template <typename G, typename... Args>
concept EmbeddableWith = requires(Graph& graph, Args&&... args) {
  graph.template embed<G>("child", std::forward<Args>(args)...);
};

static_assert(EmbeddableWith<MoveOnlyGraph, std::unique_ptr<int>>);
static_assert(!EmbeddableWith<AuthoringHandleGraph, OperatorHandle<EmbeddingOp>>);

// REQ: GRAPH-021
// Topology:
//   parent:  source --[out→in]--> child(fusion: first→second) --[out→in]--> target
//            child.first/second are internal; only child.input / child.output are exposed.
//   cyclic:  loop --[out→in]--> body --[out→in]--> loop   (rejected: GRAPH_DIRECT_CYCLE)
TEST(SubgraphEmbedding, ParentWiresOnlyCopiedInterfacesAndRejectsLeakedInternalPorts) {
  Graph parent{"parent"};
  auto source = parent.op<EmbeddingOp>("source");
  auto target = parent.op<EmbeddingOp>("target");
  auto child = parent.embed<ReusableGraph>("fusion", "owned");

  // Key setup: embed a ReusableGraph, wire its exported ports, then attempt to wire its
  // internal ports (first/second) — which must be invisible to the parent.
  ASSERT_TRUE(child);
  const auto child_path = child.path();
  ASSERT_TRUE(child_path);
  EXPECT_EQ(decoded_path(child_path), std::vector<std::string>{"fusion"});
  (void)parent.add_flow(source->output, child->input);
  (void)parent.add_flow(child->output, target->input);

  const Error internal_target =
      capture_graph_error([&] { (void)parent.add_flow(source->output, child->first()->input); });
  EXPECT_TRUE(internal_target.message.starts_with("GRAPH_PORT_NOT_FOUND"));
  EXPECT_NE(internal_target.message.find("source.output"), std::string::npos);
  const Error internal_source =
      capture_graph_error([&] { (void)parent.add_flow(child->second()->output, target->input); });
  EXPECT_TRUE(internal_source.message.starts_with("GRAPH_PORT_NOT_FOUND"));
  EXPECT_NE(internal_source.message.find("target.input"), std::string::npos);
  const Error internal_configuration =
      capture_graph_error([&] { (void)parent.configure(child->first()->input); });
  EXPECT_EQ(internal_configuration.message, "GRAPH_PORT_NOT_FOUND");

  const GraphView view = parent.view();
  EXPECT_EQ(view.edges().size(), 3U);
  EXPECT_EQ(view.interface_bindings().size(), 2U);
  ASSERT_EQ(view.graphs().size(), 2U);
  EXPECT_FALSE(view.graphs()[0].path.has_value());
  EXPECT_EQ(decoded_path(view.graphs()[1]), std::vector<std::string>{"fusion"});

  Graph cyclic{"cyclic"};
  auto loop = cyclic.op<EmbeddingOp>("loop");
  auto body = cyclic.embed<ReusableGraph>("body");
  (void)cyclic.add_flow(loop->output, body->input);
  const Error cycle =
      capture_graph_error([&] { (void)cyclic.add_flow(body->output, loop->input); });
  EXPECT_EQ(cycle.message, "GRAPH_DIRECT_CYCLE");
}

// REQ: GRAPH-021
// REQ: GRAPH-036
TEST(SubgraphEmbedding, OneShotConstructionAcceptsMoveOnlyOwnership) {
  Graph parent{"parent"};
  auto child = parent.embed<MoveOnlyGraph>("owned", std::make_unique<int>(42));

  ASSERT_TRUE(child);
  EXPECT_EQ(child->value(), 42);
}

// REQ: GRAPH-021
// Topology:
//   parent:  source --[out → child.exported_input]
//            child.unexported_input and child.unexported_output: C++ public but not exported.
TEST(SubgraphEmbedding, PublicPortsRemainPrivateUntilExplicitlyExported) {
  Graph parent{"parent"};
  auto source = parent.op<EmbeddingOp>("source");
  auto target = parent.op<EmbeddingOp>("target");
  auto child = parent.embed<PartiallyExportedGraph>("child");

  // Public C++ visibility permits typed access but does not grant parent wiring authority.
  (void)parent.add_flow(source->output, child->exported_input);
  EXPECT_TRUE(capture_graph_error([&] {
                (void)parent.add_flow(source->output, child->unexported_input);
              }).message.starts_with("GRAPH_PORT_NOT_FOUND"));
  EXPECT_TRUE(capture_graph_error([&] {
                (void)parent.add_flow(child->unexported_output, target->input);
              }).message.starts_with("GRAPH_PORT_NOT_FOUND"));
  EXPECT_EQ(capture_graph_error([&] { (void)parent.configure(child->unexported_input); }).message,
            "GRAPH_PORT_NOT_FOUND");

  const GraphView view = parent.view();
  EXPECT_EQ(view.interface_bindings().size(), 1U);
  EXPECT_EQ(view.edges().size(), 1U);
}

// REQ: GRAPH-022
// Topology:
//   parent:
//     left(NestedGraph) → left/leaf(ReusableGraph): first, second
//     right(NestedGraph) → right/leaf(ReusableGraph): first, second
//   Five graphs total; "first" operators live in two distinct graph scopes.
TEST(SubgraphEmbedding, RepeatedAndNestedInstancesRetainDistinctCanonicalPaths) {
  Graph parent{"parent"};
  auto left = parent.embed<NestedGraph>("left");
  auto right = parent.embed<NestedGraph>("right");
  ASSERT_TRUE(left);
  ASSERT_TRUE(right);
  EXPECT_NE(left.operator->(), right.operator->());
  const auto left_path = left.path();
  const auto nested_leaf_path = left->leaf().path();
  ASSERT_TRUE(left_path);
  ASSERT_TRUE(nested_leaf_path);
  EXPECT_EQ(decoded_path(left_path), std::vector<std::string>{"left"});
  EXPECT_EQ(decoded_path(nested_leaf_path), std::vector<std::string>{"leaf"});

  const GraphView view = parent.view();
  ASSERT_EQ(view.graphs().size(), 5U);
  std::vector<std::vector<std::string>> paths;
  paths.reserve(view.graphs().size());
  for (const auto& graph : view.graphs())
    paths.push_back(decoded_path(graph));
  EXPECT_NE(std::find(paths.begin(), paths.end(), std::vector<std::string>{}), paths.end());
  EXPECT_NE(std::find(paths.begin(), paths.end(), std::vector<std::string>{"left"}), paths.end());
  EXPECT_NE(std::find(paths.begin(), paths.end(), std::vector<std::string>{"left", "leaf"}),
            paths.end());
  EXPECT_NE(std::find(paths.begin(), paths.end(), std::vector<std::string>{"right"}), paths.end());
  EXPECT_NE(std::find(paths.begin(), paths.end(), std::vector<std::string>{"right", "leaf"}),
            paths.end());

  std::unordered_set<std::uint64_t> first_graphs;
  for (const auto& vertex : view.vertices()) {
    if (vertex.name == "first")
      first_graphs.emplace(vertex.graph.value);
  }
  EXPECT_EQ(first_graphs.size(), 2U);
}

// REQ: GRAPH-023
// Topology:
//   parent:
//     left(NestedGraph) → left/leaf(ReusableGraph): first, second
//     right(NestedGraph) → right/leaf(ReusableGraph): first, second
//   Parent adds: left/leaf/first ──kStart──> right/leaf/second (cross-instance lifecycle).
//   A foreign graph must not author the same dependency.
TEST(SubgraphEmbedding, ExactParentMayAuthorizeNestedCrossInstanceLifecycleOnlyExplicitly) {
  Graph parent{"parent"};
  auto left = parent.embed<NestedGraph>("left");
  auto right = parent.embed<NestedGraph>("right");
  const std::size_t dependency_count = parent.view().lifecycle_dependencies().size();
  const std::size_t permission_count = parent.view().lifecycle_restart_permissions().size();

  parent.region("left_deployment").add(left);
  parent.lifecycle()
      .before(left->leaf()->first(), right->leaf()->second(), LifecycleStage::kStart)
      .allow_restart_request(left->leaf()->first(), right->leaf()->second());
  const GraphView authorized = parent.view();
  EXPECT_EQ(authorized.lifecycle_dependencies().size(), dependency_count + 1U);
  EXPECT_EQ(authorized.lifecycle_restart_permissions().size(), permission_count + 1U);

  // Key assertion: resolve flattened vertex IDs by path and verify the exact dependency and
  // permission targets — record counts alone cannot catch a wrong-scope wiring.
  const GraphId left_graph = graph_id_at_path(authorized, {"left"});
  const GraphId left_leaf_graph = graph_id_at_path(authorized, {"left", "leaf"});
  const GraphId right_leaf_graph = graph_id_at_path(authorized, {"right", "leaf"});
  const VertexId left_first = vertex_id_by_name(authorized, left_leaf_graph, "first");
  const VertexId left_second = vertex_id_by_name(authorized, left_leaf_graph, "second");
  const VertexId right_second = vertex_id_by_name(authorized, right_leaf_graph, "second");

  const auto deployment = std::find_if(
      authorized.regions().begin(), authorized.regions().end(), [](const auto& region) {
        return region.name == "left_deployment";
      });
  ASSERT_NE(deployment, authorized.regions().end());
  EXPECT_TRUE(deployment->members.empty());
  EXPECT_EQ(deployment->graph_members, std::vector<GraphId>{left_graph});

  const auto child_region = std::find_if(
      authorized.regions().begin(), authorized.regions().end(), [&](const auto& region) {
        return region.graph == left_leaf_graph && region.name == "child_region";
      });
  ASSERT_NE(child_region, authorized.regions().end());
  EXPECT_EQ(child_region->members, std::vector<VertexId>{left_first});
  ASSERT_EQ(child_region->constraints.size(), 1U);
  EXPECT_EQ(child_region->constraints[0].kind(), RegionConstraintKind::kBoundedMemory);
  EXPECT_EQ(child_region->profile, Profile::kRealTime);

  const auto readiness = std::find_if(
      authorized.readiness_sources().begin(),
      authorized.readiness_sources().end(),
      [&](const auto& source) { return source.owner == left_first && source.name == "readable"; });
  ASSERT_NE(readiness, authorized.readiness_sources().end());
  EXPECT_EQ(readiness->declaration_ordinal, 0U);
  EXPECT_EQ(readiness->kind, ReadinessSourceKind::kReadable);

  const auto child_dependency = std::find_if(authorized.lifecycle_dependencies().begin(),
                                             authorized.lifecycle_dependencies().end(),
                                             [&](const auto& dependency) {
                                               return dependency.before == left_first &&
                                                      dependency.after == left_second &&
                                                      dependency.stage == LifecycleStage::kStart;
                                             });
  EXPECT_NE(child_dependency, authorized.lifecycle_dependencies().end());
  const auto cross_dependency = authorized.lifecycle_dependencies().back();
  EXPECT_EQ(cross_dependency.before, left_first);
  EXPECT_EQ(cross_dependency.after, right_second);
  EXPECT_EQ(cross_dependency.stage, LifecycleStage::kStart);

  const auto child_permission =
      std::find_if(authorized.lifecycle_restart_permissions().begin(),
                   authorized.lifecycle_restart_permissions().end(),
                   [&](const auto& permission) {
                     return permission.origin == LifecycleRestartOrigin::kOperatorRequest &&
                            permission.requester == left_first && permission.target == left_second;
                   });
  EXPECT_NE(child_permission, authorized.lifecycle_restart_permissions().end());
  const auto cross_permission = authorized.lifecycle_restart_permissions().back();
  EXPECT_EQ(cross_permission.origin, LifecycleRestartOrigin::kOperatorRequest);
  EXPECT_EQ(cross_permission.requester, left_first);
  EXPECT_EQ(cross_permission.target, right_second);

  Graph unrelated{"unrelated"};
  const Error dependency = capture_graph_error([&] {
    unrelated.lifecycle().before(
        left->leaf()->first(), right->leaf()->second(), LifecycleStage::kConfigure);
  });
  EXPECT_EQ(dependency.message, "GRAPH_LIFECYCLE_HANDLE_INVALID");
  const Error permission = capture_graph_error([&] {
    unrelated.lifecycle().allow_restart_request(left->leaf()->first(), right->leaf()->second());
  });
  EXPECT_EQ(permission.message, "GRAPH_LIFECYCLE_HANDLE_INVALID");
  EXPECT_TRUE(unrelated.view().lifecycle_dependencies().empty());
  EXPECT_TRUE(unrelated.view().lifecycle_restart_permissions().empty());
}

// REQ: GRAPH-020
// REQ: GRAPH-021
TEST(SubgraphEmbedding, EmbeddingIsAtomicAcrossFailuresAndHandlesExpireWithParent) {
  // Key setup: sweep checkpoints 0..511, injecting a bad_alloc at each mutation site;
  // each failure must leave the parent view identical to its pre-embed snapshot.
  constexpr std::size_t kMaximumEmbeddingCheckpoints = 512U;
  std::optional<std::size_t> first_nonfailing_checkpoint;
  for (std::size_t checkpoint = 0; checkpoint < kMaximumEmbeddingCheckpoints; ++checkpoint) {
    Graph parent{"atomic"};
    const GraphView before = parent.view();
    detail::GraphCompilerAccess::fail_mutation_after(parent, checkpoint);
    try {
      auto child = parent.embed<ReusableGraph>("candidate", "fault-sweep");
      EXPECT_TRUE(child);
      first_nonfailing_checkpoint = checkpoint;
      break;
    } catch (const std::bad_alloc&) {
      const GraphView after = parent.view();
      EXPECT_EQ(after.generation(), before.generation()) << "checkpoint " << checkpoint;
      EXPECT_EQ(after.identity(), before.identity()) << "checkpoint " << checkpoint;
      EXPECT_EQ(after.graphs().size(), before.graphs().size()) << "checkpoint " << checkpoint;
      EXPECT_EQ(after.vertices().size(), before.vertices().size()) << "checkpoint " << checkpoint;
      EXPECT_EQ(after.ports().size(), before.ports().size()) << "checkpoint " << checkpoint;
      EXPECT_EQ(after.edges().size(), before.edges().size()) << "checkpoint " << checkpoint;
      auto retry = parent.embed<ReusableGraph>("candidate", "fault-sweep");
      EXPECT_TRUE(retry) << "checkpoint " << checkpoint;
      EXPECT_EQ(decoded_path(parent.view().graphs().back()), std::vector<std::string>{"candidate"});
    }
  }
  ASSERT_TRUE(first_nonfailing_checkpoint.has_value());
  EXPECT_GT(first_nonfailing_checkpoint.value_or(0U), 0U);

  // Key assertion: after the parent is destroyed, the handle evaluates to false and any
  // member access throws; retained GraphView snapshots remain valid read-only.
  std::optional<GraphHandle<ReusableGraph>> retired_handle;
  GraphView retained_view;
  {
    auto parent = std::make_unique<Graph>("lifetime");
    retired_handle = parent->embed<ReusableGraph>("child", "lifetime");
    retained_view = parent->view();
    ASSERT_TRUE(*retired_handle);
    ASSERT_NE(retired_handle->operator->(), nullptr);
    ASSERT_TRUE(retired_handle->path());
    EXPECT_EQ(retained_view.graphs().size(), 2U);
  }
  EXPECT_FALSE(*retired_handle);
  const Error retired_error = capture_graph_error([&] { (void)(*retired_handle)->input; });
  EXPECT_EQ(retired_error.code, ErrorCode::kInvalidArgument);
  EXPECT_EQ(retired_error.message, "GRAPH_HANDLE_INVALID");
  EXPECT_FALSE(retired_handle->path());
  ASSERT_EQ(retained_view.graphs().size(), 2U);
  EXPECT_EQ(decoded_path(retained_view.graphs()[1]), std::vector<std::string>{"child"});
}

// REQ: GRAPH-020
// Topology: each verify_failure lambda embeds a failing child into a parent that already
// has one retained operator ("retained"); after failure the name must still be available
// for a clean retry, and all parent view counts must remain unchanged.
TEST(SubgraphEmbedding, ConstructionAndCompositionFailuresDoNotPublishParentState) {
  const auto verify_failure = [](auto&& attempt, std::string_view expected_error) {
    Graph parent{"failure-parent"};
    (void)parent.op<EmbeddingOp>("retained");
    const GraphView before = parent.view();

    const Error failure =
        capture_graph_error([&] { std::forward<decltype(attempt)>(attempt)(parent); });
    EXPECT_TRUE(failure.message.starts_with(expected_error)) << failure.message;
    EXPECT_NE(failure.message.find("graph instance=candidate"), std::string::npos);
    expect_same_authoring_state(before, parent.view());

    // Failure must not consume the requested name or prevent a clean retry.
    auto retry = parent.embed<AuthoredOrderLeafGraph>("candidate", "retry");
    EXPECT_TRUE(retry);
  };

  verify_failure(
      [](Graph& parent) { (void)parent.embed<ConstructorFailureGraph>("candidate", false); },
      "GRAPH_EMBED_CONSTRUCTION_FAILED");
  verify_failure(
      [](Graph& parent) { (void)parent.embed<ConstructorFailureGraph>("candidate", true); },
      "GRAPH_EMBED_CONSTRUCTION_UNKNOWN_EXCEPTION");
  verify_failure([](Graph& parent) { (void)parent.embed<ComposeFailureGraph>("candidate", false); },
                 "GRAPH_COMPOSE_EXCEPTION");
  verify_failure([](Graph& parent) { (void)parent.embed<ComposeFailureGraph>("candidate", true); },
                 "GRAPH_COMPOSE_UNKNOWN_EXCEPTION");
}

// REQ: GRAPH-020
// Topology:
//   parent: import_topic with endpoint_name="shared_endpoint"
//   child(ConflictingEndpointGraph): also uses endpoint_name="shared_endpoint"
//   Conflict is detected at registration (after successful child composition).
TEST(SubgraphEmbedding, RegistrationValidationFailureDoesNotPublishParentState) {
  Graph parent{"validation-parent"};
  TopicOptions options;
  options.endpoint_name = "shared_endpoint";
  (void)parent.import_topic<std::uint32_t>("holomq://parent-topic", options);
  const GraphView before = parent.view();

  // The child is valid by itself; only its publication into this parent conflicts.
  const Error failure =
      capture_graph_error([&] { (void)parent.embed<ConflictingEndpointGraph>("candidate"); });
  EXPECT_EQ(failure.message, "GRAPH_ENDPOINT_NAME_DUPLICATE");
  expect_same_authoring_state(before, parent.view());

  auto retry = parent.embed<AuthoredOrderLeafGraph>("candidate", "retry");
  EXPECT_TRUE(retry);
}

// REQ: GRAPH-020
// REQ: GRAPH-022
// Topology (same logical structure, two authoring orders):
//   forward:  alpha→beta, alpha→gamma; embed left(ReusableGraph), right(ReusableGraph)
//   reverse:  gamma←alpha→beta; embed right(ReusableGraph), left(ReusableGraph)
//   Canonical snapshot identity hash must match regardless of insertion order.
TEST(PartitioningPassTest, EmbeddedGraphIdentityIsCanonical) {
  Graph forward{"canonical"};
  author_permuted_parent(forward, false);
  Graph reverse{"canonical"};
  author_permuted_parent(reverse, true);

  const auto forward_snapshot = detail::GraphCompilerAccess::canonical_snapshot(forward);
  const auto reverse_snapshot = detail::GraphCompilerAccess::canonical_snapshot(reverse);
  ASSERT_TRUE(forward_snapshot);
  ASSERT_TRUE(reverse_snapshot);
  EXPECT_EQ(forward_snapshot->generation, reverse_snapshot->generation);
  EXPECT_EQ(forward_snapshot->identity.version, reverse_snapshot->identity.version);
  EXPECT_EQ(forward_snapshot->identity.hash, reverse_snapshot->identity.hash);
  EXPECT_EQ(canonical_record_order(*forward_snapshot), canonical_record_order(*reverse_snapshot));
}

// REQ: GRAPH-022
// Topology (same linear chain, two authoring orders):
//   forward:  alpha authored first, then beta; edge alpha→beta
//   reverse:  beta authored first, then alpha; same edge alpha→beta
//   Identity hash must match; canonical_record_order must also match.
TEST(PartitioningPassTest, RootGraphIdentityIsCanonical) {
  Graph forward{"canonical"};
  auto forward_alpha = forward.op<EmbeddingOp>("alpha");
  auto forward_beta = forward.op<EmbeddingOp>("beta");
  (void)forward.add_flow(forward_alpha->output, forward_beta->input);

  Graph reverse{"canonical"};
  auto reverse_beta = reverse.op<EmbeddingOp>("beta");
  auto reverse_alpha = reverse.op<EmbeddingOp>("alpha");
  (void)reverse.add_flow(reverse_alpha->output, reverse_beta->input);

  const GraphView forward_view = forward.view();
  const GraphView reverse_view = reverse.view();
  EXPECT_EQ(forward_view.identity().hash, reverse_view.identity().hash);

  const auto forward_snapshot = detail::GraphCompilerAccess::canonical_snapshot(forward);
  const auto reverse_snapshot = detail::GraphCompilerAccess::canonical_snapshot(reverse);
  ASSERT_TRUE(forward_snapshot);
  ASSERT_TRUE(reverse_snapshot);
  EXPECT_EQ(forward_snapshot->identity.hash, reverse_snapshot->identity.hash);
  EXPECT_EQ(canonical_record_order(*forward_snapshot), canonical_record_order(*reverse_snapshot));
}

// REQ: GRAPH-022
// Topology:
//   standalone(NonLexicalAuthoredGraph):  z → a  (reverse-alphabetical authoring)
//   parent embeds standalone as "child"; vertex and sub-graph records must arrive in
//   authored order (z before a, z_child before a_child).
TEST(SubgraphEmbedding, EmbeddedRecordsPreserveChildAuthoredOrder) {
  // Every child record must retain its authored position and remapped identity after embedding.
  NonLexicalAuthoredGraph standalone;
  const GraphView child_view = standalone.view();
  Graph parent{"parent"};
  (void)parent.embed<NonLexicalAuthoredGraph>("child");
  const GraphView parent_view = parent.view();

  // Key assertion: authored_projection_order of the embedded child matches the standalone
  // child's order verbatim after normalising graph IDs and path prefix.
  EXPECT_EQ(authored_projection_order(parent_view, 1U, 1U, 1U),
            authored_projection_order(child_view, 0U, 0U, 0U));
  ASSERT_GE(parent_view.vertices().size(), 2U);
  EXPECT_EQ(parent_view.vertices()[0].name, "z");
  EXPECT_EQ(parent_view.vertices()[1].name, "a");
  ASSERT_EQ(parent_view.graphs().size(), 4U);
  EXPECT_EQ(parent_view.graphs()[2].name, "z_child");
  EXPECT_EQ(parent_view.graphs()[3].name, "a_child");
}

// REQ: GRAPH-021
// REQ: GRAPH-023
// Topology:
//   parent:
//     source (EmbeddingOp)
//     left(OpaqueNestedGraph) → left/leaf(ReusableGraph): first, second  [no export]
//     right(OpaqueNestedGraph) → right/leaf(ReusableGraph): first, second [no export]
//   Parent may author cross-instance lifecycle; a foreign graph must be rejected.
TEST(SubgraphEmbedding, NestedInterfaceAndLifecycleAuthorityRemainScoped) {
  Graph parent{"parent"};
  auto source = parent.op<EmbeddingOp>("source");
  auto left = parent.embed<OpaqueNestedGraph>("left");
  auto right = parent.embed<OpaqueNestedGraph>("right");

  const Error leaked_interface =
      capture_graph_error([&] { (void)parent.add_flow(source->output, left->leaf()->input); });
  EXPECT_TRUE(leaked_interface.message.starts_with("GRAPH_PORT_NOT_FOUND"));

  const std::size_t dependency_count = parent.view().lifecycle_dependencies().size();
  const std::size_t permission_count = parent.view().lifecycle_restart_permissions().size();
  parent.lifecycle()
      .before(left->leaf()->first(), right->leaf()->second(), LifecycleStage::kStart)
      .allow_restart_request(left->leaf()->first(), right->leaf()->second());
  EXPECT_EQ(parent.view().lifecycle_dependencies().size(), dependency_count + 1U);
  EXPECT_EQ(parent.view().lifecycle_restart_permissions().size(), permission_count + 1U);

  Graph unrelated{"unrelated"};
  const Error foreign_dependency = capture_graph_error([&] {
    unrelated.lifecycle().before(
        left->leaf()->first(), right->leaf()->second(), LifecycleStage::kStart);
  });
  EXPECT_EQ(foreign_dependency.message, "GRAPH_LIFECYCLE_HANDLE_INVALID");
  const Error foreign_permission = capture_graph_error([&] {
    unrelated.lifecycle().allow_restart_request(left->leaf()->first(), right->leaf()->second());
  });
  EXPECT_EQ(foreign_permission.message, "GRAPH_LIFECYCLE_HANDLE_INVALID");
}

// REQ: GRAPH-021
// REQ: GRAPH-022
// Topology:
//   parent:
//     left(ReusableGraph): first("left-first"), second  source_clock(default), offset=3 conv
//     right(ReusableGraph): first("right-first"), second source_clock(default), offset=3 conv
//     root: source_clock, target_clock(default), offset=9 conversion
TEST(SubgraphEmbedding, RepeatedEmbedsRetainIndependentFactories) {
  Graph parent{"factories"};
  auto left = parent.embed<ReusableGraph>("left", "left");
  auto right = parent.embed<ReusableGraph>("right", "right");
  ASSERT_TRUE(left);
  ASSERT_TRUE(right);
  EXPECT_NE(left.operator->(), right.operator->());
  EXPECT_EQ(left->first()->label_, "left-first");
  EXPECT_EQ(right->first()->label_, "right-first");

  const ClockKey root_source = parent.add_clock<ManualClock>("source_clock", std::int64_t{11});
  const ClockKey root_target = parent.add_clock<ManualClock>("target_clock", std::int64_t{12});
  parent.set_default_clock(root_target);
  parent.add_clock_conversion(root_target, root_source, AffineClockConversion{.offset_ns = 9});

  // Key setup: add root-level clocks after embedding; each child instance must retain its
  // own source_clock as default and its own offset=3 conversion independently.
  const auto operator_count = detail::GraphCompilerAccess::instantiate_operator_factories(parent);
  ASSERT_TRUE(operator_count);
  EXPECT_EQ(operator_count.value(), 4U);
  const auto clock_count = detail::GraphCompilerAccess::instantiate_clock_factories(parent);
  ASSERT_TRUE(clock_count);
  EXPECT_EQ(clock_count.value(), 6U);

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(parent);
  ASSERT_TRUE(snapshot);
  ASSERT_EQ(snapshot->graphs.size(), 3U);
  ASSERT_EQ(snapshot->clocks.size(), 6U);
  ASSERT_EQ(snapshot->clock_conversions.size(), 3U);
  std::map<std::uint64_t, std::string> graph_paths;
  for (const auto& graph : snapshot->graphs)
    graph_paths.emplace(graph.id.value, decoded_path_text(graph));
  std::map<std::string, std::size_t> clocks_per_graph;
  std::map<std::string, std::size_t> defaults_per_graph;
  for (const auto& clock : snapshot->clocks) {
    const std::string& path = graph_paths.at(clock.graph.value);
    ++clocks_per_graph[path];
    if (clock.is_default) {
      ++defaults_per_graph[path];
      EXPECT_EQ(clock.key.name(), path == "_root" ? "target_clock" : "source_clock");
    }
  }
  EXPECT_EQ(clocks_per_graph["/left"], 2U);
  EXPECT_EQ(clocks_per_graph["/right"], 2U);
  EXPECT_EQ(clocks_per_graph["_root"], 2U);
  EXPECT_EQ(defaults_per_graph["/left"], 1U);
  EXPECT_EQ(defaults_per_graph["/right"], 1U);
  EXPECT_EQ(defaults_per_graph["_root"], 1U);
  for (const auto& conversion : snapshot->clock_conversions) {
    const std::string& path = graph_paths.at(conversion.graph.value);
    if (path == "_root") {
      EXPECT_EQ(conversion.from.name(), "target_clock");
      EXPECT_EQ(conversion.to.name(), "source_clock");
      EXPECT_EQ(conversion.conversion.offset_ns, 9);
    } else {
      EXPECT_TRUE(path == "/left" || path == "/right");
      EXPECT_EQ(conversion.from.name(), "source_clock");
      EXPECT_EQ(conversion.to.name(), "target_clock");
      EXPECT_EQ(conversion.conversion.offset_ns, 3);
    }
  }
}

// REQ: OPERATOR-051
TEST(SubgraphEmbedding, CaptureEligibilityPropagatesThroughCanonicalEmbeddedFactories) {
  Graph parent{"capture-eligibility-embedding"};
  (void)parent.embed<CaptureEligibleGraph>("child");

  const auto snapshot = detail::GraphCompilerAccess::canonical_snapshot(parent);
  ASSERT_TRUE(snapshot);
  ASSERT_EQ(snapshot->vertices.size(), 1U);
  EXPECT_TRUE(snapshot->vertices.front().cuda_graph_capture_eligible);

  const auto factories = detail::GraphCompilerAccess::executable_factories(parent, *snapshot);
  ASSERT_EQ(factories.operators.size(), 1U);
  EXPECT_TRUE(factories.operators.front().cuda_graph_capture_eligible);
  EXPECT_FALSE(factories.operators.front().root_operator_index.has_value());
}

// REQ: GRAPH-026
// Reusable Graph clock dependencies remain explicit, instance-qualified, and forwardable.
TEST(SubgraphEmbedding, ClockImportsBindAcrossExactReusableGraphBoundaries) {
  Graph parent{"clock_parent"};
  const ClockKey root_clock = parent.add_clock<ManualClock>("root_timeline", std::int64_t{7});
  parent.set_default_clock(root_clock);
  auto left = parent.embed<ImportedClockGraph>("left");
  auto right = parent.embed<ImportedClockGraph>("right");

  const auto expect_atomic_failure = [&](auto&& operation, std::string_view token) {
    const GraphView before = parent.view();
    EXPECT_EQ(capture_graph_error(std::forward<decltype(operation)>(operation)).message, token);
    expect_same_authoring_state(before, parent.view());
  };
  expect_atomic_failure([&] { parent.bind_clock(left, root_clock, root_clock); },
                        "GRAPH_CLOCK_IMPORT_NOT_FOUND");
  expect_atomic_failure(
      [&] { parent.bind_clock(left, left->clock_import(), ClockKey::named("missing")); },
      "GRAPH_CLOCK_NOT_FOUND");
  parent.bind_clock(left, left->clock_import(), root_clock);
  parent.bind_clock(right, right->clock_import(), root_clock);

  const GraphView view = parent.view();
  ASSERT_FALSE(view.graphs().empty());
  const GraphId parent_graph_id = view.graphs().front().id;
  ASSERT_EQ(view.clock_imports().size(), 2U);
  ASSERT_EQ(view.clock_bindings().size(), 2U);
  EXPECT_TRUE(
      std::all_of(view.clock_imports().begin(), view.clock_imports().end(), [](const auto& row) {
        return row.key == ClockKey::named("timeline") && row.is_default;
      }));
  EXPECT_TRUE(
      std::all_of(view.clock_bindings().begin(), view.clock_bindings().end(), [&](const auto& row) {
        return row.parent_graph == parent_graph_id && row.parent_clock == root_clock &&
               row.child_clock == ClockKey::named("timeline");
      }));

  expect_atomic_failure([&] { parent.bind_clock(left, left->clock_import(), root_clock); },
                        "GRAPH_CLOCK_BINDING_DUPLICATE");
  Graph unrelated{"unrelated"};
  const ClockKey unrelated_clock =
      unrelated.add_clock<ManualClock>("unrelated_clock", std::int64_t{0});
  const GraphView unrelated_before = unrelated.view();
  EXPECT_EQ(capture_graph_error([&] {
              unrelated.bind_clock(left, left->clock_import(), unrelated_clock);
            }).message,
            "GRAPH_CLOCK_BINDING_HANDLE_INVALID");
  expect_same_authoring_state(unrelated_before, unrelated.view());

  const auto stale = [] {
    Graph ephemeral{"ephemeral"};
    return ephemeral.embed<ImportedClockGraph>("child");
  }();
  const GraphView stale_before = unrelated.view();
  EXPECT_EQ(capture_graph_error([&] {
              unrelated.bind_clock(stale, ClockKey::named("timeline"), unrelated_clock);
            }).message,
            "GRAPH_CLOCK_BINDING_HANDLE_INVALID");
  expect_same_authoring_state(stale_before, unrelated.view());
}

// REQ: GRAPH-026
TEST(SubgraphEmbedding, ClockImportNamespaceCollisionsRejectAtomically) {
  Graph duplicate_import{"duplicate-import"};
  (void)duplicate_import.import_clock("timeline");
  const GraphView duplicate_before = duplicate_import.view();
  EXPECT_EQ(capture_graph_error([&] { (void)duplicate_import.import_clock("timeline"); }).message,
            "GRAPH_CLOCK_NAME_DUPLICATE");
  expect_same_authoring_state(duplicate_before, duplicate_import.view());

  Graph concrete_first{"concrete-first"};
  (void)concrete_first.add_clock<ManualClock>("timeline", std::int64_t{0});
  const GraphView concrete_before = concrete_first.view();
  EXPECT_EQ(capture_graph_error([&] { (void)concrete_first.import_clock("timeline"); }).message,
            "GRAPH_CLOCK_NAME_DUPLICATE");
  expect_same_authoring_state(concrete_before, concrete_first.view());

  Graph import_first{"import-first"};
  (void)import_first.import_clock("timeline");
  const GraphView import_before = import_first.view();
  EXPECT_EQ(capture_graph_error([&] {
              (void)import_first.add_clock<ManualClock>("timeline", std::int64_t{0});
            }).message,
            "GRAPH_CLOCK_NAME_DUPLICATE");
  expect_same_authoring_state(import_before, import_first.view());
}

// REQ: GRAPH-026
TEST(SubgraphEmbedding, ClockImportsForwardOnlyThroughExplicitNestedBindings) {
  Graph root{"clock_root"};
  const ClockKey root_clock = root.add_clock<ManualClock>("root_timeline", std::int64_t{3});
  auto wrapper = root.embed<ForwardedClockGraph>("wrapper");
  root.bind_clock(wrapper, wrapper->clock_import(), root_clock);

  const GraphView view = root.view();
  ASSERT_EQ(view.clock_imports().size(), 2U);
  ASSERT_EQ(view.clock_bindings().size(), 2U);
  const auto forwarded =
      std::find_if(view.clock_bindings().begin(), view.clock_bindings().end(), [](const auto& row) {
        return row.parent_clock == ClockKey::named("outer_timeline");
      });
  ASSERT_NE(forwarded, view.clock_bindings().end());
  EXPECT_NE(forwarded->parent_graph, GraphId{1U});
}

// REQ: GRAPH-020
// REQ: GRAPH-021
TEST(SubgraphEmbedding, RejectsInvalidDuplicateAndRecursiveEmbedding) {
  // Key setup: embed a child, then attempt all invalid embed cases: duplicate name,
  // reserved path segments ("_root", ".", ".."), direct recursion, indirect recursion.
  Graph parent{"invalid"};
  auto original = parent.embed<ReusableGraph>("child");
  ASSERT_TRUE(original);
  EXPECT_EQ(capture_graph_error([&] { (void)parent.embed<ReusableGraph>("child"); }).message,
            "GRAPH_EMBED_INSTANCE_DUPLICATE");
  EXPECT_EQ(capture_graph_error([&] { (void)parent.embed<ReusableGraph>("_root"); }).message,
            "GRAPH_EMBED_INSTANCE_PATH_INVALID");
  EXPECT_EQ(capture_graph_error([&] { (void)parent.embed<ReusableGraph>("."); }).message,
            "GRAPH_EMBED_INSTANCE_PATH_INVALID");
  EXPECT_EQ(capture_graph_error([&] { (void)parent.embed<ReusableGraph>(".."); }).message,
            "GRAPH_EMBED_INSTANCE_PATH_INVALID");

  Graph decoded_segments{"segments"};
  (void)decoded_segments.embed<ReusableGraph>("camera/left");
  (void)decoded_segments.embed<ReusableGraph>("left%rgb");
  (void)decoded_segments.embed<ReusableGraph>("\xE8\x9E\x8D\xE5\x90\x88");
  const GraphView decoded_view = decoded_segments.view();
  std::vector<std::vector<std::string>> decoded;
  for (const auto& graph : decoded_view.graphs())
    decoded.push_back(decoded_path(graph));
  EXPECT_NE(std::find(decoded.begin(), decoded.end(), std::vector<std::string>{"camera/left"}),
            decoded.end());
  EXPECT_NE(std::find(decoded.begin(), decoded.end(), std::vector<std::string>{"left%rgb"}),
            decoded.end());
  EXPECT_NE(
      std::find(
          decoded.begin(), decoded.end(), std::vector<std::string>{"\xE8\x9E\x8D\xE5\x90\x88"}),
      decoded.end());

  Graph direct{"direct"};
  EXPECT_EQ(
      capture_graph_error([&] { (void)direct.embed<DirectRecursiveGraph>("recursive"); }).message,
      "GRAPH_EMBED_CYCLE");
  EXPECT_EQ(direct.view().graphs().size(), 1U);
  Graph indirect{"indirect"};
  EXPECT_EQ(capture_graph_error([&] {
              (void)indirect.embed<IndirectRecursiveGraphA>("recursive");
            }).message,
            "GRAPH_EMBED_CYCLE");
  EXPECT_EQ(indirect.view().graphs().size(), 1U);
}

// REQ: GRAPH-020
//   hosted-nesting                    hosted-recursion
//   `-- parent(type A)                `-- parent(type A)
//       `-- child(type B)                 `-- child(type B)
//                                            `-- child(type A) [rejected]
TEST(SubgraphEmbedding, HostedTypeIdentityPreservesRecursionChecks) {
  // Key setup: Embed distinct hosted logical types, then attempt an indirect A -> B -> A cycle.
  static constexpr std::byte kLogicalTypeA{};
  static constexpr std::byte kLogicalTypeB{};
  const HostedGraphTypeIdentity type_a{&kLogicalTypeA};
  const HostedGraphTypeIdentity type_b{&kLogicalTypeB};

  Graph nested{"hosted-nesting"};
  (void)nested.embed<LanguageHostedGraph>(
      "parent", type_a, std::vector<HostedGraphTypeIdentity>{type_b});
  EXPECT_EQ(nested.view().graphs().size(), 3U);

  Graph recursive{"hosted-recursion"};
  const Error recursive_error = capture_graph_error([&] {
    (void)recursive.embed<LanguageHostedGraph>(
        "parent", type_a, std::vector<HostedGraphTypeIdentity>{type_b, type_a});
  });

  // Key assertion: Cross-type nesting succeeds while indirect logical recursion fails closed.
  EXPECT_EQ(recursive_error.message, "GRAPH_EMBED_CYCLE");
  EXPECT_EQ(recursive.view().graphs().size(), 1U);
}

}  // namespace
}  // namespace holoscan
