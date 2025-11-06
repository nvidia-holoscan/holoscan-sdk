# Subgraph Examples

This directory contains examples demonstrating Holoscan's subgraph functionality for creating reusable, composable operator patterns. Subgraphs enable modular application design by encapsulating related operators and exposing clean interfaces for external connections.

Subgraphs provide a powerful way to:
- **Create reusable components** that can be instantiated multiple times within an application
- **Encapsulate complex logic** behind clean, well-defined interfaces
- **Enable modular design** patterns that improve code organization and maintainability
- **Support flexible connection patterns** between subgraphs and operators

## Examples

### ping_multi_receiver/

This example demonstrates fundamental subgraph concepts including:
- Reusable subgraph instantiation with qualified naming
- Interface port mapping between subgraphs and internal operators
- Multi-receiver patterns for operators that accept input from multiple sources
- Mixed connection patterns (subgraph-to-subgraph and Operator-to-subgraph)

The example shows how to build modular applications where the same subgraph logic can be reused multiple times with different configurations, making it ideal for scenarios like multi-stage processing pipelines that need to be applied across multiple data streams.

## Key Concepts

**Subgraph Reusability**: The same Subgraph class can be instantiated multiple times with different instance names, each creating its own set of qualified operators.

**Interface Ports**: Subgraphs expose external ports that map to internal operator ports, providing clean APIs while hiding internal complexity.

**Qualified Naming**: The framework automatically qualifies operator names within subgraphs to prevent naming conflicts (e.g., "tx1_transmitter", "tx2_transmitter").

**Flexible Connections**: Subgraphs can be connected to other subgraphs or operators using the same `add_flow` API, enabling flexible application architectures.

**Nested Support**: Subgraphs can contain other subgraphs for hierarchical decomposition of complex applications.

## Getting Started

For detailed implementation examples and run instructions, see the individual example directories. Each example includes both C++ and Python implementations demonstrating the same concepts using language-appropriate APIs.

The subgraphs used in these examples are intentionally simple to focus on the API concepts, but the true utility would be realized in real-world applications involving larger graphs. An example might be a multi-stage camera pre-processing pipeline subgraph that could be reused across multiple cameras in the same application or shared across different applications using the same camera processing logic.

