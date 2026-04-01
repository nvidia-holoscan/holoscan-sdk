(holoscan-core-concepts)=
# Holoscan Core Concepts


An `Application` is composed of `Fragments`, each of which runs a graph of `Operators`. The implementation of that graph is sometimes referred to as a pipeline, or workflow, which can be visualized below:

:::{figure-md} fig-core-concepts-application
:align: center

![](images/core_concepts_application.png)

Core concepts: Application

:::

:::{figure-md} fig-core-concepts-port
:align: center

![](images/core_concepts_port.png)

Core concepts: Port

:::

The core concepts of the Holoscan API are:

- **{ref}`Application <exhale_class_classholoscan_1_1Application>`**: An
  application acquires and processes streaming data. An application is a
  collection of fragments where each fragment can be allocated to execute on an
  exclusive or shared physical node of a Holoscan cluster.
- **{ref}`Fragment <exhale_class_classholoscan_1_1Fragment>`**: A fragment is a building block of the application. It is a directed graph of operators. A fragment can be assigned to a physical node of a Holoscan cluster during execution. The runtime execution manages communication across fragments. In a fragment, Operators (Graph Nodes) are connected to each other by flows (Graph Edges).
- **{ref}`Operator <exhale_class_classholoscan_1_1Operator>`**: An operator is the most basic unit of work in this framework. An operator receives streaming data at an input port, processes it, and publishes it to one of its output ports. In applications executed by {cpp:class}`~holoscan::gxf::GXFExecutor`, an operator corresponds to the GXF-specific {ref}`Codelet <holoscan-core-concepts-gxf>` concept.
- **{ref}`Port <exhale_class_classholoscan_1_1IOSpec>`**: A port is an interaction point between two operators. Operators ingest data at Input ports and publish data at Output ports. `Port` is a Holoscan concept used by both {cpp:class}`~holoscan::gxf::GXFExecutor` and {cpp:class}`~holoscan::GPUResidentExecutor`. In {cpp:class}`~holoscan::gxf::GXFExecutor`, ports map to the GXF-specific `Receiver`, `Transmitter`, and `MessageRouter` concepts used to move data between connected operators.
- **{ref}`Executor <exhale_class_classholoscan_1_1Executor>`**: An executor that manages the execution of a fragment on a physical node. The framework provides executor implementations such as {cpp:class}`~holoscan::gxf::GXFExecutor` and {cpp:class}`~holoscan::GPUResidentExecutor` to execute an application.

(holoscan-core-concepts-gxf)=
## GXF Executor-specific Concepts

The following concepts are specific to applications executed by {cpp:class}`~holoscan::gxf::GXFExecutor`:

- **Codelet**: A GXF-specific unit of execution. In the Holoscan SDK, this role is represented by an {ref}`Operator <exhale_class_classholoscan_1_1Operator>`.
- **{ref}`(Operator) Resource <exhale_class_classholoscan_1_1Resource>`**:
        Resources such as system memory or a GPU memory pool that an
        operator needs to perform its job. Resources are allocated during
        the initialization phase of the application. This matches the
        semantics of GXF's Memory `Allocator` or any other components
        derived from the `Component` class in GXF.
(holoscan-concepts-condition)=
- **{ref}`Condition <exhale_class_classholoscan_1_1Condition>`**: A condition is a predicate that can be evaluated at runtime to determine if an operator should execute.
- **{ref}`Message <exhale_class_classholoscan_1_1Message>`**: A message is a generic data object used by operators to communicate information.

:::{note}
Holoscan 4.1 uses `FlowGraph` and `FlowGraphImpl` for the existing flow-oriented
graph API, along with the Python module `holoscan.flow_graphs`.
:::
