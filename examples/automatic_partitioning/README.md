# Automatic graph partitioning

This installed C++ example composes a full-DAG diamond (fan-out, fan-in, and
the linear tail) and compiles it with the accepted automatic partitioning
planner. It then prints the complete `ExecutionPlan::partitions()` result in
stable ID, lane, and member order and exercises the cold inspection handles:

- lookup by dense partition ID, authored name, operator path, and root operator handle;
- an intentionally failed dynamic lookup with its typed `kNotFound` error;
- a root `include_all()` partition that contains an embedded graph descendant;
- a hybrid plan with one exact named Green Context anchor and an automatic
  remainder, when the host exposes a usable CUDA Green Context capability.

The example does not assign Green Contexts to automatic partitions. The
compiler owns the automatic residual, while the named anchor is bound to the
queried device and structured quantum. Partition inspection is a compile-cold
operation; this program does not start the graph runtime.

## Build and run

From an installed SDK consumer build:

```bash
cmake -S /path/to/holoscan/share/examples/automatic_partitioning/cpp \
  -B build/automatic_partitioning \
  -Dholoscan_DIR=/path/to/holoscan/lib/cmake/holoscan
cmake --build build/automatic_partitioning
ctest --test-dir build/automatic_partitioning --output-on-failure
```

Run only the GPU-independent automatic and `include_all()` cases with
`automatic_partitioning --mode cpu`. Run only the Green Context hybrid case
with `automatic_partitioning --mode hybrid`. Invoking the executable without a
mode retains the combined walkthrough.

The regular example build installs the executable under
`${CMAKE_INSTALL_DATADIR}/examples/automatic_partitioning/cpp` (by default,
`share/examples/automatic_partitioning/cpp`) and honors a configured data
directory. It registers
`EXAMPLE_CPP_AUTOMATIC_PARTITIONING_CPU_TEST` and
`EXAMPLE_CPP_AUTOMATIC_PARTITIONING_HYBRID_TEST` for the installed-example
package test. The CPU test always evaluates automatic planning and
`include_all()` independently. A host without a usable CUDA driver, or without
two structured Green Context units, skips only the GPU-qualified hybrid test
with CTest's accepted skip code (`77`).

## Expected topology

The ordinary plan is the complete source-to-tail DAG:

```text
             -> left -\
source -----------------> join -> tail
             -> right -/
```

`partitions()` lists every partition, including each member path and its
origin. Automatic partitions have no authored names; use the ID, path, or
operator-handle lookup APIs instead of inventing a generated name. The
include-all case demonstrates that a root partition includes embedded
descendants, not only the root graph's direct operators.

Published Operator paths are exact canonical keys. Each decoded Graph or
Operator UTF-8 segment is percent-encoded bytewise before slash joining. Root
and embedded keys both use the no-leading-slash `OperatorPath` grammar shared
by Parameter records and Tensor output placement. Reuse a key returned by
`operator_paths()` verbatim with `lookup_for_path()`; lookup performs no decoding
or normalization.
