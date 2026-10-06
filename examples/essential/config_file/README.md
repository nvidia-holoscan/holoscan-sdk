# essential/config_file: from YAML or JSON to a compiled plan

> **Essential module.** This example uses the configuration and parameter-tooling libraries from
> the Core installation closure. `config` and `config_io` remain separate install components, while
> `params` and `params_io` are exported through `holoscan-core`. A plain `find_package(holoscan)`
> imports the closure; no `COMPONENTS` request is required for this example.

This example shows the normal startup configuration path in Holoscan SDK 5:

```text
load file -> try_decode<T>() -> operator.set_params() -> compile -> run
```

The application loads equivalent JSON and YAML files, decodes one typed parameter value, compiles
the graph, runs it, and archives the stopped run's parameters. It also restores that archive into a
fresh graph to demonstrate the snapshot path.

Installed consumers link the parameter I/O library, which ships with the `holoscan-core`
component and is not a separately requestable `find_package` component:

```cmake
find_package(holoscan REQUIRED CONFIG)
target_link_libraries(my_app PRIVATE holoscan::params_io)
```

## Startup configuration

Load a file with an explicit format:

```cpp
auto config = holoscan::config_io::load(
    "detector.yaml", holoscan::config_io::Format::kYaml);
```

Decode the configuration with an explicit baseline for omitted fields:

```cpp
const DetectorParams baseline{
    .threshold = 6U,
    .gain = 1.0F,
    .window = 8U,
};
auto params = holoscan::params::try_decode<DetectorParams>(config, "/detector", baseline);
if (!params) {
  std::cerr << holoscan::params::format_error(params.error()) << '\n';
  return 1;
}

holoscan::Graph graph{"config-file"};
auto detector = graph.op<Detector>("detector");
detector.set_params(std::move(*params));
auto plan = holoscan::compile(graph);
```

`PreparedParams<DetectorParams>` owns both the value and its configuration evidence. Fields absent
from `/detector` use the explicit baseline. Assigning the prepared value to the operator retains its
evidence for compilation.

The short `load()` form throws on startup failure. `try_load()` returns the structured error.
Likewise, `decode()` throws and `try_decode()` returns a result. This example uses `try_decode()`
because `decode()` has no explicit-baseline overload and the example also checks a deliberate
duplicate-field failure. One catch at the application boundary reports any throwing configuration
failure without adding checks around each startup call.

Operator `set_params()` does not freeze or snapshot the graph. The graph owns the parameter value
and evidence, and `compile()` resolves them against the final graph, profile, placement, and schema.
Graph construction can therefore continue afterward without making call order change its meaning.

For an advanced all-or-nothing sparse update spanning several operators, use explicit graph
attachment:

```cpp
auto backup_detector = graph.op<Detector>("backup_detector");
auto configured = holoscan::params::set_params(
    graph,
    *config,
    {
        holoscan::params::bind(detector, "/detector"),
        holoscan::params::bind(backup_detector, "/detector"),
    });
if (!configured) {
  std::cerr << holoscan::params::format_error(configured.error()) << '\n';
  return 1;
}
```

`bind()` maps a typed operator handle to a configuration path. It does not infer the graph or attach
anything by itself. The example application uses the smaller typed path above.

## Paths and schemas

Document paths are RFC 6901 JSON Pointers. `/profiles/camera~1front/1` and an operator named
`camera/front` use different escaping rules. Advanced `params::bind()` does not infer one from the
other.

The operator schema validates field names, aliases, types, ranges, and other declared constraints.
Built-in field codecs handle common containers. Custom container element types need only their
element codec when a field-local codec can be composed.

## Snapshot archive

After the run stops, the example captures and saves the detector parameters:

```cpp
auto snapshot = session.capture_params();
auto detector_snapshot = snapshot->slice("detector");
auto saved = holoscan::params::save(*detector_snapshot, archive_path);
```

Restore uses the same graph-owned model:

```cpp
auto restored = holoscan::params::load_and_set_params(
    restore_graph,
    archive_path,
    holoscan::params::ParamRebindTargetScope::operator_(detector_path));
```

The example uses a fresh graph because the original graph has already been compiled and run. The
snapshot remains portable input that can be attached to another compatible graph. Attachment mutates
that graph's authoring state and is not a non-mutating restore preflight.

A graph accepts one pending all-or-nothing parameter transaction. Advanced configuration and
snapshot restore share that slot, including across embedded graphs.

## Build

From the `cpp/` directory:

```bash
cmake -B build -S .
cmake --build build
```

## Run

Still from the `cpp/` directory:

```bash
./build/config_file --config-dir .
```

The final line is:

```text
config_file: digests=match effective=(threshold=9 gain=2.5 window=8) duplicate=PARAM_CONFIG_FIELD_DUPLICATE roundtrip=(saved=3 contributed=3 retained=0) detections=3/6 termination=stopped
```

For temporary archive locations and cleanup, see the
[artifact-storage policy](../../compile_and_check/README.md#temporary-artifacts).
