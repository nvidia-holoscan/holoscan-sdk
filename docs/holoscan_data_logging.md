(holoscan-data-logging)=

# Data Logging

Holoscan provides a flexible data logging system that allows applications to capture and log data flowing through operator workflows. This system is designed to help with debugging, monitoring, and analyzing the behavior of Holoscan applications.

## Overview

The data logging system consists of several key components:

- **{cpp:class}`DataLogger <holoscan::DataLogger>`** - The core interface that defines methods for logging different data types
- **{cpp:class}`DataLoggerResource <holoscan::DataLoggerResource>`** - A resource-based implementation with common configuration parameters
- **{cpp:class}`BasicConsoleLogger <holoscan::data_loggers::BasicConsoleLogger>`** - A concrete implementation that logs data to the console
- **{cpp:class}`SimpleTextSerializer <holoscan::data_loggers::SimpleTextSerializer>`** - A serializer for converting data to human-readable text

### When Logging Occurs

Data logging occurs automatically during the execution of operator workflows:

- **Input logging**: When operators call `receive()` methods on their input ports
- **Output logging**: When operators call `emit()` methods on their output ports

The timing of logging corresponds exactly to these `receive` and `emit` calls within each operator's `compute()` method.

### Data Types Supported

The data logging system provides specialized methods for different data types:

- **`log_data()`** - For general data types (passed as `std::any`)
- **`log_tensor_data()`** - For `Tensor` objects with optional data content logging
- **`log_tensormap_data()`** - For `TensorMap` objects with optional data content logging

## DataLogger Interface

The `DataLogger` interface defines the contract that all data loggers must implement:

`````{tab-set}
````{tab-item} C++
```cpp
class DataLogger {
 public:
  // Log general data types
  virtual bool log_data(std::any data, const std::string& unique_id,
                        int64_t acquisition_timestamp = -1,
                        std::shared_ptr<MetadataDictionary> metadata = nullptr,
                        IOSpec::IOType io_type = IOSpec::IOType::kOutput) = 0;

  // Log Tensor data
  virtual bool log_tensor_data(const std::shared_ptr<Tensor>& tensor,
                               const std::string& unique_id,
                               int64_t acquisition_timestamp = -1,
                               const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                               IOSpec::IOType io_type = IOSpec::IOType::kOutput) = 0;

  // Log TensorMap data
  virtual bool log_tensormap_data(const TensorMap& tensor_map,
                                  const std::string& unique_id,
                                  int64_t acquisition_timestamp = -1,
                                  const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                                  IOSpec::IOType io_type = IOSpec::IOType::kOutput) = 0;

  // Configuration methods
  virtual bool should_log_output() const = 0;
  virtual bool should_log_input() const = 0;
};
```
````
````{tab-item} Python
```python
class DataLogger:
    def log_data(self, data, unique_id: str, acquisition_timestamp: int = -1,
                 metadata=None, io_type=IOSpec.IOType.OUTPUT) -> bool:
        """Log general data types"""
        pass

    def log_tensor_data(self, tensor, unique_id: str, acquisition_timestamp: int = -1,
                        metadata=None, io_type=IOSpec.IOType.OUTPUT) -> bool:
        """Log Tensor data"""
        pass

    def log_tensormap_data(self, tensor_map, unique_id: str, acquisition_timestamp: int = -1,
                           metadata=None, io_type=IOSpec.IOType.OUTPUT) -> bool:
        """Log TensorMap data"""
        pass

    def should_log_output(self) -> bool:
        """Check if output ports should be logged"""
        pass

    def should_log_input(self) -> bool:
        """Check if input ports should be logged"""
        pass
```
````
`````

## DataLoggerResource Base Class

The `DataLoggerResource` class provides a convenient base implementation with common configuration parameters:

### Configuration Parameters

- **`log_inputs`** (bool, default: `true`) - Whether to log data received on input ports
- **`log_outputs`** (bool, default: `true`) - Whether to log data emitted on output ports
- **`log_metadata`** (bool, default: `true`) - Whether to include metadata in logs
- **`log_tensor_data_content`** (bool, default: `false`) - Whether to log actual tensor data arrays (if false, only header info is logged)
- **`allowlist_patterns`** (vector<string>, default: empty) - Regex patterns for message IDs to always log
- **`denylist_patterns`** (vector<string>, default: empty) - Regex patterns for message IDs to never log

Also note that Holoscan resources can have other Resource classes as a parameter as demonstrated by having a separate `SimpleTextSerializer` resource in the concrete `BasicConsoleLogger` class that inherits from `DataLoggerResource`. Note that when implementing a class that inherits from this one, it is mandatory to call the `DataLoggerResource::setup` method within the derived class's `setup` method as in this example from `BasicConsoleLogger`. If the initialize method is overridden the parent class's `initialize` method should also be called. This is an example from `BasicConsoleLogger`:

```cpp

void BasicConsoleLogger::setup(ComponentSpec& spec) {
  spec.param(serializer_, "serializer", "Serializer", "Serializer to use for logging data");
  // setup the parameters present on the base DataLoggerResource
  DataLoggerResource::setup(spec);
}

void BasicConsoleLogger::initialize() {
  // Find if there is an argument for 'serializer'
  auto has_serializer = std::find_if(
      args().begin(), args().end(), [](const auto& arg) { return (arg.name() == "serializer"); });

  // Create appropriate serializer if none was provided
  if (has_serializer == args().end()) {
    add_arg(Arg("serializer", fragment()->make_resource<SimpleTextSerializer>("serializer")));
  }

  // call parent initialize after adding missing serializer arg above
  DataLoggerResource::initialize();
}

```

:::{note}
If `allowlist_patterns` is specified, only messages matching those patterns will be logged. If no allowlist is specified, all messages will be logged except those matching `denylist_patterns`.
:::

:::{note}
Currently this simple `DataLoggerResource` performs logging synchronously on the same thread that is executing the `Operator::compute` call. In cases where logging overhead may be non-negligable (e.g. logging tensor contents to disk), a different design with asynchronous logging managed by dedicated logging threads will likely be advantageous. An example of such an `AsyncDataLoggerResource` may be provided in a future release of the SDK (or via an example on Holohub).
:::

## BasicConsoleLogger Example

The `BasicConsoleLogger` is a concrete implementation that outputs logs to the console:

`````{tab-set}
````{tab-item} C++
```cpp
#include "holoscan/data_loggers/basic_console_logger/basic_console_logger.hpp"
#include "holoscan/data_loggers/basic_console_logger/simple_text_serializer.hpp"

class MyApp : public holoscan::Application {
 public:
  void compose() override {
    // Create operators (example)
    auto source = make_operator<SourceOp>("source");
    auto processor = make_operator<ProcessorOp>("processor");
    auto sink = make_operator<SinkOp>("sink");

    // Create and configure data logger
    auto logger = make_resource<holoscan::data_loggers::BasicConsoleLogger>(
        "console_logger",
        Arg("log_inputs", true),
        Arg("log_outputs", true),
        Arg("log_metadata", false),
        Arg("log_tensor_data_content", false),
        Arg("denylist_patterns", std::vector<std::string>{".*debug.*"})
    );

    // Add logger to application
    add_data_logger(logger);

    // Define workflow
    add_flow(source, processor);
    add_flow(processor, sink);
  }
};
```
````
````{tab-item} Python
```python
from holoscan.core import Application
from holoscan.data_loggers import BasicConsoleLogger, SimpleTextSerializer

class MyApp(Application):
    def compose(self):
        # Create operators (example)
        source = SourceOp(self, name="source")
        processor = ProcessorOp(self, name="processor")
        sink = SinkOp(self, name="sink")

        # Create and configure data logger
        logger = BasicConsoleLogger(
            self,
            name="console_logger",
            log_inputs=True,
            log_outputs=True,
            log_metadata=False,
            log_tensor_data_content=False,
            denylist_patterns=[".*debug.*"]
        )

        # Add logger to application
        self.add_data_logger(logger)

        # Define workflow
        self.add_flow(source, processor)
        self.add_flow(processor, sink)
```
````
`````

The example above shows example code adding the logger within the `compose` method, but it can
also be added from the `main` application file via as done in the following example applications:

1. Tensor Interop ([C++](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v3.4.0/examples/tensor_interop/cpp/tensor_interop.cpp), [Python](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v3.4.0/examples/tensor_interop/python/tensor_interop.py))
2. Multithread Scheduling ([C++](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v3.4.0/examples/multithread/cpp/multithread.cpp), [Python](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v3.4.0/examples/multithread/python/multithread.py))
3. Video Replayer ([C++](https://github.com/nvidia-holoscan/holoscan-sdk/tree/v3.4.0/examples/video_replayer/cpp/video_replayer.cpp), [Python](https://github.com/nvidia-holoscan/holoscan-sdk/blob/v3.4.0/examples/video_replayer/python/video_replayer.py))

As with any other resource or operator in the SDK, parameters can be passed in directly via
arguments or indirectly via reading from the YAML config.

## YAML Configuration

Data loggers can be configured using YAML configuration files, making it easy to adjust logging behavior without recompiling:

### YAML Configuration Example

```yaml
# Data logging configuration
data_logging: true

basic_console_logger:
  log_inputs: true
  log_outputs: true
  log_metadata: true
  log_tensor_data_content: false
  allowlist_patterns: []
  denylist_patterns:
    - ".*debug.*"
    - ".*internal.*"

# Optional: Configure the text serializer
simple_text_serializer:
  max_elements: 10
  max_metadata_items: 5
  log_python_object_contents: true  # Python only
```

### Loading Configuration from YAML

`````{tab-set}
````{tab-item} C++
```cpp
class MyApp : public holoscan::Application {
 public:
  void compose() override {
    // Create operators
    auto source = make_operator<SourceOp>("source");
    auto processor = make_operator<ProcessorOp>("processor");
    auto sink = make_operator<SinkOp>("sink");

    // Check if data logging is enabled
    auto enable_data_logging = from_config("data_logging").as<bool>();
    if (enable_data_logging) {
      auto text_serializer = make_resource<holoscan::data_loggers::SimpleTextSerializer>(
        "text-serializer",
        from_config("simple_text_serializer")
      )

      // Create logger with YAML configuration
      auto logger = make_resource<holoscan::data_loggers::BasicConsoleLogger>(
          "console_logger",
          holoscan::Arg("serializer", text_serializer),
          from_config("basic_console_logger")
      );
      add_data_logger(logger);
    }

    // Define workflow
    add_flow(source, processor);
    add_flow(processor, sink);
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<MyApp>();

  // Load configuration
  app->config("path/to/config.yaml");

  app->run();
  return 0;
}
```
````
````{tab-item} Python
```python
from holoscan.core import Application
from holoscan.data_loggers import BasicConsoleLogger, SimpleTextSerializer

class MyApp(Application):
    def compose(self):
        # Create operators
        source = SourceOp(self, name="source")
        processor = ProcessorOp(self, name="processor")
        sink = SinkOp(self, name="sink")

        # Check if data logging is enabled
        config = self.kwargs("data_logging")
        if config.get("data_logging", False):
            # Create logger with YAML configuration
            logger_config = self.kwargs("basic_console_logger")
            serializer_config = self.kwargs("simple_text_serializer")

            logger = BasicConsoleLogger(
                self,
                name="console_logger",
                serializer=SimpleTextSerializer(
                    self,
                    name="text_serializer",
                    **serializer_config
                ),
                **logger_config
            )
            self.add_data_logger(logger)

        # Define workflow
        self.add_flow(source, processor)
        self.add_flow(processor, sink)

def main():
    app = MyApp()

    # Load configuration
    app.config("path/to/config.yaml")

    app.run()

if __name__ == "__main__":
    main()
```
````
`````

## Custom Data Logger Implementation

You can create custom data loggers by implementing the `DataLogger` interface. To be able to use
Holoscan Parameters and configure them via YAML it may be useful to inherit from the provided
`DataLoggerResource` (as done for `BasicConsoleLogger`). Note that it is not **required** to inherit
from `DataLoggerResource`, though, only the `DataLogger` interface.

`````{tab-set}
````{tab-item} C++
```cpp
#include "holoscan/core/resources/data_logger.hpp"

class MyCustomLogger : public holoscan::DataLoggerResource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS_SUPER(MyCustomLogger, DataLoggerResource)

  void setup(ComponentSpec& spec) override {
    // Add custom parameters
    spec.param(output_file_, "output_file", "Output File",
               "Path to output log file", std::string(""));

    // Call parent setup for common parameters
    DataLoggerResource::setup(spec);
  }

  bool log_data(std::any data, const std::string& unique_id,
                int64_t acquisition_timestamp = -1,
                std::shared_ptr<MetadataDictionary> metadata = nullptr,
                IOSpec::IOType io_type = IOSpec::IOType::kOutput) override {
    // Implement custom logging logic
    return true;
  }

  bool log_tensor_data(const std::shared_ptr<Tensor>& tensor,
                       const std::string& unique_id,
                       int64_t acquisition_timestamp = -1,
                       const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                       IOSpec::IOType io_type = IOSpec::IOType::kOutput) override {
    // Implement custom tensor logging
    return true;
  }

  bool log_tensormap_data(const TensorMap& tensor_map,
                          const std::string& unique_id,
                          int64_t acquisition_timestamp = -1,
                          const std::shared_ptr<MetadataDictionary>& metadata = nullptr,
                          IOSpec::IOType io_type = IOSpec::IOType::kOutput) override {
    // Implement custom tensor map logging
    return true;
  }

 private:
  Parameter<std::string> output_file_;
};
```
````
````{tab-item} Python
```python
from holoscan.core import DataLoggerResource, ComponentSpec

class MyCustomLogger(DataLoggerResource):
    def __init__(self, fragment, output_file="", *args, **kwargs):
        self.output_file = output_file
        super().__init__(fragment, *args, **kwargs)

    def setup(self, spec: ComponentSpec):
        spec.param("output_file", "")
        # Call parent setup for common parameters
        super().setup(spec)

    def log_data(self, data, unique_id: str, acquisition_timestamp: int = -1,
                 metadata=None, io_type=None) -> bool:
        # Implement custom logging logic
        return True

    def log_tensor_data(self, tensor, unique_id: str, acquisition_timestamp: int = -1,
                        metadata=None, io_type=None) -> bool:
        # Implement custom tensor logging
        return True

    def log_tensormap_data(self, tensor_map, unique_id: str, acquisition_timestamp: int = -1,
                           metadata=None, io_type=None) -> bool:
        # Implement custom tensor map logging
        return True
```
````
`````

## Filtering and Pattern Matching

Data loggers inheriting from `DataLoggerResource` will automatically support filtering of messages using regex patterns:

```yaml
basic_console_logger:
  # Only log messages from specific operators with "source" or "processor" appearing in the operator or port name
  allowlist_patterns:
    - ".*source.*"
    - ".*processor.*"

  # Exclude messages from ports with "debug" appearing in their name
  denylist_patterns:
    - ".*debug.*"
```

For a distributed application, the fragments will be named and the `unique_id` format used for each port will be:

- `{fragment_name}.{operator_name}.{port_name}`
- `{fragment_name}.{operator_name}.{port_name}:index`   (for N:1 receiver ports (`IOSpec::kAnySize`))

For non-distributed applications, the single fragment is typically not named and the following simpler `unique_id` format is used:

- `{operator_name}.{port_name}`
- `{operator_name}.{port_name}:index`   (for N:1 receiver ports (`IOSpec::kAnySize`))

The `allowlist_patterns` and `denylist_patterns` provide a way to include or exclude messages based on operator and/or port names. Note that `allowlist_patterns` is used whenever it is provided. Only if `allowlist_patterns` is empty will `denylist_patterns` be applied.

:::{tip}
Use the [multithread example](https://github.com/nvidia-holoscan/holoscan-sdk/blob/main/examples/multithread) as a reference for a complete working implementation with data logging enabled.
:::

:::{note}
Data logging can impact performance, especially when `log_tensor_data_content` is enabled for large tensors. Use filtering patterns to log only the data you need for debugging or monitoring.
:::