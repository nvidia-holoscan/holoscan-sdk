# Operator Status Tracking Example

This example demonstrates how to use the Operator Status Tracking API in Holoscan SDK to monitor the execution status of operators in your application.

## Overview

The application shows how to:
1. Create a pipeline with source, processor, and consumer operators
2. Use an independent monitoring operator to track the status of other operators
3. Make execution flow decisions based on operator status
4. Terminate the application when all operators have completed their work

The example implements a simple data processing pipeline where:
- A source operator emits a fixed number of values and then stops
- A processor operator transforms those values
- A consumer operator receives and displays the processed values
- A monitor operator tracks the status of all other operators and terminates the application when processing is complete

## C++ API

The application consists of four operator types:
1. `FiniteSourceOp`: Emits a specified number of values and then stops
2. `ProcessorOp`: Receives data, processes it by multiplying by 2, and emits the result
3. `ConsumerOp`: Receives and displays the processed data
4. `MonitorOp`: Monitors the status of all other operators and terminates the application when all operators have completed

The monitor operator uses the ExecutionContext API to:
- Retrieve the current status of each operator using `context.get_operator_status()`
- Detect when all operators are idle for a sufficient period
- Terminate itself using `stop_execution()` when processing is complete

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
./examples/execution_control/operator_status_tracking/cpp/operator_status_tracking
```

## Python API

The Python implementation demonstrates the same concepts using the Python API. It includes the same four operator types with equivalent functionality:
1. `FiniteSourceOp`: Source operator that emits a fixed number of values
2. `ProcessorOp`: Processes incoming data by multiplying by 2
3. `ConsumerOp`: Consumes and displays the processed data
4. `MonitorOp`: Monitors operator status and terminates execution when complete

The Python example demonstrates:
- Using the Python bindings for the `OperatorStatus` enum
- Retrieving operator status with `context.get_operator_status()`
- Terminating execution with the `stop_execution()` method

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
python3 ./examples/execution_control/operator_status_tracking/python/operator_status_tracking.py
```

## Key API Features Demonstrated

- `ExecutionContext.get_operator_status()`: Retrieves the current status of an operator
- `OperatorStatus` enum values: NOT_STARTED, START_PENDING, STARTED, TICK_PENDING, TICKING, IDLE, STOP_PENDING
- `Operator.stop_execution()`: Stops the execution of an operator by setting its condition to NEVER
- `ExecutionContext.find_operator()`: Finds an operator by name within the application

## Expected Output

The example will:
1. Show the source operator emitting 5 values
2. Display the processor transforming those values
3. Show the consumer receiving the transformed values
4. Demonstrate the monitor tracking status of all operators
5. Terminate when all operators have completed their work

The final output will indicate that the application has completed successfully after all operators have finished processing.
