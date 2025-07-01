# Asynchronous Operator Execution Control Example

This example demonstrates how to coordinate the execution flow of operators running in separate threads using the `Operator::async_condition()` API.

Asynchronous operator execution control is crucial for applications that need fine-grained control over the execution order of operators, especially when specific processing steps must be performed in a precise sequence. This pattern is useful for implementing complex workflows, coordinating multiple processing pipelines, or integrating external control systems with a Holoscan application.

### Practical Use Cases

This pattern is particularly valuable in the following scenarios:
- Medical imaging workflows where processing steps must occur in a strict sequence
- Real-time systems that need to synchronize with external hardware or timing signals
- Applications with dynamic execution paths that depend on runtime decisions
- Systems requiring manual intervention or approval between processing stages
- Complex pipelines where some operators need to wait for others to complete specific tasks

### Comparison with Alternative Approaches

This asynchronous execution control pattern offers distinct advantages over other approaches:

| Approach | Description | When to Use | Limitations |
|----------|-------------|-------------|-------------|
| **Async Condition API** | Controls operator execution via event states | For fine-grained external control | More complex to implement |
| **Standard Flow-based** | Operators execute when inputs are available according to scheduler policies | For typical data processing pipelines | Limited external control over execution timing |
| **Status Tracking** | Uses operator status tracking to monitor execution | For monitoring rather than control | Reactive rather than proactive control |

## Overview

The application shows how to:
1. Create operators that can be controlled via their asynchronous scheduling conditions
2. Execute operators in a specific sequence from outside the Holoscan runtime
3. Implement a notification mechanism for operators to signal completion
4. Gracefully shut down an application with asynchronous operators

The example implements:
- A controller operator that manages execution state and coordinates other operators
- Multiple simple operators that wait for control signals before executing
- A thread-safe notification mechanism between operators and the controller
- External control of operators from the main application thread

The example uses the `EventBasedScheduler` with multiple worker threads, though it would work with any scheduler type (such as `GreedyScheduler` or `MultiThreadScheduler`).

### Application Workflow

The execution flow follows this pattern:
```
Main Thread                ControllerOp Thread             SimpleOp Threads (op1, op2, op3)
-------------              ------------------              --------------------------------
Start app_run_async()      Initialize (kEventWaiting)      Initialize (kWait)
                           Start Controller                Wait for execution
Get controller ──────────> Return controller
Execute op3 ─────────────> Set op3 to (kEventDone) ──────> op3 executes
Wait for notification <─── Wait for notification   <────── op3 sends notification
Execute op2 ─────────────> Set op2 to (kEventDone) ──────> op2 executes
Wait for notification <─── Wait for notification   <────── op2 sends notification
Execute op1 ─────────────> Set op1 to (kEventDone) ──────> op1 executes
Wait for notification <─── Wait for notification   <────── op1 sends notification
Shutdown ────────────────> Set ops to kEventNever          Operators stop
                           Set controller to (kEventDone)
                           Controller compute() executes
                           Stop execution
Application exits
```

## C++ API

The application consists of two operator types:
1. `SimpleOp`: A basic operator that:
   - Initializes in kWait state
   - Executes its compute method when triggered
   - Notifies the controller upon completion
   - Returns to kWait state to await the next trigger

2. `ControllerOp`: A controller operator that:
   - Maintains a registry of all operators in the application
   - Provides methods to execute specific operators by name
   - Uses a notification callback to receive signals from operators
   - Controls application shutdown by setting appropriate event states

Key implementation aspects:
- SimpleOp operators use `async_condition()->event_state(kWait)` to pause execution
- The controller uses `async_condition()->event_state(kEventWaiting)` to prevent the application from being terminated at start by the deadlock detector
- The main thread triggers operators via `controller->execute_operator("op_name")`
- Notification callbacks and condition variables ensure thread-safe synchronization

### Build instructions

Built with the SDK, see instructions from the top level README.

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
./examples/execution_control/async_operator_execution_control/cpp/async_operator_execution_control
```

## Python API

The Python implementation demonstrates the same concepts using the Python API. It includes the same two operator types with equivalent functionality:
1. `SimpleOp`: A basic operator that:
   - Initializes in kWait state
   - Executes its compute method when triggered
   - Notifies the controller upon completion
   - Returns to kWait state to await the next trigger
2. `ControllerOp`: A controller operator that:
   - Maintains a registry of all operators in the application
   - Provides methods to execute specific operators by name
   - Uses a notification callback to receive signals from operators
   - Controls application shutdown by setting appropriate event states

The Python example demonstrates the same execution control patterns using Python's equivalent APIs:
- Using `async_condition.event_state` to control operator execution states (equivalent to C++ `async_condition()->event_state()`)
- Implementing notification callbacks between operators
- Controlling operator execution sequence from outside the runtime
- Thread synchronization using Python's threading primitives (Condition variables and locks)

### Run instructions

First, go in your `build` or `install` directory (automatically done by `./run launch`).

Then, run:
```bash
python3 ./examples/execution_control/async_operator_execution_control/python/async_operator_execution_control.py
```

## Key API Features Demonstrated

- `Operator::async_condition()->event_state()`: Controls operator execution states
- `AsynchronousEventState` enum values:
  - `kWait` (`WAIT`): Makes the operator wait for the controller to change its state to kReady
  - `kEventWaiting` (`EVENT_WAITING`): Prevents the application from being terminated at start by the deadlock detector (at least one operator needs to be in this state)
  - `kEventDone` (`EVENT_DONE`): Signals the scheduler that the operator is ready to execute
  - `kEventNever` (`EVENT_NEVER`): Changes the operator's scheduling condition type to kNever, preventing execution
- Thread synchronization using condition variables and callbacks
- External control of operator execution from outside the runtime
- Graceful application shutdown sequence

For more detailed information about the asynchronous condition API, see the Holoscan SDK API documentation on `holoscan::AsynchronousCondition` and `holoscan::AsynchronousEventState`.

### Important Considerations

When implementing this pattern, keep these points in mind:
- **Deadlock prevention**: Ensure at least one operator is in `kEventWaiting` state rather than `kWait` to prevent deadlock detection from terminating the application
- **Thread safety**: All communication between operators must be thread-safe; use appropriate synchronization primitives
- **Error handling**: Consider how to handle failures in asynchronously executing operators
- **Performance impact**: Synchronization between threads can introduce overhead; use this pattern when the control benefits outweigh performance costs
- **Debugging complexity**: Asynchronous execution patterns can be more challenging to debug; consider adding detailed logging

### Advanced Usage and Extensions

This example demonstrates the basic pattern, but it can be extended in several ways:

- **Dynamic operator selection**: Instead of a fixed execution sequence, implement logic to determine which operator to execute next based on runtime conditions
- **External API integration**: Expose the controller's methods through a network API to allow remote control of the application
- **Conditional execution**: Add support for conditions that determine whether an operator should execute based on the results of previous operations
- **Progress tracking**: Implement a progress monitoring system to track the execution status of long-running operations
- **Timeout handling**: Add timeout mechanisms to prevent infinite waiting if an operator fails to complete

## Expected Output

The example will output:
1. An introduction explaining the example and key concepts
2. Status messages showing operators being executed in sequence (op3 → op2 → op1)
3. Notification messages when operators complete execution
4. Shutdown sequence messages
5. Confirmation that the application has completed successfully

Sample output (from the C++ implementation, Python output is similar):
```
This example demonstrates async operator execution control.
The controller operator runs in a separate thread and synchronizes
the execution of multiple SimpleOp operators using async_condition().
-------------------------------------------------------------------
Key concepts demonstrated:
1. Using asynchronous conditions to control operator execution states
2. External execution control from outside the Holoscan runtime
3. Notification callbacks for coordination between operators
4. Manual scheduling of operators in a specific order (op3 → op2 → op1)
5. Graceful shutdown of an application with async operators
-------------------------------------------------------------------
Execution flow:
- All operators start in kWait state except the controller (kEventWaiting)
- Main thread gets controller and executes operators in sequence
- Each operator signals completion via the notification callback
- The controller then shuts down the application
-------------------------------------------------------------------
[op3] Executing compute method
[op2] Executing compute method
[op1] Executing compute method
[controller] Shutting down controller
[controller] Stopping controller execution
-------------------------------------------------------------------
Application completed. All operators have finished execution.
