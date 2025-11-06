# Testing Operators

## Overview

Holoscan provides a test harness interface for testing Holoscan operators with:
- **Input/Output Port Management**: Easy setup of test data and validation
- **Condition Support**: Add execution conditions like CountCondition, PeriodicCondition
- **Validation Framework**: Built-in validators for exact equality, floating-point comparison, and custom validation
- **Fluent API**: Chainable method calls for clean, readable test setup


## Concepts

### Test Harness

The `OperatorTestHarness` is a specialized Holoscan application designed to test individual operators in isolation. It automates the setup of a test pipeline by creating source operators to provide input data and sink operators to collect and validate output data.

**Main Components:**

1. **`OperatorTestHarness<OperatorType, Args...>`**: The main test harness class that orchestrates the test
   - Creates and manages source operators for each input port
   - Creates and manages sink operators for each output port
   - Connects all operators in the test pipeline
   - Provides a fluent API for configuration

2. **`TestHarnessSourceOp<T>`**: A specialized operator that emits predetermined test data
   - Emits one value per compute cycle from a provided vector
   - Automatically manages iteration through test data
   - Used internally by the test harness for each input port

3. **`TestHarnessSinkOp<T>`**: A specialized operator that collects and validates outputs
   - Receives data from the operator under test
   - Applies validation functions to each received value
   - Tracks received data count for verification
   - Used internally by the test harness for each output port

4. **`OperatorTestBase`**:
   - A base test fixture class designed for use with Google Test (`gtest`).
   - Just a skeleton fixture, does not currently extend `SetUp` or `TearDown` methods.

**How It Works:**

When you create a test harness with `create_operator_test<YourOp>()`:
1. The harness creates a Holoscan application with your operator in the middle
2. For each `add_input_port()` call, it creates a `TestHarnessSourceOp` that feeds test data
3. For each `add_output_port()` call, it creates a `TestHarnessSinkOp` that collects and validates output
4. When you call `run_test()`, the entire pipeline executes and validations are performed automatically

The test harness ensures that:
- All input ports receive the same number of data elements
- Data flows correctly from sources through your operator to sinks
- Conditions (like `CountCondition`) control execution properly
- Validation failures are reported via Google Test assertions

### Validator Functions

Validator functions are callable objects that verify operator outputs during test execution. They follow the signature `void(const T&)` where `T` is the output data type. Validators are called automatically by `TestHarnessSinkOp` for each value received from the operator under test.

**Built-in Validators:**

Holoscan provides three types of built-in validators:

1. **`create_exact_equality_validator<T>(expected_values)`**
   - Compares each output against expected values using `operator==`
   - Best for: integers, strings, and other types with well-defined equality
   - Automatically tracks which output index is being validated
   ```cpp
   std::vector<int> expected = {2, 4, 6};
   auto validator = create_exact_equality_validator(expected);
   ```

2. **`create_float_equality_validator<T>(expected_values, tolerance = T{})`**
   - Compares floating-point values with approximate equality
   - Uses Google Test's `EXPECT_FLOAT_EQ` (default) or `EXPECT_NEAR` (with tolerance)
   - Best for: float, double, and other floating-point types
   ```cpp
   std::vector<float> expected = {1.5f, 2.7f, 3.1f};
   auto validator = create_float_equality_validator(expected, 0.01f);
   ```

3. **`create_transform_equality_validator<InputT, OutputT>(expected_values, transform_func)`**
   - Applies a transformation before comparing
   - Best for: complex types where you want to validate a specific property
   ```cpp
   auto validator = create_transform_equality_validator<MyStruct, int>(
     {10, 20, 30},
     [](const MyStruct& s) { return s.field; }
   );
   ```

**Using Multiple Validators:**

You can apply multiple validators to a single output port using the `validators<T>()` helper:

```cpp
test->add_output_port<float>("output", validators<float>(
  create_float_equality_validator(expected_values),
  custom_range_validator,
  custom_format_validator
));
```

**Custom Validators:**

Create custom validators by providing any callable that matches `void(const T&)`:

```cpp
auto custom_validator = [](const int& value) {
  EXPECT_GT(value, 0) << "Value must be positive";
  EXPECT_LT(value, 100) << "Value must be less than 100";
};

test->add_output_port<int>("output", validators<int>(custom_validator));
```

Custom validators can perform any verification logic and use any Google Test assertion macros (`EXPECT_*`, `ASSERT_*`). Validation failures will cause the test to fail with descriptive error messages.

## Basic Test Structure

Suppose we have an operator (`DoublerOp`) that has a single input and single output port, both `int`. The operator simply doubles the input value and emits that.

There are **two ways** to set up tests:

### Approach 1: Fluent API

*Fluent* is a design pattern that enables code to be written in a way that flows naturally, often by chaining method calls together. This style improves readability and expressiveness, making it easier to set up complex objects or configurations in a concise and intuitive manner. The term "fluent API" refers to this general programming approach and is not specific to Holoscan.

```cpp
#include "holoscan/test/test_harness.h"
#include "holoscan/test/validation_functions.h"

using namespace holoscan::test {

TEST_F(OperatorTestBase, BasicFluentTest) {
  // Test data
  std::vector<int> input_data = {1, 2, 3};
  std::vector<int> expected_output = {2, 4, 6};

  // Create and configure test with fluent API - everything in one chain
  auto test_harness = create_operator_test<DoublerOp>()
    ->add_input_port<int>("input", input_data)
    ->add_output_port<int>("output", holoscan::test::validators<int>(
      create_exact_equality_validator(expected_output)))
    ->add_condition<holoscan::CountCondition>("count", input_data.size());

  test_harness->run_test();

  // Additional verification if needed
  auto sink = test_harness->get_sink<int>("output");
  EXPECT_EQ(sink->get_received_count(), expected_output.size());
}

}  // namespace holoscan::test
```

### Approach 2: Step-by-Step Setup
```cpp
#include "holoscan/test/test_harness.h"
#include "holoscan/test/validation_functions.h"

using namespace holoscan::test {

TEST_F(OperatorTestBase, BasicStepByStepTest) {
  // Test data
  std::vector<int> input_data = {1, 2, 3};
  std::vector<int> expected_output = {2, 4, 6};

  // Create test harness first
  auto test_harness = create_operator_test<DoublerOp>();

  // Add components step by step (allows for conditional logic)
  test_harness->add_input_port<int>("input", input_data);
  test_harness->add_output_port<int>("output", holoscan::test::validators<int>(
    create_exact_equality_validator(expected_output)));
  test_harness->add_condition<holoscan::CountCondition>("count", input_data.size());

  test_harness->run_test();

  // Additional verification
  auto sink = test_harness->get_sink<int>("output");
  EXPECT_EQ(sink->get_received_count(), expected_output.size());
}

}  // namespace holoscan::test
```

## API Reference

### Creating Test Harness
```cpp
// Without parameters
auto test = create_operator_test<DoublerOp>();

// With parameters
auto test = create_operator_test<DoublerOp>(
  holoscan::Arg("param1") = value1,
  holoscan::Arg("param2") = value2
);
```

### Adding Input Ports

The values that are passed to the operator's input ports are defined and added via the `add_input_port` function. This function takes a `std::vector<DataType>` object, the elements of which will be passed to the operator one-by-one.

```cpp
test->add_input_port<DataType>("port_name", test_data_vector);
```

### Adding Output Ports

The output ports of the operator are added to the test harness using the `add_output_port` function. This function can be used in two ways: without a validator, to simply collect the output data, or with a validator function to automatically check that the output matches expected results.

```cpp
// Without validation
test->add_output_port<DataType>("port_name");

// With validation
test->add_output_port<DataType>("port_name",
  holoscan::test::validators<DataType>(validator_function));
```

### Adding Conditions
```cpp
test->add_condition<holoscan::CountCondition>("name", count);
test->add_condition<holoscan::PeriodicCondition>("name",
  holoscan::Arg("recess_period") = std::string("100ms"));
```

### Running Tests
```cpp
test->run_test();
```

## Testing Patterns

### Source Operators

Operators with output ports, but no input ports.

```cpp
auto test = create_operator_test<SourceOp>(params)
  ->add_condition<CountCondition>("count", iterations)
  ->add_output_port<T>("output", validators);

test->run_test();

// Verify internal state (optional)
auto src_op = test->get_operator_under_test();
// Check src_op internal state...
```

### Sink Operators

Operators with input ports, but no input ports.

```cpp
auto test = create_operator_test<SinkOp>()
  ->add_input_port<T>("input", test_data);

test->run_test();

// Verify internal state (optional)
auto sink_op = test->get_operator_under_test();
// Check sink_op internal state...
```

### Transform Operators

Operators with both input and output ports

```cpp
auto test = create_operator_test<TransformOp>(params)
  ->add_input_port<InputT>("input", input_data)
  ->add_output_port<OutputT>("output", output_validators)
  ->add_condition<CountCondition>("count", data_size);
```