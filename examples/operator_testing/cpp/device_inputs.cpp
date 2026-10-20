/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cuda_runtime_api.h>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <holoscan/cuda_execution/cuda_green_context.hpp>
#include <holoscan/memory/tensor_storage.hpp>
#include <holoscan/memory/unbounded_allocator.hpp>
#include <holoscan/testing/operator_test_harness.hpp>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace holoscan;
class DeviceConcatOp final : public Operator<> {
 public:
  void setup(OperatorSpec& spec) override {
    const TensorInputSpec input_spec{.representation = {.memory_kind = MemoryKind::kCudaDevice,
                                                        .dtype = DLDataType{kDLUInt, 8, 1},
                                                        .rank = 1U},
                                     .bounds = tensor_bounds(4U)};
    spec.input(left, "left").expects_tensor(input_spec);
    spec.input(right, "right").expects_tensor(input_spec);
    spec.output(out, "out")
        .produces_tensor(TensorOutputSpec{.representation = {.memory_kind = MemoryKind::kCudaDevice,
                                                             .dtype = DLDataType{kDLUInt, 8, 1},
                                                             .rank = 1U},
                                          .bounds = tensor_bounds(8U),
                                          .storage = TensorOutputStorage::kRuntimePool});
  }
  Contract contract() const override {
    Contract result;
    result.trigger(OnAligned{.ports = ports(left, right), .window = std::chrono::nanoseconds{0}})
        .input(left, Exact{})
        .input(right, Exact{});
    return result;
  }
  expected<void, Error> compute(ExecutionContext& context) override {
    auto a = left.receive();
    auto b = right.receive();
    if (!a)
      return make_unexpected(std::move(a).error());
    if (!b)
      return make_unexpected(std::move(b).error());
    if (a->data.nbytes() != 4 || b->data.nbytes() != 4)
      return make_unexpected(Error{ErrorCode::kInvalidArgument, "expected four bytes per input"});
    const auto stream = context.cuda_stream();
    auto ready = a->data.synchronize_for_consumer_stream(stream);
    if (!ready)
      return ready;
    ready = b->data.synchronize_for_consumer_stream(stream);
    if (!ready)
      return ready;
    const std::array<std::int64_t, 1> shape{8};
    auto loan = out.allocate_tensor(shape);
    if (!loan)
      return make_unexpected(std::move(loan).error());
    auto copied = loan->write_and_commit(stream, [&](const auto& write) -> expected<void, Error> {
      auto status =
          cudaMemcpyAsync(write.data(), a->data.data(), 4U, cudaMemcpyDeviceToDevice, stream.get());
      if (status == cudaSuccess)
        status = cudaMemcpyAsync(static_cast<std::byte*>(write.data()) + 4U,
                                 b->data.data(),
                                 4U,
                                 cudaMemcpyDeviceToDevice,
                                 stream.get());
      if (status != cudaSuccess)
        return make_unexpected(Error{ErrorCode::kFailure, cudaGetErrorString(status)});
      return {};
    });
    if (!copied)
      return copied;
    return out.emit(std::move(*loan));
  }
  Input<Tensor> left;
  Input<Tensor> right;
  Output<Tensor> out;
};

Tensor device_input(std::uint8_t value) {
  const std::array<std::int64_t, 1> shape{4};
  auto tensor = make_tensor(shape,
                            DLDataType{kDLUInt, 8, 1},
                            std::nullopt,
                            MemoryStorageType::kDevice,
                            std::make_shared<UnboundedAllocator>());
  if (!tensor)
    throw RuntimeError{std::move(tensor).error()};
  auto destination = tensor->mutable_data();
  if (!destination)
    throw RuntimeError{std::move(destination).error()};
  const auto status = cudaMemset(*destination, value, 4U);
  if (status != cudaSuccess)
    throw std::runtime_error{cudaGetErrorString(status)};
  return std::move(*tensor);
}
// Query and report the optional Green Context deployment before creating the test harness.
[[nodiscard]] std::optional<std::uint32_t> configure_green_context(CompileOptions& options,
                                                                   DeviceId device, int ordinal,
                                                                   bool shared_partition) {
  cudaDeviceProp properties{};
  int driver_version{};
  int runtime_version{};
  for (const auto status : {cudaGetDeviceProperties(&properties, ordinal),
                            cudaDriverGetVersion(&driver_version),
                            cudaRuntimeGetVersion(&runtime_version)}) {
    if (status != cudaSuccess)
      throw std::runtime_error{cudaGetErrorString(status)};
  }
  std::cout << "CUDA device " << ordinal << ": " << properties.name
            << ", compute_capability=" << properties.major << '.' << properties.minor
            << ", driver_api_version=" << driver_version << ", runtime_version=" << runtime_version
            << '\n';
  const auto capability = query_cuda_green_context_capability(device);
  if (!capability) {
    std::cout << "SKIP: Green Context unavailable: " << capability.error().message << '\n';
    return std::nullopt;
  }
  std::cout << "Green Context structured split: total_sms=" << capability->total_sm_count
            << ", min_partition_size=" << capability->min_sm_partition_size
            << ", coscheduled_alignment=" << capability->sm_coscheduled_alignment
            << ", quantum=" << capability->structured_sm_quantum
            << ", max_contexts=" << capability->max_structured_context_count << '\n';
  const std::uint32_t required_contexts = shared_partition ? 1U : 2U;
  if (capability->structured_sm_quantum == 0U ||
      capability->max_structured_context_count < required_contexts) {
    std::cout << "SKIP: Green Context requires " << required_contexts
              << " structured group(s); device " << ordinal
              << " provides max_contexts=" << capability->max_structured_context_count
              << ", quantum=" << capability->structured_sm_quantum << '\n';
    return std::nullopt;
  }
  const std::uint32_t requested_sm_count = capability->structured_sm_quantum;
  options.deployment.bind_cuda_green_context("operator",
                                             {.device = device, .sm_count = requested_sm_count});
  return requested_sm_count;
}
void verify_device_result(const holoscan::testing::TestResult& result, DeviceId device,
                          bool shared_partition, bool green, std::uint32_t requested_sm_count) {
  if (!result.ok())
    throw std::runtime_error{result.describe_failure()};
  if (result.plan().execution_partitions().size() != (shared_partition ? 1U : 3U))
    throw std::runtime_error{"unexpected example partition count"};
  const auto outputs = result.snapshots<Tensor>("out");
  if (outputs.size() != 2U)
    throw std::runtime_error{"expected two complete output records"};
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    std::vector<std::byte> expected(4U, std::byte{static_cast<std::uint8_t>(2U * index + 1U)});
    expected.insert(expected.end(), 4U, std::byte{static_cast<std::uint8_t>(2U * index + 2U)});
    if (outputs[index].bytes() != expected ||
        outputs[index].source_device().device_id != device.value)
      throw std::runtime_error{"device output mismatch"};
  }
  if (green) {
    const auto requests = result.plan().cuda_green_context_requests();
    if (requests.size() != 1U)
      throw std::runtime_error{"expected one Green Context request"};
    const auto& accepted = requests.front().accepted;
    if (!accepted || accepted->device != device || accepted->sm_count != requested_sm_count)
      throw std::runtime_error{
          "Green Context request was not accepted with the requested placement"};
  }
}
}  // namespace

int main(int argc, char** argv) {
  using namespace holoscan;
  using namespace holoscan::testing;
  using namespace std::chrono_literals;
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: operator_testing_device DEVICE_ID "
                 "[--green-context|--green-context-shared-partition]\n";
    return 1;
  }
  const std::string_view device_text{argv[1]};
  int ordinal{};
  const auto parsed =
      std::from_chars(device_text.data(), device_text.data() + device_text.size(), ordinal);
  const bool green = argc == 3;
  const bool shared_partition =
      green && std::string_view(argv[2]) == "--green-context-shared-partition";
  const bool parse_failed = parsed.ec != std::errc();
  const bool valid_option =
      !green || shared_partition || std::string_view(argv[2]) == "--green-context";
  if (parse_failed || parsed.ptr != device_text.data() + device_text.size() || ordinal < 0 ||
      !valid_option) {
    std::cerr << "invalid device or deployment option\n";
    return 1;
  }
  int count{};
  if (cudaGetDeviceCount(&count) != cudaSuccess || ordinal >= count) {
    std::cout << "SKIP: requested CUDA device unavailable\n";
    return 77;
  }
  const DeviceId device{ordinal};
  try {
    const auto selected = cudaSetDevice(ordinal);
    if (selected != cudaSuccess)
      throw std::runtime_error{cudaGetErrorString(selected)};
    CompileOptions options;
    options.deployment.bind_thread_placement("operator", ThreadPlacement{});
    std::uint32_t requested_sm_count{};
    if (green) {
      const auto configured = configure_green_context(options, device, ordinal, shared_partition);
      if (!configured) {
        return 77;
      }
      requested_sm_count = *configured;
    }
    auto h = create_operator_test<DeviceConcatOp>();
    h.push_frame(frame_at(100ms, input("left", device_input(1)), input("right", device_input(2))));
    h.push_frame(frame_at(200ms, input("left", device_input(3)), input("right", device_input(4))));
    h.input_source_device("left", device);
    h.input_source_device("right", device);
    h.output_device("out", device);
    h.capture_snapshot<Tensor>("out", 2U, 8U);
    h.set_compile_options(options);
    h.configure_graph([shared_partition](auto& f) {
      if (shared_partition) {
        // Keep every GPU participant in the dedicated group so no residual work is required.
        auto partition = f.graph().partition("operator");
        f.add_sources_to(partition);
        partition.add(f.tested_operator());
        f.add_collectors_to(partition);
        return;
      }
      auto sources = f.graph().partition("inputs");
      f.add_sources_to(sources);
      f.graph().partition("operator").add(f.tested_operator());
      auto collectors = f.graph().partition("outputs");
      f.add_collectors_to(collectors);
    });
    const auto result = h.run();
    verify_device_result(result, device, shared_partition, green, requested_sm_count);
    std::cout << "grouped CUDA inputs verified on device " << ordinal
              << (green ? " with Green Context" : " with ordinary deployment")
              << (shared_partition ? " in one shared partition" : " across three partitions")
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
