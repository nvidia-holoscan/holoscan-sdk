/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HOLOINFER_SRC_INCLUDE_HOLOINFER_UTILS_HPP
#define HOLOINFER_SRC_INCLUDE_HOLOINFER_UTILS_HPP

#include <sys/utsname.h>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "holoinfer_buffer.hpp"
#include "holoinfer_constants.hpp"

// Forward declaration for YAML::Node to allow use use in header files
namespace YAML {
class Node;
}

namespace holoscan {
namespace inference {

/**
 * @brief Checks Cuda result status
 * @param result Cuda result code
 */
cudaError_t check_cuda(cudaError_t result);

/**
 * Reports error with module, submodule and message, but does not throw an exception
 *
 * @param module    Module of error occurrence
 * @param submodule Submodule/Function of error occurrence with the error message (as string)
 */
int _HOLOSCAN_EXTERNAL_API_ report_error(const std::string& module, const std::string& submodule);

/**
 * Raise error with module, submodule and message
 *
 * @param module    Module of error occurrence
 * @param submodule Submodule/Function of error occurrence with the error message (as string)
 * @returns 1 (enum value of GXF_FAILURE)
 */
void _HOLOSCAN_EXTERNAL_API_ raise_error(const std::string& module, const std::string& submodule);

/**
 * @brief Setup inference I/O map for inference
 * @param pre_map Pre-processor map
 * @param inf_map Inference map
 * @param model_inputs Model inputs
 * @param model_outputs Model outputs
 * @param transmit_outputs Transmit outputs
 * @param out_tensor_names Output tensor names
 * @return InferStatus with appropriate code and message
 */
InferStatus setup_inference_io(const MultiMappings& pre_map, const MultiMappings& inf_map,
                               std::vector<std::string>& model_inputs,
                               std::vector<std::string>& model_outputs,
                               std::vector<std::string>& transmit_outputs,
                               const std::vector<std::string>& out_tensor_names);

/**
 * @brief Validate dependency map
 * @param pre_processor_map Map with model name as key, mapped to vector of input tensor names
 * @param dependency_map Map with model name as key, mapped to vector of prerequisite model names
 * @return InferStatus with appropriate code and message
 */
InferStatus validate_dependency_map(const MultiMappings& pre_processor_map,
                                    const MultiMappings& dependency_map);

/**
 * @brief Build execution plan
 * @param pre_processor_map Map with model name as key, mapped to vector of input tensor names
 * @param inference_map Map with model name as key, mapped to vector of output tensor names
 * @param execution_plan Vector of levels; each level contains models that can run in parallel.
 * @return InferStatus with appropriate code and message
 */
InferStatus build_execution_plan(const MultiMappings& pre_processor_map,
                                 const MultiMappings& inference_map,
                                 std::vector<std::vector<std::string>>& execution_plan);

/**
 * @brief Validate inference parameters
 * @param model_path_map Map with model name as key, path to model as value
 * @param pre_processor_map Map with model name as key, mapped to vector of input tensor names
 * @param inference_map Map with model name as key, mapped to vector of output tensor names
 * @param in_tensor_names Input tensor names
 * @param out_tensor_names Output tensor names
 * @return InferStatus with appropriate code and message
 */
InferStatus inference_validity_check(const Mappings& model_path_map,
                                     const MultiMappings& pre_processor_map,
                                     const MultiMappings& inference_map,
                                     std::vector<std::string>& in_tensor_names,
                                     std::vector<std::string>& out_tensor_names);

/**
 * @brief Checks for correctness of processing parameters from configuration.
 * @param processed_map Map with input tensor name as key, mapped to vector of output tensor names
 * @param in_tensor_names Input tensor names
 * @param out_tensor_names Output tensor names
 */
InferStatus processor_validity_check(const MultiMappings& processed_map,
                                     const std::vector<std::string>& in_tensor_names,
                                     const std::vector<std::string>& out_tensor_names);

/**
 * @brief Checks if the processor is arm based
 */
bool is_platform_aarch64();

#if defined(HOLOINFER_TORCH_ENABLED)
/**
 * @brief Check if PyTorch CUDA is available.
 *
 * @return true if at least one CUDA device is available, false otherwise
 */
bool is_torch_cuda_available();

/**
 * @brief Check if PyTorch CUDA is available and compatible with the current GPU.
 *
 * Returns false if:
 * - CUDA is not available (no GPU, no drivers, PyTorch not built with CUDA)
 * - GPU SM architecture is not supported by this PyTorch build ("no kernel image" error)
 *
 * This is useful to check compatibility before creating a TorchInfer context.
 *
 * @param device_id The CUDA device ID to check (default: 0)
 * @return true if CUDA is available and SM compatible, false otherwise
 */
bool is_torch_cuda_sm_compatible(int device_id = 0);
#endif

void timer_init(TimePoint& _t);

/**
 * @brief Logs the module name and (end - start) time difference (at DEBUG level).
 *
 * @returns 0 if successful.
 */
int64_t timer_check(TimePoint& start, TimePoint& end, const std::string& module);

void string_split(const std::string& line, std::vector<std::string>& tokens, char c);

/**
 * @brief Checks for correctness of input tensor dimensions.
 * @param pre_processor_map Map with model name as key, mapped to vector of input tensor names
 * @param model_input_dimensions Map with model name as key, mapped to input dimensions
 * @param dims_per_tensor Map with input tensor as key, mapped to its dimension
 * @param all_input_tensors Vector of all input tensor names
 */
InferStatus tensor_dimension_check(const MultiMappings& pre_processor_map,
                                   const DimType& model_input_dimensions,
                                   const std::map<std::string, std::vector<int>>& dims_per_tensor,
                                   const std::vector<std::string>& all_input_tensors);

using node_type = std::map<std::string, std::map<std::string, std::string>>;

// NOLINTNEXTLINE(cert-err58-cpp)
static const std::map<std::string, holoinfer_datatype> kHoloInferDataTypeMap = {
    {"kFloat32", holoinfer_datatype::h_Float32},
    {"kInt32", holoinfer_datatype::h_Int32},
    {"kInt8", holoinfer_datatype::h_Int8},
    {"kUInt8", holoinfer_datatype::h_UInt8},
    {"kInt64", holoinfer_datatype::h_Int64},
    {"kFloat16", holoinfer_datatype::h_Float16},
    {"kBool", holoinfer_datatype::h_Bool}};

InferStatus parse_yaml_node(const YAML::Node& in_config, std::vector<std::string>& names,
                            std::vector<std::vector<int64_t>>& dims,
                            std::vector<std::string>& types);

InferStatus load_yaml(const std::string& yaml_file,
                      std::map<std::string, std::string>& model_path_map,
                      std::map<std::string, std::vector<std::string>>& pre_processor_map,
                      std::map<std::string, std::vector<std::string>>& inference_map,
                      std::map<std::string, std::vector<std::string>>& batch_sizes,
                      std::vector<std::string>& in_tensor_names,
                      std::vector<std::string>& out_tensor_names,
                      std::map<std::string, size_t>& tensor_to_buffersize,
                      std::map<std::string, holoinfer_datatype>& tensor_to_datatype);

}  // namespace inference
}  // namespace holoscan
#endif /* HOLOINFER_SRC_INCLUDE_HOLOINFER_UTILS_HPP */
