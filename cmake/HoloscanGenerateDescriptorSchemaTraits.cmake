# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Included by HoloscanGenerateSchemaTraits.cmake after the common package evidence has been
# validated and its BFBS reflection graph has been decoded. Reuse that canonical graph instead of
# producing a second undeclared JSON intermediate.
set(reflection_json "${reflected_schema}")
get_filename_component(reflection_output_directory "${HST_BFBS_FILE}" DIRECTORY)
get_filename_component(reflection_output_stem "${HST_BFBS_FILE}" NAME_WE)
# Remove the undeclared intermediate emitted by older generator revisions during an incremental
# build. Fresh generation decodes reflection only in the common temporary directory.
file(REMOVE "${reflection_output_directory}/${reflection_output_stem}.json")

string(REGEX REPLACE "^::" "" configured_root_type "${HST_ROOT_TYPE}")
string(REPLACE "::" "." configured_root_name "${configured_root_type}")
string(REGEX REPLACE "::[^:]+$" "" descriptor_root_namespace "${configured_root_type}")
set(tensor_forwarding_declaration)
if(NOT descriptor_root_namespace STREQUAL "holoscan::schema")
    string(APPEND tensor_forwarding_declaration
        "// FlatBuffers 25.12 emits an unqualified CreateTensor call from a cross-namespace "
        "object-API Pack\n"
        "// body. Keep this forwarding overload ahead of the generated root include until "
        "upstream qualifies\n"
        "// that call.\n"
        "namespace ${descriptor_root_namespace} {\n\n"
        "inline flatbuffers::Offset<holoscan::schema::Tensor> CreateTensor(\n"
        "    flatbuffers::FlatBufferBuilder& builder, const holoscan::Tensor* tensor,\n"
        "    const flatbuffers::rehasher_function_t* rehasher = nullptr) {\n"
        "  return holoscan::schema::CreateTensor(builder, tensor, rehasher);\n"
        "}\n\n"
        "}  // namespace ${descriptor_root_namespace}\n\n"
    )
endif()

string(JSON reflected_root_name GET "${reflection_json}" root_table name)
if(NOT reflected_root_name STREQUAL configured_root_name)
    message(FATAL_ERROR
        "${HST_ROOT_LOGICAL_PATH}: reflected root '${reflected_root_name}' does not match "
        "TRAITS_ROOT_TYPE '${HST_ROOT_TYPE}'"
    )
endif()

string(JSON reflected_file_identifier GET "${reflection_json}" file_ident)
if(NOT reflected_file_identifier STREQUAL "")
    message(FATAL_ERROR
        "${HST_ROOT_LOGICAL_PATH}: FlatBuffers file_identifier is unsupported by descriptor "
        "traits generation because the current verifier contract has no identifier parameter"
    )
endif()

string(JSON object_count LENGTH "${reflection_json}" objects)
set(root_object_index -1)
if(object_count GREATER 0)
    math(EXPR object_last "${object_count} - 1")
    foreach(object_index RANGE 0 ${object_last})
        string(JSON object_name GET "${reflection_json}" objects ${object_index} name)
        if(object_name STREQUAL reflected_root_name)
            set(root_object_index "${object_index}")
            break()
        endif()
    endforeach()
endif()
if(root_object_index EQUAL -1)
    message(FATAL_ERROR
        "${HST_ROOT_LOGICAL_PATH}: reflected root object '${reflected_root_name}' is missing"
    )
endif()

string(JSON root_is_struct GET
    "${reflection_json}" objects ${root_object_index} is_struct
)
if(root_is_struct)
    message(FATAL_ERROR
        "${HST_ROOT_LOGICAL_PATH}: descriptor root '${reflected_root_name}' must be a table"
    )
endif()
string(JSON root_field_count LENGTH
    "${reflection_json}" objects ${root_object_index} fields
)
if(root_field_count EQUAL 0)
    message(FATAL_ERROR
        "${HST_ROOT_LOGICAL_PATH}: descriptor root '${reflected_root_name}' has no fields"
    )
endif()

# Reflection stores object fields by name. Re-index them by explicit FlatBuffers id, validating the
# complete sequence even though descriptor mode also asks flatc to require explicit IDs.
math(EXPR root_field_last "${root_field_count} - 1")
foreach(reflected_field_index RANGE 0 ${root_field_last})
    string(JSON field_name GET "${reflection_json}"
        objects ${root_object_index} fields ${reflected_field_index} name
    )
    string(JSON field_id GET "${reflection_json}"
        objects ${root_object_index} fields ${reflected_field_index} id
    )
    if(NOT field_id MATCHES "^[0-9]+$" OR field_id GREATER_EQUAL root_field_count)
        message(FATAL_ERROR
            "${HST_ROOT_LOGICAL_PATH}:${field_name}: FlatBuffers field id '${field_id}' "
            "is missing or outside the canonical 0..${root_field_last} sequence"
        )
    endif()
    if(DEFINED descriptor_field_${field_id}_reflection_index)
        message(FATAL_ERROR
            "${HST_ROOT_LOGICAL_PATH}:${field_name}: duplicate FlatBuffers field id ${field_id}"
        )
    endif()
    set(descriptor_field_${field_id}_reflection_index "${reflected_field_index}")
endforeach()
foreach(field_id RANGE 0 ${root_field_last})
    if(NOT DEFINED descriptor_field_${field_id}_reflection_index)
        message(FATAL_ERROR
            "${HST_ROOT_LOGICAL_PATH}: missing FlatBuffers field id ${field_id}"
        )
    endif()
endforeach()

set(scalar_base_types
    Bool Byte UByte Short UShort Int UInt Long ULong Float Double UType
)
set(integer_base_types
    Bool Byte UByte Short UShort Int UInt Long ULong UType
)

function(hst_require_supported_object_alignment object_index schema_path)
    string(JSON object_is_struct GET
        "${reflection_json}" objects ${object_index} is_struct
    )
    if(NOT object_is_struct)
        return()
    endif()
    string(JSON object_min_alignment GET
        "${reflection_json}" objects ${object_index} minalign
    )
    if(object_min_alignment GREATER 8)
        message(FATAL_ERROR
            "${schema_path}: FlatBuffers struct alignment ${object_min_alignment} exceeds the "
            "descriptor metadata envelope alignment of 8"
        )
    endif()
endfunction()

function(hst_collect_nested_prepare_reset
         object_index native_prefix root_prefix schema_path ancestor_indexes
         output_prepare output_reset output_validate output_assign)
    if(object_index IN_LIST ancestor_indexes)
        message(FATAL_ERROR
            "${schema_path}: recursive tables are unsupported by descriptor traits generation; "
            "omit DESCRIPTOR_TENSOR and provide a hand-written standard codec"
        )
    endif()
    list(APPEND ancestor_indexes "${object_index}")
    string(JSON nested_field_count LENGTH
        "${reflection_json}" objects ${object_index} fields
    )
    set(nested_prepare)
    set(nested_reset)
    set(nested_validate)
    set(nested_assign)
    if(nested_field_count GREATER 0)
        math(EXPR nested_field_last "${nested_field_count} - 1")
        foreach(nested_reflection_index RANGE 0 ${nested_field_last})
            string(JSON nested_name GET "${reflection_json}"
                objects ${object_index} fields ${nested_reflection_index} name
            )
            string(JSON nested_base_type GET "${reflection_json}"
                objects ${object_index} fields ${nested_reflection_index} type base_type
            )
            string(JSON nested_element_type GET "${reflection_json}"
                objects ${object_index} fields ${nested_reflection_index} type element
            )
            string(JSON nested_type_index GET "${reflection_json}"
                objects ${object_index} fields ${nested_reflection_index} type index
            )
            string(JSON nested_deprecated GET "${reflection_json}"
                objects ${object_index} fields ${nested_reflection_index} deprecated
            )
            string(JSON nested_required GET "${reflection_json}"
                objects ${object_index} fields ${nested_reflection_index} required
            )
            set(nested_path "${schema_path}.${nested_name}")
            set(native_field "${native_prefix}${nested_name}")
            if(nested_deprecated)
                message(FATAL_ERROR
                    "${nested_path}: deprecated nested fields are unsupported by descriptor "
                    "traits generation"
                )
            elseif(nested_required AND
                   (nested_base_type STREQUAL "String"
                    OR nested_base_type STREQUAL "Vector"
                    OR nested_base_type STREQUAL "Obj"))
                message(FATAL_ERROR
                    "${nested_path}: required FlatBuffers reference fields are unsupported by "
                    "descriptor traits generation"
                )
            elseif(nested_base_type STREQUAL "Union" OR nested_base_type STREQUAL "UType")
                message(FATAL_ERROR
                    "${nested_path}: nested FlatBuffers unions are unsupported by descriptor "
                    "traits generation"
                )
            elseif(nested_base_type IN_LIST scalar_base_types)
                if(nested_base_type IN_LIST integer_base_types)
                    string(JSON nested_default GET "${reflection_json}"
                        objects ${object_index} fields ${nested_reflection_index} default_integer
                    )
                else()
                    string(JSON nested_default GET "${reflection_json}"
                        objects ${object_index} fields ${nested_reflection_index} default_real
                    )
                    if(NOT nested_default MATCHES
                       "^-?([0-9]+(\\.[0-9]*)?|\\.[0-9]+)([eE][+-]?[0-9]+)?$")
                        message(FATAL_ERROR
                            "${nested_path}: non-finite floating-point defaults are unsupported "
                            "by descriptor traits generation"
                        )
                    endif()
                endif()
                string(APPEND nested_reset
                    "      ${native_field} =\n"
                    "          static_cast<std::remove_cvref_t<decltype(${native_field})>>"
                    "(${nested_default});\n"
                )
                string(APPEND nested_assign
                    "      ${native_field} = ${root_prefix}${nested_name}();\n"
                )
            elseif(nested_base_type STREQUAL "String")
                string(APPEND nested_reset
                    "      ${native_field}.clear();\n"
                )
                string(APPEND nested_validate
                    "      if (const auto* nested_field = ${root_prefix}${nested_name}();\n"
                    "          nested_field != nullptr && "
                    "nested_field->size() > kMetadataCapacity) {\n"
                    "        return holoscan::make_unexpected(holoscan::Error{\n"
                    "            holoscan::ErrorCode::kProtocolError,\n"
                    "            \"${nested_path}: string exceeds descriptor metadata "
                    "capacity\"});\n"
                    "      }\n"
                )
                string(APPEND nested_assign
                    "      if (const auto* nested_field = ${root_prefix}${nested_name}();\n"
                    "          nested_field != nullptr) {\n"
                    "        ${native_field}.assign(nested_field->c_str(), "
                    "nested_field->size());\n"
                    "      } else {\n"
                    "        ${native_field}.clear();\n"
                    "      }\n"
                )
            elseif(nested_base_type STREQUAL "Vector")
                set(nested_vector_element_supported FALSE)
                if(nested_element_type STREQUAL "String"
                   OR nested_element_type IN_LIST scalar_base_types)
                    set(nested_vector_element_supported TRUE)
                endif()
                if(nested_element_type STREQUAL "Obj")
                    message(FATAL_ERROR
                        "${nested_path}: nested vectors of tables or structs are unsupported by "
                        "descriptor traits generation"
                    )
                elseif(nested_element_type STREQUAL "Union"
                       OR nested_element_type STREQUAL "UType"
                       OR NOT nested_vector_element_supported)
                    message(FATAL_ERROR
                        "${nested_path}: nested vector element type '${nested_element_type}' is "
                        "unsupported by descriptor traits generation"
                    )
                endif()
                string(APPEND nested_reset
                    "      ${native_field}.clear();\n"
                )
                string(APPEND nested_validate
                    "      if (const auto* nested_field = ${root_prefix}${nested_name}();\n"
                    "          nested_field != nullptr && "
                    "nested_field->size() > kMetadataCapacity) {\n"
                    "        return holoscan::make_unexpected(holoscan::Error{\n"
                    "            holoscan::ErrorCode::kProtocolError,\n"
                    "            \"${nested_path}: vector exceeds descriptor metadata "
                    "capacity\"});\n"
                    "      }\n"
                )
                if(nested_element_type STREQUAL "String")
                    string(APPEND nested_validate
                        "      if (const auto* nested_field = "
                        "${root_prefix}${nested_name}(); nested_field != nullptr) {\n"
                        "        for (flatbuffers::uoffset_t index = 0U;\n"
                        "             index < nested_field->size(); ++index) {\n"
                        "          const auto* element = nested_field->Get(index);\n"
                        "          if (element == nullptr || "
                        "element->size() > kMetadataCapacity) {\n"
                        "            return holoscan::make_unexpected(holoscan::Error{\n"
                        "                holoscan::ErrorCode::kProtocolError,\n"
                        "                \"${nested_path}: string-vector element is invalid\"});\n"
                        "          }\n"
                        "        }\n"
                        "      }\n"
                    )
                    string(APPEND nested_assign
                        "      if (const auto* nested_field = "
                        "${root_prefix}${nested_name}(); nested_field != nullptr) {\n"
                        "        ${native_field}.resize(nested_field->size());\n"
                        "        for (flatbuffers::uoffset_t index = 0U;\n"
                        "             index < nested_field->size(); ++index) {\n"
                        "          const auto* element = nested_field->Get(index);\n"
                        "          ${native_field}[index].assign(element->c_str(), "
                        "element->size());\n"
                        "        }\n"
                        "      } else {\n"
                        "        ${native_field}.clear();\n"
                        "      }\n"
                    )
                else()
                    string(APPEND nested_assign
                        "      if (const auto* nested_field = "
                        "${root_prefix}${nested_name}(); nested_field != nullptr) {\n"
                        "        ${native_field}.resize(nested_field->size());\n"
                        "        for (flatbuffers::uoffset_t index = 0U;\n"
                        "             index < nested_field->size(); ++index) {\n"
                        "          ${native_field}[index] = static_cast<typename "
                        "std::remove_cvref_t<decltype(${native_field})>::value_type>(\n"
                        "              nested_field->Get(index));\n"
                        "        }\n"
                        "      } else {\n"
                        "        ${native_field}.clear();\n"
                        "      }\n"
                    )
                endif()
            elseif(nested_base_type STREQUAL "Obj")
                if(nested_type_index LESS 0 OR nested_type_index GREATER_EQUAL object_count)
                    message(FATAL_ERROR
                        "${nested_path}: nested object type index '${nested_type_index}' is invalid"
                    )
                endif()
                string(JSON nested_object_name GET
                    "${reflection_json}" objects ${nested_type_index} name
                )
                if(nested_object_name STREQUAL "holoscan.schema.Tensor")
                    message(FATAL_ERROR
                        "${nested_path}: Tensor fields must be direct members of the descriptor "
                        "root so their canonical field ids remain available to Runtime carriage"
                    )
                endif()
                string(JSON nested_is_struct GET
                    "${reflection_json}" objects ${nested_type_index} is_struct
                )
                hst_require_supported_object_alignment("${nested_type_index}" "${nested_path}")
                string(REPLACE "." "::" nested_cpp_type "${nested_object_name}")
                if(nested_is_struct)
                    set(nested_native_type "${nested_cpp_type}")
                else()
                    set(nested_native_type "${nested_cpp_type}T")
                endif()
                if(nested_is_struct)
                    string(APPEND nested_reset
                        "      if (${native_field}) {\n"
                        "        *${native_field} = ${nested_native_type}{};\n"
                        "      }\n"
                    )
                    string(APPEND nested_assign
                        "      if (const auto* nested_field = "
                        "${root_prefix}${nested_name}(); nested_field != nullptr) {\n"
                        "        if (!${native_field}) {\n"
                        "          ${native_field} = "
                        "std::make_shared<${nested_native_type}>();\n"
                        "        }\n"
                        "        *${native_field} = *nested_field;\n"
                        "      } else {\n"
                        "        ${native_field}.reset();\n"
                        "      }\n"
                    )
                else()
                    set(nested_object_variable
                        "nested_object_${nested_type_index}"
                    )
                    hst_collect_nested_prepare_reset(
                        "${nested_type_index}"
                        "${native_field}->"
                        "${nested_object_variable}->"
                        "${nested_path}"
                        "${ancestor_indexes}"
                        nested_child_prepare
                        nested_child_reset
                        nested_child_validate
                        nested_child_assign
                    )
                    string(APPEND nested_prepare
                        "    if (${native_field}) {\n"
                        "${nested_child_prepare}"
                        "    }\n"
                    )
                    string(APPEND nested_reset
                        "      if (${native_field}) {\n"
                        "${nested_child_reset}"
                        "      }\n"
                    )
                    string(APPEND nested_validate
                        "      if (const auto* ${nested_object_variable} = "
                        "${root_prefix}${nested_name}(); "
                        "${nested_object_variable} != nullptr) {\n"
                        "${nested_child_validate}"
                        "      }\n"
                    )
                    string(APPEND nested_assign
                        "      if (const auto* ${nested_object_variable} = "
                        "${root_prefix}${nested_name}(); "
                        "${nested_object_variable} != nullptr) {\n"
                        "        if (!${native_field}) {\n"
                        "          ${native_field} = "
                        "std::make_shared<${nested_native_type}>();\n"
                        "        }\n"
                        "${nested_child_assign}"
                        "      } else {\n"
                        "        ${native_field}.reset();\n"
                        "      }\n"
                    )
                endif()
            else()
                message(FATAL_ERROR
                    "${nested_path}: nested object-API base type '${nested_base_type}' is "
                    "unsupported by descriptor traits generation"
                )
            endif()
        endforeach()
    endif()
    set("${output_prepare}" "${nested_prepare}" PARENT_SCOPE)
    set("${output_reset}" "${nested_reset}" PARENT_SCOPE)
    set("${output_validate}" "${nested_validate}" PARENT_SCOPE)
    set("${output_assign}" "${nested_assign}" PARENT_SCOPE)
endfunction()

set(tensor_count 0)
set(tensor_field_names)
set(tensor_field_ids)
set(tensor_accessor_items)
set(tensor_member_items)
set(tensor_native_assertions)
set(prepare_statements)
set(reset_statements)
set(encode_offset_statements)
set(encode_add_statements)
set(validate_statements)
set(assign_statements)

foreach(field_id RANGE 0 ${root_field_last})
    set(reflected_field_index "${descriptor_field_${field_id}_reflection_index}")
    string(JSON field_name GET "${reflection_json}"
        objects ${root_object_index} fields ${reflected_field_index} name
    )
    string(JSON field_base_type GET "${reflection_json}"
        objects ${root_object_index} fields ${reflected_field_index} type base_type
    )
    string(JSON field_element_type GET "${reflection_json}"
        objects ${root_object_index} fields ${reflected_field_index} type element
    )
    string(JSON field_type_index GET "${reflection_json}"
        objects ${root_object_index} fields ${reflected_field_index} type index
    )
    string(JSON field_deprecated GET "${reflection_json}"
        objects ${root_object_index} fields ${reflected_field_index} deprecated
    )
    string(JSON field_required GET "${reflection_json}"
        objects ${root_object_index} fields ${reflected_field_index} required
    )
    if(field_deprecated)
        message(FATAL_ERROR
            "${HST_ROOT_LOGICAL_PATH}:${field_name}: deprecated object-API fields are unsupported "
            "by descriptor traits generation; omit DESCRIPTOR_TENSOR and provide a hand-written "
            "standard codec"
        )
    endif()
    if(field_base_type STREQUAL "Union" OR field_base_type STREQUAL "UType")
        message(FATAL_ERROR
            "${HST_ROOT_LOGICAL_PATH}:${field_name}: FlatBuffers unions are unsupported by "
            "descriptor traits generation; omit DESCRIPTOR_TENSOR and provide a hand-written "
            "standard codec"
        )
    endif()
    if(field_required AND
       (field_base_type STREQUAL "String"
        OR field_base_type STREQUAL "Vector"
        OR field_base_type STREQUAL "Obj"))
        message(FATAL_ERROR
            "${HST_ROOT_LOGICAL_PATH}:${field_name}: required FlatBuffers reference fields are "
            "unsupported by descriptor traits generation"
        )
    endif()

    set(is_tensor_field FALSE)
    set(referenced_object_name)
    set(referenced_cpp_type)
    set(referenced_native_type)
    set(referenced_is_struct FALSE)
    if(field_base_type STREQUAL "Obj")
        if(field_type_index LESS 0 OR field_type_index GREATER_EQUAL object_count)
            message(FATAL_ERROR
                "${HST_ROOT_LOGICAL_PATH}:${field_name}: object type index "
                "'${field_type_index}' is invalid"
            )
        endif()
        string(JSON referenced_object_name GET
            "${reflection_json}" objects ${field_type_index} name
        )
        string(JSON referenced_is_struct GET
            "${reflection_json}" objects ${field_type_index} is_struct
        )
        hst_require_supported_object_alignment(
            "${field_type_index}" "${HST_ROOT_LOGICAL_PATH}:${field_name}"
        )
        string(REPLACE "." "::" referenced_cpp_type "${referenced_object_name}")
        if(referenced_is_struct)
            set(referenced_native_type "${referenced_cpp_type}")
        else()
            set(referenced_native_type "${referenced_cpp_type}T")
        endif()
        if(referenced_object_name STREQUAL "holoscan.schema.Tensor")
            set(is_tensor_field TRUE)
        endif()
    endif()

    if(is_tensor_field)
        list(APPEND tensor_field_names "${field_name}")
        list(APPEND tensor_field_ids "${field_id}")
        list(APPEND tensor_accessor_items
            "                                            &${HST_ROOT_TYPE}::${field_name}"
        )
        list(APPEND tensor_member_items
            "                                                            &${HST_NATIVE_TYPE}::${field_name}"
        )
        string(APPEND tensor_native_assertions
            "static_assert(std::same_as<decltype(${HST_NATIVE_TYPE}::${field_name}),\n"
            "                           std::shared_ptr<holoscan::Tensor>>,\n"
            "              \"${HST_ROOT_LOGICAL_PATH}:${field_name}: generated native Tensor "
            "member must be std::shared_ptr<holoscan::Tensor>\");\n"
        )
        string(APPEND encode_offset_statements
            "    const auto _${field_name} = tensor_offsets[${tensor_count}U];\n"
        )
        math(EXPR tensor_count "${tensor_count} + 1")
    elseif(field_base_type IN_LIST scalar_base_types)
        if(field_base_type IN_LIST integer_base_types)
            string(JSON field_default GET "${reflection_json}"
                objects ${root_object_index} fields ${reflected_field_index} default_integer
            )
        else()
            string(JSON field_default GET "${reflection_json}"
                objects ${root_object_index} fields ${reflected_field_index} default_real
            )
            if(NOT field_default MATCHES
               "^-?([0-9]+(\\.[0-9]*)?|\\.[0-9]+)([eE][+-]?[0-9]+)?$")
                message(FATAL_ERROR
                    "${HST_ROOT_LOGICAL_PATH}:${field_name}: non-finite floating-point defaults "
                    "are unsupported by descriptor traits generation"
                )
            endif()
        endif()
        string(APPEND reset_statements
            "    value.${field_name} =\n"
            "        static_cast<std::remove_cvref_t<decltype(value.${field_name})>>"
            "(${field_default});\n"
        )
        string(APPEND assign_statements
            "    value.${field_name} = root.${field_name}();\n"
        )
    elseif(field_base_type STREQUAL "String")
        string(APPEND reset_statements
            "    value.${field_name}.clear();\n"
        )
        string(APPEND encode_offset_statements
            "    const auto _${field_name} = value.${field_name}.empty()\n"
            "        ? flatbuffers::Offset<flatbuffers::String>{}\n"
            "        : builder.CreateString(value.${field_name});\n"
        )
        string(APPEND validate_statements
            "    if (const auto* field = root.${field_name}();\n"
            "        field != nullptr && field->size() > kMetadataCapacity) {\n"
            "      return holoscan::make_unexpected(holoscan::Error{\n"
            "          holoscan::ErrorCode::kProtocolError,\n"
            "          \"${HST_ROOT_LOGICAL_PATH}:${field_name}: string exceeds descriptor "
            "metadata capacity\"});\n"
            "    }\n"
        )
        string(APPEND assign_statements
            "    if (const auto* field = root.${field_name}(); field != nullptr) {\n"
            "      value.${field_name}.assign(field->c_str(), field->size());\n"
            "    } else {\n"
            "      value.${field_name}.clear();\n"
            "    }\n"
        )
    elseif(field_base_type STREQUAL "Vector")
        if(field_element_type STREQUAL "Union" OR field_element_type STREQUAL "UType")
            message(FATAL_ERROR
                "${HST_ROOT_LOGICAL_PATH}:${field_name}: vectors of unions are unsupported by "
                "descriptor traits generation"
            )
        elseif(field_element_type STREQUAL "Obj")
            message(FATAL_ERROR
                "${HST_ROOT_LOGICAL_PATH}:${field_name}: vectors of tables or structs are "
                "unsupported by descriptor traits generation; omit DESCRIPTOR_TENSOR and "
                "provide a hand-written standard codec"
            )
        elseif(NOT field_element_type STREQUAL "String"
               AND NOT field_element_type IN_LIST scalar_base_types)
            message(FATAL_ERROR
                "${HST_ROOT_LOGICAL_PATH}:${field_name}: vector element type "
                "'${field_element_type}' is unsupported by descriptor traits generation"
            )
        endif()
        string(APPEND reset_statements
            "    value.${field_name}.clear();\n"
        )
        if(field_element_type STREQUAL "String")
            set(vector_create_expression
                "builder.CreateVectorOfStrings(value.${field_name})"
            )
        elseif(field_type_index GREATER_EQUAL 0)
            if(field_element_type STREQUAL "Byte")
                set(vector_wire_type "std::int8_t")
            elseif(field_element_type STREQUAL "UByte"
                   OR field_element_type STREQUAL "Bool")
                set(vector_wire_type "std::uint8_t")
            elseif(field_element_type STREQUAL "Short")
                set(vector_wire_type "std::int16_t")
            elseif(field_element_type STREQUAL "UShort")
                set(vector_wire_type "std::uint16_t")
            elseif(field_element_type STREQUAL "Int")
                set(vector_wire_type "std::int32_t")
            elseif(field_element_type STREQUAL "UInt")
                set(vector_wire_type "std::uint32_t")
            elseif(field_element_type STREQUAL "Long")
                set(vector_wire_type "std::int64_t")
            elseif(field_element_type STREQUAL "ULong")
                set(vector_wire_type "std::uint64_t")
            else()
                message(FATAL_ERROR
                    "${HST_ROOT_LOGICAL_PATH}:${field_name}: enum vector storage "
                    "'${field_element_type}' is unsupported"
                )
            endif()
            set(vector_create_expression
                "builder.CreateVectorScalarCast<${vector_wire_type}>("
                "flatbuffers::data(value.${field_name}), value.${field_name}.size())"
            )
            string(JOIN "" vector_create_expression ${vector_create_expression})
        else()
            set(vector_create_expression "builder.CreateVector(value.${field_name})")
        endif()
        string(APPEND encode_offset_statements
            "    using _${field_name}_offset_type = decltype(${vector_create_expression});\n"
            "    const _${field_name}_offset_type _${field_name} = value.${field_name}.empty()\n"
            "        ? _${field_name}_offset_type{}\n"
            "        : ${vector_create_expression};\n"
        )
        string(APPEND validate_statements
            "    if (const auto* field = root.${field_name}();\n"
            "        field != nullptr && field->size() > kMetadataCapacity) {\n"
            "      return holoscan::make_unexpected(holoscan::Error{\n"
            "          holoscan::ErrorCode::kProtocolError,\n"
            "          \"${HST_ROOT_LOGICAL_PATH}:${field_name}: vector exceeds descriptor "
            "metadata capacity\"});\n"
            "    }\n"
        )
        if(field_element_type STREQUAL "String")
            string(APPEND validate_statements
                "    if (const auto* field = root.${field_name}(); field != nullptr) {\n"
                "      for (flatbuffers::uoffset_t index = 0U; index < field->size(); ++index) {\n"
                "        const auto* element = field->Get(index);\n"
                "        if (element == nullptr || element->size() > kMetadataCapacity) {\n"
                "          return holoscan::make_unexpected(holoscan::Error{\n"
                "              holoscan::ErrorCode::kProtocolError,\n"
                "              \"${HST_ROOT_LOGICAL_PATH}:${field_name}: string-vector element "
                "is invalid\"});\n"
                "        }\n"
                "      }\n"
                "    }\n"
            )
            string(APPEND assign_statements
                "    if (const auto* field = root.${field_name}(); field != nullptr) {\n"
                "      value.${field_name}.resize(field->size());\n"
                "      for (flatbuffers::uoffset_t index = 0U; index < field->size(); ++index) {\n"
                "        const auto* element = field->Get(index);\n"
                "        value.${field_name}[index].assign(element->c_str(), element->size());\n"
                "      }\n"
                "    } else {\n"
                "      value.${field_name}.clear();\n"
                "    }\n"
            )
        else()
            string(APPEND assign_statements
                "    if (const auto* field = root.${field_name}(); field != nullptr) {\n"
                "      value.${field_name}.resize(field->size());\n"
                "      for (flatbuffers::uoffset_t index = 0U; index < field->size(); ++index) {\n"
                "        value.${field_name}[index] = static_cast<typename "
                "std::remove_cvref_t<decltype(value.${field_name})>::value_type>"
                "(field->Get(index));\n"
                "      }\n"
                "    } else {\n"
                "      value.${field_name}.clear();\n"
                "    }\n"
            )
        endif()
    elseif(field_base_type STREQUAL "Obj")
        string(APPEND tensor_native_assertions
            "static_assert(std::same_as<decltype(${HST_NATIVE_TYPE}::${field_name}),\n"
            "                           std::shared_ptr<${referenced_native_type}>>,\n"
            "              \"${HST_ROOT_LOGICAL_PATH}:${field_name}: generated native object "
            "member must use the configured std::shared_ptr form\");\n"
        )
        if(referenced_is_struct)
            string(APPEND reset_statements
                "    if (value.${field_name}) {\n"
                "      *value.${field_name} = ${referenced_native_type}{};\n"
                "    }\n"
            )
            string(APPEND encode_offset_statements
                "    const auto* _${field_name} = value.${field_name}.get();\n"
            )
            string(APPEND assign_statements
                "    if (const auto* field = root.${field_name}(); field != nullptr) {\n"
                "      if (!value.${field_name}) {\n"
                "        value.${field_name} = std::make_shared<${referenced_native_type}>();\n"
                "      }\n"
                "      *value.${field_name} = *field;\n"
                "    } else {\n"
                "      value.${field_name}.reset();\n"
                "    }\n"
            )
        else()
            hst_collect_nested_prepare_reset(
                "${field_type_index}"
                "value.${field_name}->"
                "nested_root->"
                "${HST_ROOT_LOGICAL_PATH}:${field_name}"
                "${root_object_index}"
                nested_prepare_statements
                nested_reset_statements
                nested_validate_statements
                nested_assign_statements
            )
            string(APPEND prepare_statements
                "    if (value.${field_name}) {\n"
                "${nested_prepare_statements}"
                "    }\n"
            )
            string(APPEND reset_statements
                "    if (value.${field_name}) {\n"
                "${nested_reset_statements}"
                "    }\n"
            )
            string(APPEND validate_statements
                "    if (const auto* nested_root = root.${field_name}(); "
                "nested_root != nullptr) {\n"
                "${nested_validate_statements}"
                "    }\n"
            )
            string(APPEND encode_offset_statements
                "    const auto _${field_name} = value.${field_name}\n"
                "        ? ${referenced_cpp_type}::Pack(builder, value.${field_name}.get())\n"
                "        : flatbuffers::Offset<${referenced_cpp_type}>{};\n"
            )
            string(APPEND assign_statements
                "    if (const auto* nested_root = root.${field_name}(); "
                "nested_root != nullptr) {\n"
                "      if (!value.${field_name}) {\n"
                "        value.${field_name} = std::make_shared<${referenced_native_type}>();\n"
                "      }\n"
                "${nested_assign_statements}"
                "    } else {\n"
                "      value.${field_name}.reset();\n"
                "    }\n"
            )
        endif()
    else()
        message(FATAL_ERROR
            "${HST_ROOT_LOGICAL_PATH}:${field_name}: object-API base type "
            "'${field_base_type}' is unsupported by descriptor traits generation; omit "
            "DESCRIPTOR_TENSOR and provide a hand-written standard codec"
        )
    endif()

    if(field_base_type IN_LIST scalar_base_types AND NOT is_tensor_field)
        string(APPEND encode_add_statements
            "    root_builder.add_${field_name}(value.${field_name});\n"
        )
    else()
        string(APPEND encode_add_statements
            "    root_builder.add_${field_name}(_${field_name});\n"
        )
    endif()
endforeach()

if(tensor_count EQUAL 0)
    message(FATAL_ERROR
        "${HST_ROOT_LOGICAL_PATH}: descriptor root '${reflected_root_name}' has no field of exact "
        "type holoscan.schema.Tensor"
    )
endif()

set(tensor_field_name_entries)
set(tensor_field_id_entries)
foreach(tensor_field_name IN LISTS tensor_field_names)
    hst_escape_cpp_string(escaped_tensor_field_name "${tensor_field_name}")
    string(APPEND tensor_field_name_entries
        "    \"${escaped_tensor_field_name}\",\n"
    )
endforeach()
foreach(tensor_field_id IN LISTS tensor_field_ids)
    string(APPEND tensor_field_id_entries
        "    ${tensor_field_id}U,\n"
    )
endforeach()

string(REGEX MATCH "^(.*)::([^:]+)$" native_type_match "${HST_NATIVE_TYPE}")
if(native_type_match)
    set(tensor_selector_namespace "${CMAKE_MATCH_1}")
    set(tensor_selector_native_name "${CMAKE_MATCH_2}")
else()
    set(tensor_selector_namespace)
    set(tensor_selector_native_name "${HST_NATIVE_TYPE}")
endif()
string(REGEX REPLACE "T$" "" tensor_selector_root_name "${tensor_selector_native_name}")
set(tensor_selector_type_name "${tensor_selector_root_name}TensorFields")
set(tensor_selector_entries)
set(tensor_native_accessor_cases)
math(EXPR tensor_field_last "${tensor_count} - 1")
foreach(tensor_index RANGE 0 ${tensor_field_last})
    list(GET tensor_field_names ${tensor_index} tensor_field_name)
    list(GET tensor_field_ids ${tensor_index} tensor_field_id)
    string(APPEND tensor_selector_entries
        "  inline static constexpr holoscan::TensorFieldSelector<${HST_NATIVE_TYPE}> "
        "${tensor_field_name}{${tensor_field_id}U, ${tensor_index}U};\n"
    )
    string(APPEND tensor_native_accessor_cases
        "      case ${tensor_index}U:\n"
        "        return value.${tensor_field_name}.get();\n"
    )
endforeach()
set(tensor_selector_declaration
    "struct ${tensor_selector_type_name} {\n${tensor_selector_entries}};\n"
)
if(tensor_selector_namespace)
    set(tensor_selector_declaration
        "namespace ${tensor_selector_namespace} {\n\n"
        "${tensor_selector_declaration}\n"
        "}  // namespace ${tensor_selector_namespace}\n"
    )
endif()

string(CONCAT tensor_native_accessor
    "  [[nodiscard]] static const Tensor* tensor_at(\n"
    "      const ${HST_NATIVE_TYPE}& value, const std::size_t index) noexcept {\n"
    "    switch (index) {\n"
    "${tensor_native_accessor_cases}"
    "      default:\n"
    "        return nullptr;\n"
    "    }\n"
    "  }\n"
)
if(tensor_count EQUAL 1)
    string(APPEND tensor_native_accessor
        "\n"
        "  [[nodiscard]] static const Tensor* tensor(\n"
        "      const ${HST_NATIVE_TYPE}& value) noexcept {\n"
        "    return tensor_at(value, 0U);\n"
        "  }\n"
    )
endif()
string(JOIN ",\n" tensor_accessor_entries ${tensor_accessor_items})
string(JOIN ",\n" tensor_member_entries ${tensor_member_items})

set(descriptor_generator_format_version 4)
set(cpp_if "if")
string(CONFIGURE [=[
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Generated by holoscan_add_flatbuffer_schema(DESCRIPTOR_TENSOR). Do not edit.
 */

#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <type_traits>

#include <holoscan/core/tensor_field.hpp>
#include <holoscan/schema/flatbuffer_schema_package.hpp>

#include "tensor_generated.h"

@tensor_forwarding_declaration@
#include "@HST_BFBS_GENERATED_HEADER_ESCAPED@"
#include "@HST_ROOT_GENERATED_HEADER_ESCAPED@"

namespace @HST_PACKAGE_NAMESPACE@ {

inline constexpr std::size_t kMaxSerializedSize = @HST_MAX_SERIALIZED_SIZE@U;
inline constexpr std::size_t kMetadataCapacity = @HST_DESCRIPTOR_METADATA_CAPACITY@U;
inline constexpr std::array<std::string_view, @tensor_count@> kDescriptorTensorFields{
@tensor_field_name_entries@};
inline constexpr std::array<std::uint16_t, @tensor_count@> kDescriptorTensorFieldIds{
@tensor_field_id_entries@};
inline constexpr std::array<std::string_view, @flag_count@> kFlatcFlags{
@flag_entries@};
inline constexpr std::array<holoscan::SchemaPackageSource, @include_count@> kIncludes{
@include_entries@};

[[nodiscard]] consteval holoscan::SchemaIdentity make_identity() {
  holoscan::SchemaIdentity identity{
      .canonical_name = "@HST_CANONICAL_NAME_ESCAPED@",
      .version = @HST_SCHEMA_VERSION@U,
      .schema_sha256 = holoscan::schema::detail::schema_hex_bytes<32>(
          "@bfbs_sha256@"),
  };
  identity.id = holoscan::derive_schema_id(identity);
  return identity;
}

inline constexpr holoscan::SchemaIdentity kIdentity = make_identity();
static_assert(kIdentity.id == holoscan::derive_schema_id(kIdentity));

}  // namespace @HST_PACKAGE_NAMESPACE@

@tensor_selector_declaration@

namespace holoscan::schema {

template <>
struct FlatBufferSchemaTraits<@HST_ROOT_TYPE@> {
  using root_type = @HST_ROOT_TYPE@;
  static constexpr SchemaIdentity identity = @HST_PACKAGE_NAMESPACE@::kIdentity;
  static constexpr std::uint32_t schema_version = identity.version;
  static constexpr std::string_view root_name = identity.canonical_name;
  static constexpr std::size_t default_inline_capacity = @HST_INLINE_CAPACITY@U;
  static constexpr std::size_t default_descriptor_metadata_capacity =
      @HST_PACKAGE_NAMESPACE@::kMetadataCapacity;
};

template <>
struct DescriptorFlatBufferTensorViewTraits<@HST_ROOT_TYPE@>
    : DescriptorFlatBufferTensorViewAdapter<@HST_ROOT_TYPE@,
@tensor_accessor_entries@> {};

}  // namespace holoscan::schema

namespace holoscan {

template <>
consteval SchemaIdentity schema_identity<@HST_ROOT_TYPE@>() {
  return @HST_PACKAGE_NAMESPACE@::kIdentity;
}

template <>
consteval SchemaIdentity schema_identity<@HST_NATIVE_TYPE@>() {
  return @HST_PACKAGE_NAMESPACE@::kIdentity;
}

@tensor_native_assertions@
template <>
struct DescriptorTensorSchemaNativeTraits<@HST_NATIVE_TYPE@> {
  using root_type = @HST_ROOT_TYPE@;
  static constexpr std::size_t metadata_capacity = @HST_PACKAGE_NAMESPACE@::kMetadataCapacity;
  static constexpr std::size_t tensor_count = @tensor_count@U;
  static constexpr auto tensor_field_ids = @HST_PACKAGE_NAMESPACE@::kDescriptorTensorFieldIds;

@tensor_native_accessor@};

template <>
inline const SchemaPackageDescriptor<@HST_ROOT_TYPE@>&
schema_package<@HST_ROOT_TYPE@>() noexcept {
  static const SchemaPackageDescriptor<@HST_ROOT_TYPE@> package =
      schema::make_flatbuffer_package<@HST_ROOT_TYPE@, @HST_BINARY_SCHEMA_TYPE@>(
          schema_identity<@HST_ROOT_TYPE@>(),
          SchemaPackageEvidence{
              .flatc_build = "@FLATC_BUILD_ESCAPED@",
              .ordered_flags = @HST_PACKAGE_NAMESPACE@::kFlatcFlags,
              .root_logical_path = "@HST_ROOT_LOGICAL_PATH_ESCAPED@",
              .includes = @HST_PACKAGE_NAMESPACE@::kIncludes,
              .include_root_mapping = "@HST_INCLUDE_ROOT_MAPPING_ESCAPED@",
              .host_paths_stripped = true,
              .generator_format_version = @descriptor_generator_format_version@U,
              .package_format_version = @package_format_version@U,
              .runtime_abi = FlatBufferRuntimeABI{
                  .major = @runtime_major@U,
                  .minor = @runtime_minor@U,
                  .revision = @runtime_revision@U,
              },
          });
  return package;
}

}  // namespace holoscan

namespace @HST_PACKAGE_NAMESPACE@::detail {

struct GeneratedDescriptorCodec {
  [[nodiscard]] holoscan::expected<void, holoscan::Error> prepare(
      @HST_NATIVE_TYPE@& value) const {
@prepare_statements@    return {};
  }

  void reset(@HST_NATIVE_TYPE@& value) const noexcept {
@reset_statements@  }

  [[nodiscard]] holoscan::expected<flatbuffers::Offset<@HST_ROOT_TYPE@>, holoscan::Error>
  encode_root(
      flatbuffers::FlatBufferBuilder& builder, const @HST_NATIVE_TYPE@& value,
      const std::span<const flatbuffers::Offset<holoscan::schema::Tensor>>
          tensor_offsets) const {
    @cpp_if@ (tensor_offsets.size() != @tensor_count@U) {
      return holoscan::make_unexpected(
          holoscan::Error{holoscan::ErrorCode::kInvalidArgument,
                          "descriptor Tensor offset count does not match reflected fields"});
    }
@encode_offset_statements@    @HST_ROOT_TYPE@Builder root_builder{builder};
@encode_add_statements@    return root_builder.Finish();
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> assign_root(
      @HST_NATIVE_TYPE@& value, const @HST_ROOT_TYPE@& root) const {
@validate_statements@@assign_statements@    return {};
  }
};

}  // namespace @HST_PACKAGE_NAMESPACE@::detail

namespace holoscan::detail {

template <>
inline const RuntimePortTypeOps* runtime_port_type_ops<@HST_NATIVE_TYPE@>() noexcept {
  static const RuntimePortTypeOps ops =
      schema::make_standard_descriptor_tensor_port_type_ops<@HST_NATIVE_TYPE@,
                                                            @HST_ROOT_TYPE@,
                                                            @HST_PACKAGE_NAMESPACE@::kMetadataCapacity,
@tensor_member_entries@>(
          @HST_PACKAGE_NAMESPACE@::detail::GeneratedDescriptorCodec{});
  return &ops;
}

}  // namespace holoscan::detail
]=] generated_traits @ONLY)
