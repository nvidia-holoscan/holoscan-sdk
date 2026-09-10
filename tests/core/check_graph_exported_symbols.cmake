# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# REQ: GRAPH-001

if(NOT DEFINED HOLOSCAN_CORE_LIBRARY OR NOT EXISTS "${HOLOSCAN_CORE_LIBRARY}")
    message(FATAL_ERROR "HOLOSCAN_CORE_LIBRARY must name an existing Core library")
endif()

get_filename_component(core_name "${HOLOSCAN_CORE_LIBRARY}" NAME)
set(symbol_command)
if(core_name MATCHES "\\.(dll|lib)$" AND
   DEFINED HOLOSCAN_DUMPBIN_EXECUTABLE AND EXISTS "${HOLOSCAN_DUMPBIN_EXECUTABLE}")
    if(core_name MATCHES "\\.dll$")
        set(symbol_command "${HOLOSCAN_DUMPBIN_EXECUTABLE}" /NOLOGO /EXPORTS
            "${HOLOSCAN_CORE_LIBRARY}")
    else()
        set(symbol_command "${HOLOSCAN_DUMPBIN_EXECUTABLE}" /NOLOGO /LINKERMEMBER:1
            "${HOLOSCAN_CORE_LIBRARY}")
    endif()
elseif(core_name MATCHES "\\.(so([.][0-9]+)*)$" AND
       DEFINED HOLOSCAN_NM_EXECUTABLE AND EXISTS "${HOLOSCAN_NM_EXECUTABLE}")
    set(symbol_command "${HOLOSCAN_NM_EXECUTABLE}" -D -C --defined-only
        "${HOLOSCAN_CORE_LIBRARY}")
elseif(core_name MATCHES "\\.a$" AND
       DEFINED HOLOSCAN_READELF_EXECUTABLE AND EXISTS "${HOLOSCAN_READELF_EXECUTABLE}")
    set(symbol_command "${HOLOSCAN_READELF_EXECUTABLE}" --wide --syms
        "${HOLOSCAN_CORE_LIBRARY}")
    set(filter_static_visibility TRUE)
else()
    message("GRAPH_SYMBOL_SCAN_UNSUPPORTED: no supported exported-symbol inspector for ${core_name}")
    return()
endif()

execute_process(
    COMMAND ${symbol_command}
    RESULT_VARIABLE symbol_result
    OUTPUT_VARIABLE exported_symbols
    ERROR_VARIABLE symbol_error
)
if(NOT symbol_result EQUAL 0)
    message(FATAL_ERROR "Unable to inspect exported Core symbols: ${symbol_error}")
endif()

# A static archive can contain global symbols with hidden ELF visibility. Those
# implementation symbols are not exported by a shared Core and are not part of
# the public package boundary. Keep only defined GLOBAL/WEAK DEFAULT entries;
# unlike nm --extern-only this preserves the visibility distinction.
if(filter_static_visibility)
    string(REGEX MATCHALL
        "[^\n]*[ \t](GLOBAL|WEAK)[ \t]+DEFAULT[ \t]+(ABS|COM|[0-9]+)[ \t]+[^\n]+"
        exported_symbols
        "${exported_symbols}"
    )
endif()

string(REGEX MATCH
    "(holoscan::(Application|Pipeline)([^A-Za-z0-9_]|$)|(Application|Pipeline)@holoscan@@)"
    forbidden_root_symbol
    "${exported_symbols}"
)
if(forbidden_root_symbol)
    message(FATAL_ERROR "Forbidden Application/Pipeline symbol exported by Core")
endif()

string(REGEX MATCH
    "(PartitionAlgorithm|PartitionPlanner|partition_of|set_partition_algorithm|source_chain|serial_baseline)"
    forbidden_partition_symbol
    "${exported_symbols}"
)
if(forbidden_partition_symbol)
    message(FATAL_ERROR "Forbidden public partition-selection symbol exported by Core")
endif()
