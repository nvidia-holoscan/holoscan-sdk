# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# Install one Linux imported shared library with a SONAME and a development namelink.
# The explicit namelink filename need not match the imported target's name.
macro(holoscan_imported_install_helper target namelink destination component)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "holoscan_imported_install_helper supports Linux shared libraries only")
    endif()
    install(IMPORTED_RUNTIME_ARTIFACTS ${target}
        LIBRARY DESTINATION "${destination}"
        COMPONENT "${component}"
    )

    # Stage relative links so file(INSTALL) retains CPack, DESTDIR, and install manifests.
    set(_holoscan_imported_target "${target}")
    set(_holoscan_imported_namelink "${namelink}")
    set(_holoscan_imported_destination "${destination}")
    string(CONFIGURE [=[
        set(_namelink_dir "@CMAKE_CURRENT_BINARY_DIR@/imported-namelinks/$<CONFIG>")
        set(_namelink "${_namelink_dir}/@_holoscan_imported_namelink@")
        file(MAKE_DIRECTORY "${_namelink_dir}")
        file(CREATE_LINK "$<TARGET_SONAME_FILE_NAME:@_holoscan_imported_target@>"
            "${_namelink}" SYMBOLIC)
        set(_destination "@_holoscan_imported_destination@")
        if(NOT IS_ABSOLUTE "${_destination}")
            string(PREPEND _destination "${CMAKE_INSTALL_PREFIX}/")
        endif()
        file(INSTALL DESTINATION "${_destination}"
            TYPE FILE FILES "${_namelink}")
    ]=] _holoscan_imported_install_code @ONLY)
    install(CODE "${_holoscan_imported_install_code}" COMPONENT "${component}")
    unset(_holoscan_imported_target)
    unset(_holoscan_imported_namelink)
    unset(_holoscan_imported_destination)
    unset(_holoscan_imported_install_code)
endmacro()
