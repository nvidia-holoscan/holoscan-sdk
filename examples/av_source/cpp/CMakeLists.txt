# Create example
add_executable(av_source
    av_source.cpp
)

target_link_libraries(av_source
  PRIVATE
    holoscan::core
    holoscan::ops::av
    holoscan::ops::holoviz
)

# Copy config file
add_custom_target(av_source_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/av_source.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "av_source.yaml"
  BYPRODUCTS "av_source.yaml"
)

add_dependencies(av_source av_source_yaml)

# Set the install RPATH based on the location of the Holoscan SDK libraries
# The GXF extensions are loaded by the GXF libraries - no need to include here
file(RELATIVE_PATH install_lib_relative_path ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/${HOLOSCAN_INSTALL_LIB_DIR} )
set_target_properties(av_source PROPERTIES INSTALL_RPATH "\$ORIGIN/${install_lib_relative_path}")

# Get relative folder path for the app
file(RELATIVE_PATH app_relative_dest_path ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

# Install the app
install(TARGETS av_source
  DESTINATION "${app_relative_dest_path}"
  COMPONENT holoscan-examples
)
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/av_source.yaml"
  DESTINATION ${app_relative_dest_path}
  COMPONENT holoscan-examples
)
