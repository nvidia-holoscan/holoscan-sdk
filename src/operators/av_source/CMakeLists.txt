add_holoscan_operator(av av_source.cpp)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
    libavcodec
    libavformat
    libavutil
    libswscale
)

target_link_libraries(op_av
    PUBLIC
        holoscan::core
        CUDA::cuda_driver
    PRIVATE
        CUDA::cudart
        GXF::multimedia
        PkgConfig::LIBAV
)