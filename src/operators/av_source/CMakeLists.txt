add_holoscan_operator(av av_source.cpp)

target_link_libraries(op_av
    PUBLIC
        holoscan::core
        CUDA::cuda_driver
    PRIVATE
        CUDA::cudart
        GXF::multimedia
)