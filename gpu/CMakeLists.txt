cmake_minimum_required(VERSION 3.24)
project(simplepir_cutlass LANGUAGES CXX CUDA)

if(NOT DEFINED CUTLASS_PATH)
  message(FATAL_ERROR "CUTLASS_PATH must point to a CUTLASS checkout.")
endif()

add_library(simplepir_cutlass STATIC
  cutlass_bridge.cu
)

target_include_directories(simplepir_cutlass
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUTLASS_PATH}/include
    ${CUTLASS_PATH}/tools/util/include
)

set_target_properties(simplepir_cutlass PROPERTIES
  CXX_STANDARD 17
  CUDA_STANDARD 17
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
