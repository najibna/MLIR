cmake_minimum_required(VERSION 3.18)
project(mlir_nn LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MLIR_NN_ENABLE_CUDA "build cuda kernels" ON)

set(SRC
  src/graph.cpp
  src/passes.cpp
  src/cpu_kernels.cpp
  src/runtime.cpp
  src/main.cpp
)

add_executable(mlir_nn ${SRC})
target_include_directories(mlir_nn PRIVATE include)

if(MLIR_NN_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "cuda found, building gpu kernels")
    target_sources(mlir_nn PRIVATE cuda/kernels.cu)
    target_compile_definitions(mlir_nn PRIVATE MLIR_HAS_CUDA=1)
    set_target_properties(mlir_nn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(mlir_nn PRIVATE CUDA::cudart)
  else()
    message(STATUS "cuda not found, using cpu only")
    set(MLIR_NN_ENABLE_CUDA OFF)
  endif()
endif()

install(TARGETS mlir_nn RUNTIME DESTINATION bin)
