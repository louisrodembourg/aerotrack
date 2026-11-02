# C++ runtime: Python module `aerotrack_cpp`. Built by `aerotrack build-cpp` (or
# `make cpp`), which passes the Python interpreter and, on Linux + NVIDIA, the TensorRT
# library directory of the `tensorrt-cu12-libs` wheel.
#
# Without AEROTRACK_WITH_TRT only the host parts are built (letterbox, NMS, ByteTrack):
# that is what runs on a Mac, for the tests against the Python ones.
cmake_minimum_required(VERSION 3.19)
project(aerotrack_cpp LANGUAGES CXX)

option(AEROTRACK_WITH_TRT "Build the TensorRT runtime (Linux + NVIDIA only)" OFF)
set(TENSORRT_LIB_DIR "" CACHE PATH "Directory holding libnvinfer.so.10 (tensorrt_libs wheel)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Python 3 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir
                OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
                COMMAND_ERROR_IS_FATAL ANY)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(aerotrack_cpp src/bindings.cpp src/pipeline.cpp src/bytetrack.cpp)
target_compile_definitions(aerotrack_cpp PRIVATE
  AEROTRACK_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

if(AEROTRACK_WITH_TRT)
  # Headers: the apt package libnvinfer-headers-dev at the wheel's exact version (the
  # wheel ships libraries only). Libraries: the wheel's, the ones that built the engine.
  enable_language(CUDA)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)  # Tesla T4
  endif()
  find_package(CUDAToolkit REQUIRED)
  find_path(TENSORRT_INCLUDE_DIR NvInferRuntime.h
            PATHS /usr/include/x86_64-linux-gnu /usr/include/aarch64-linux-gnu REQUIRED)
  find_library(NVINFER NAMES libnvinfer.so.10 nvinfer PATHS ${TENSORRT_LIB_DIR}
               NO_DEFAULT_PATH REQUIRED)
  # The kernel in a library of its own: pybind11's compile options (LTO) are C++ only.
  add_library(aerotrack_cuda STATIC src/preprocess.cu)
  set_target_properties(aerotrack_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
  target_include_directories(aerotrack_cuda PRIVATE ${TENSORRT_INCLUDE_DIR})
  target_sources(aerotrack_cpp PRIVATE src/trt_detector.cpp)
  target_compile_definitions(aerotrack_cpp PRIVATE AEROTRACK_WITH_TRT)
  target_include_directories(aerotrack_cpp PRIVATE ${TENSORRT_INCLUDE_DIR})
  # Static CUDA runtime: the process also loads torch's own libcudart.
  target_link_libraries(aerotrack_cpp PRIVATE aerotrack_cuda ${NVINFER} CUDA::cudart_static)
  set_target_properties(aerotrack_cpp PROPERTIES BUILD_RPATH "${TENSORRT_LIB_DIR}")
  message(STATUS "TensorRT: headers ${TENSORRT_INCLUDE_DIR}, library ${NVINFER}")
endif()
