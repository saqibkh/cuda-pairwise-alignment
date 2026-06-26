# =============================================================================
#  CMakeLists.txt
# -----------------------------------------------------------------------------
#  A portable, generator-agnostic build for this project. CMake is the second
#  of the THREE build paths we provide (the others: the Visual Studio .sln, and
#  the bare nvcc scripts in scripts/). Use whichever suits you; they all compile
#  the exact same sources.
#
#  WHY OFFER CMAKE AS WELL AS A .sln?
#    * It is the de-facto standard for cross-platform C++/CUDA, and the way most
#      real CUDA projects are built today.
#    * From ONE CMakeLists, CMake can GENERATE the Visual Studio solution for you
#      (`-G "Visual Studio 18 2026"`), or build with Ninja/Makefiles on Linux.
#    * It treats CUDA as a first-class language, so you never hand-write nvcc
#      command lines.
#
#  QUICK START (from the project root):
#     cmake -S . -B build -G "Visual Studio 18 2026" -A x64
#     cmake --build build --config Release
#     .\build\Release\demo.exe          # the showcase
#     .\build\Release\verify.exe        # the test suite
#
#  Or, with Ninja (faster, single-config):
#     cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#     cmake --build build
# =============================================================================

# 3.20 is the first CMake that fully supports CMAKE_CUDA_ARCHITECTURES the modern
# way. We require it as a sensible floor.
cmake_minimum_required(VERSION 3.20)

# Declaring CUDA as a project LANGUAGE (alongside C++) is what makes CMake invoke
# nvcc for .cu files automatically and link the CUDA runtime for us. No manual
# find_package or library wiring needed for the basics.
project(CudaPairwiseAlignment LANGUAGES CXX CUDA)

# ---- Language standards -----------------------------------------------------
# C++14 is plenty for this code (lambdas, range-for, etc.). We set it for BOTH
# the host C++ and the CUDA host side so they match.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- Target GPU architecture ------------------------------------------------
# 75 = compute capability 7.5 = Turing = RTX 2080 / 2070 / 1660 Ti / Tesla T4.
# Override on the command line for a different GPU, e.g.:
#     cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86   (Ampere, RTX 30xx)
#     cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89   (Ada,     RTX 40xx)
# You can also list several: "75;86;89" to build a "fat binary" that runs on all.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Every target can find the project headers in include/.
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# The shared core: the host helpers + CPU reference + both GPU implementations.
# Both executables below link these exact same sources.
set(CORE_SOURCES
    src/alignment_common.cpp
    src/cpu_reference.cpp
    src/gpu_wavefront.cu
    src/gpu_batched.cu
)

# ---- Executable 1: the demo / showcase --------------------------------------
add_executable(demo
    demo/demo_main.cpp
    ${CORE_SOURCES}
)

# ---- Executable 2: the correctness test harness -----------------------------
# Separate target because it has its own main(); a build cannot contain two.
add_executable(verify
    tests/verify_main.cpp
    ${CORE_SOURCES}
)

# Friendly note printed at configure time so the next steps are obvious.
message(STATUS "CudaPairwiseAlignment: targeting CUDA arch(s) = ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build, then run:  demo  (the tour)  and  verify  (the tests)")
