cmake_minimum_required(VERSION 3.24)

project(klartraum_studio LANGUAGES C CXX)

option(KLARTRAUM_STUDIO_BUILD_TESTS "Build the klartraum-studio unit tests" ON)

include(FetchContent)

# klartraum (Vulkan compute-graph framework). Point FETCHCONTENT_SOURCE_DIR_KLARTRAUM
# at a local checkout to build against it instead of the pinned upstream branch.
set(KLARTRAUM_GIT_REPOSITORY "https://github.com/fortmeier/klartraum.git" CACHE STRING "klartraum repository")
# The studio needs https://github.com/fortmeier/klartraum/pull/22 (exported
# include dirs, asset root, profiling toggle); switch to develop once merged.
set(KLARTRAUM_GIT_TAG "feature/consumable-library" CACHE STRING "klartraum branch, tag or commit")
FetchContent_Declare(
  klartraum
  GIT_REPOSITORY ${KLARTRAUM_GIT_REPOSITORY}
  GIT_TAG ${KLARTRAUM_GIT_TAG}
  GIT_SUBMODULES_RECURSE TRUE
  GIT_SHALLOW TRUE
)
set(KLARTRAUM_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KLARTRAUM_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(klartraum)

# imnodes: node editor extension for Dear ImGui. It builds against the ImGui
# target that klartraum defines, so both share one ImGui context.
FetchContent_Declare(
  imnodes
  GIT_REPOSITORY https://github.com/Nelarius/imnodes.git
  GIT_TAG eb36902c892548ef94f88f51ad7e7c9c7058a71c
  # Only fetch the sources; its own CMake project expects a standalone ImGui.
  SOURCE_SUBDIR do-not-add
)
FetchContent_MakeAvailable(imnodes)
add_library(imnodes STATIC ${imnodes_SOURCE_DIR}/imnodes.cpp)
target_include_directories(imnodes PUBLIC ${imnodes_SOURCE_DIR})
target_link_libraries(imnodes PUBLIC imgui)
target_compile_features(imnodes PUBLIC cxx_std_17)

FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

# Graph model, compiler and UI; everything except main() so tests can link it.
add_library(klartraum_studio_core STATIC
  src/studio/graph_model.cpp
  src/studio/graph_serialization.cpp
  src/studio/graph_compiler.cpp
  src/studio/graph_introspection.cpp
  src/studio/graph_layout.cpp
  src/studio/studio_app.cpp
)
target_include_directories(klartraum_studio_core PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_compile_features(klartraum_studio_core PUBLIC cxx_std_20)
target_compile_definitions(klartraum_studio_core PUBLIC
  KLARTRAUM_SOURCE_DIR="${klartraum_SOURCE_DIR}"
)
target_link_libraries(klartraum_studio_core PUBLIC
  klartraum_lib
  imnodes
  nlohmann_json::nlohmann_json
)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(klartraum_studio_core PRIVATE -Wall -Wextra)
endif()

add_executable(klartraum_studio src/main.cpp)
target_link_libraries(klartraum_studio PRIVATE klartraum_studio_core)

# Renders the UI headlessly into an image, for checks without a display.
add_executable(klartraum_studio_snapshot tools/studio_snapshot.cpp)
target_link_libraries(klartraum_studio_snapshot PRIVATE klartraum_studio_core)

if(KLARTRAUM_STUDIO_BUILD_TESTS)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
  )
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(klartraum_studio_tests
    tests/test_graph_model.cpp
    tests/test_graph_serialization.cpp
    tests/test_graph_compiler.cpp
    tests/test_graph_introspection.cpp
    tests/test_graph_layout.cpp
  )
  target_link_libraries(klartraum_studio_tests PRIVATE klartraum_studio_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(klartraum_studio_tests)
endif()
