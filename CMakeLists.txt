cmake_minimum_required(VERSION 3.24)

project(klartraum_studio LANGUAGES C CXX)

option(KLARTRAUM_STUDIO_BUILD_TESTS "Build the klartraum-studio unit tests" ON)

include(FetchContent)

# klartraum (Vulkan compute-graph framework). Point FETCHCONTENT_SOURCE_DIR_KLARTRAUM
# at a local checkout to build against it instead of the pinned upstream branch.
set(KLARTRAUM_GIT_REPOSITORY "https://github.com/fortmeier/klartraum.git" CACHE STRING "klartraum repository")
set(KLARTRAUM_GIT_TAG "develop" CACHE STRING "klartraum branch, tag or commit")
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

# imgui-node-editor: node editor extension for Dear ImGui with zoom and
# panning. It builds against the ImGui target that klartraum defines, so both
# share one ImGui context. The patch adapts it to ImGui >= 1.92, which defines
# operator*(float, ImVec2) itself.
FetchContent_Declare(
  imgui_node_editor
  GIT_REPOSITORY https://github.com/thedmd/imgui-node-editor.git
  GIT_TAG 021aa0ea4da13fed864bafb2a92d4c5205076866
  PATCH_COMMAND ${CMAKE_COMMAND} -DPATCH=${PROJECT_SOURCE_DIR}/cmake/patches/imgui-node-editor-imgui192.patch
                -P ${PROJECT_SOURCE_DIR}/cmake/apply_patch.cmake
  UPDATE_DISCONNECTED TRUE
  # Only fetch the sources; the repository has no library CMake project.
  SOURCE_SUBDIR do-not-add
)
FetchContent_MakeAvailable(imgui_node_editor)
add_library(imgui_node_editor STATIC
  ${imgui_node_editor_SOURCE_DIR}/imgui_node_editor.cpp
  ${imgui_node_editor_SOURCE_DIR}/imgui_node_editor_api.cpp
  ${imgui_node_editor_SOURCE_DIR}/imgui_canvas.cpp
  ${imgui_node_editor_SOURCE_DIR}/crude_json.cpp
)
target_include_directories(imgui_node_editor PUBLIC ${imgui_node_editor_SOURCE_DIR})
target_link_libraries(imgui_node_editor PUBLIC imgui)
target_compile_features(imgui_node_editor PUBLIC cxx_std_17)

# stb: image file loading, resizing and PNG writing (header-only).
FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG 2c980bb59875b0d32144a71867fbdebb2f77cd20
  SOURCE_SUBDIR do-not-add
)
FetchContent_MakeAvailable(stb)

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
  src/studio/image_io.cpp
  src/studio/onnx_info.cpp
  src/studio/tensor_shapes.cpp
  src/studio/cpu_numbers.cpp
  src/studio/gaussian_sources.cpp
  src/studio/graph_runner.cpp
  src/studio/preview_texture.cpp
  src/studio/studio_app.cpp
)
target_include_directories(klartraum_studio_core PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_include_directories(klartraum_studio_core SYSTEM PRIVATE ${stb_SOURCE_DIR})
target_compile_features(klartraum_studio_core PUBLIC cxx_std_20)
target_compile_definitions(klartraum_studio_core PUBLIC
  KLARTRAUM_SOURCE_DIR="${klartraum_SOURCE_DIR}"
)
target_link_libraries(klartraum_studio_core PUBLIC
  klartraum_lib
  imgui_node_editor
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
    tests/test_image_io.cpp
    tests/test_onnx_info.cpp
    tests/test_tensor_shapes.cpp
    tests/test_graph_runner.cpp
    tests/test_gaussian_sources.cpp
    tests/test_cpu_numbers.cpp
  )
  target_link_libraries(klartraum_studio_tests PRIVATE klartraum_studio_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(klartraum_studio_tests)
endif()
