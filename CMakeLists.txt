cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(CMAKE_VERBOSE_MAKEFILE ON)

# Add a CMake option to control CUDA inclusion
option(USE_CUDA "Build with CUDA support" ON) # Default to ON; set to OFF to disable CUDA

if(USE_CUDA)
    project(DropletSimulation LANGUAGES CXX CUDA) # Include CUDA in project setup
else()
    project(DropletSimulation LANGUAGES CXX) # Exclude CUDA from project setup
endif()

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)



# Add GLFW
set(GLFW_DIR "${PROJECT_SOURCE_DIR}/external/glfw")
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${GLFW_DIR})

# Add GLAD
add_library(glad "${PROJECT_SOURCE_DIR}/external/glad/src/glad.c")
target_include_directories(glad PUBLIC "${PROJECT_SOURCE_DIR}/external/glad/include")

# Add Dear ImGui
file(GLOB IMGUI_SOURCES "${PROJECT_SOURCE_DIR}/external/imgui/*.cpp" "${PROJECT_SOURCE_DIR}/external/imgui/backends/imgui_impl_glfw.cpp" "${PROJECT_SOURCE_DIR}/external/imgui/backends/imgui_impl_opengl3.cpp")
add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui PUBLIC "${PROJECT_SOURCE_DIR}/external/imgui" "${PROJECT_SOURCE_DIR}/external/imgui/backends")


# Conditional CUDA integration
if(USE_CUDA)
    set(CMAKE_CUDA_STANDARD 17) # Set CUDA standard
    set(CMAKE_CUDA_ARCHITECTURES 75)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --extended-lambda")
    find_package(CUDA REQUIRED) # Find CUDA package
    include_directories("${CUDA_INCLUDE_DIRS}") # Include CUDA directories
    set(CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_cusolver_LIBRARY} ${CUDA_cublas_LIBRARY})
else()
    set(CUDA_LIBS "") # Empty if CUDA is not used
endif()

set(FREETYPE_INCLUDE_DIRS "C:/pakages/include")
set(FREETYPE_LIBRARY "C:/pakages/lib/freetype.lib")
find_package(Freetype REQUIRED)

# Include resource directory copying
file(COPY "${PROJECT_SOURCE_DIR}/resource" DESTINATION "${CMAKE_BINARY_DIR}")

# Include directories
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${GLFW_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/external/glad/include")
include_directories("${PROJECT_SOURCE_DIR}/external/imgui")
include_directories("${PROJECT_SOURCE_DIR}/external")
include_directories("${PROJECT_SOURCE_DIR}/external/eigen")
include_directories("${PROJECT_SOURCE_DIR}/external/glm")
include_directories("${PROJECT_SOURCE_DIR}/include")

#link_directories(${PROJECT_SOURCE_DIR}/libs)

add_subdirectory(src)
add_subdirectory(example)
