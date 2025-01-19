# Automatically find all source and header files in the src directory
file(GLOB_RECURSE SRC_FILES_CPP "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
file(GLOB_RECURSE SRC_FILES_CU "${CMAKE_CURRENT_SOURCE_DIR}/*.cu" "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")

# Create an object library for all source files
add_library(IGA OBJECT ${SRC_FILES_CPP} ${SRC_FILES_CU})
set_property(TARGET IGA PROPERTY FOLDER "library")

target_link_libraries(IGA Freetype::Freetype)

# Include directories for the source files
target_include_directories(IGA PUBLIC "${PROJECT_SOURCE_DIR}/src" ${FREETYPE_INCLUDE_DIRS})

target_compile_options(IGA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

# Specify properties for CUDA files
#if(USE_CUDA)
#    set_target_properties(IGA PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#endif()