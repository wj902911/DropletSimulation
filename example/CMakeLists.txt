# Automatically find all .cpp files in the example directory
file(GLOB_RECURSE EXAMPLE_FILES_CPP "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB_RECURSE EXAMPLE_FILES_CU "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

# Combine all example files
set(EXAMPLE_FILES ${EXAMPLE_FILES_CPP} ${EXAMPLE_FILES_CU})

# Add an executable for each .cpp file found
foreach(EXAMPLE_FILE ${EXAMPLE_FILES})
    # Get the filename without the extension and directory
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_FILE} NAME_WE)

    # Add the executable
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_FILE})
    set_property(TARGET ${EXAMPLE_NAME} PROPERTY FOLDER "example")

    target_link_libraries(${EXAMPLE_NAME} DSLib glfw glad imgui ${GLFW_LIBRARIES} ${CUDA_LIBS} Freetype::Freetype)
    target_include_directories(${EXAMPLE_NAME} PRIVATE ${GLFW_INCLUDE_DIRS} ${FREETYPE_INCLUDE_DIRS})

    # Set the working directory for Visual Studio to the executable directory
    set_target_properties(${EXAMPLE_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")

    # Post-build command: copy the resource directory to the target's output directory
    add_custom_command(TARGET ${EXAMPLE_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${PROJECT_SOURCE_DIR}/resource" $<TARGET_FILE_DIR:${EXAMPLE_NAME}>/resource)

endforeach()