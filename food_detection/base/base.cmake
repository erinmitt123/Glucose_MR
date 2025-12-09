# common dependencies for all demo projects
set(BASE_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/d3d_common.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/graphicsplugin_d3d11.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/graphicsplugin_d3d12.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/graphicsplugin_factory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/graphicsplugin_opengl.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/graphicsplugin_opengles.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/graphicsplugin_vulkan.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/graphicsplugin_metal.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/logger.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/platformplugin_android.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/platformplugin_factory.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/platformplugin_posix.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/platformplugin_win32.cpp
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils/pch.cpp
    ${CMAKE_CURRENT_LIST_DIR}/main.cpp
    ${CMAKE_CURRENT_LIST_DIR}/openxr_program.cpp
)
set(VULKAN_SHADERS
    ${CMAKE_CURRENT_LIST_DIR}/vulkan_shaders/frag.glsl
    ${CMAKE_CURRENT_LIST_DIR}/vulkan_shaders/vert.glsl
)
set(SECUREMR_UTILS_SRCS "")

if (USE_SECURE_MR_UTILS)
    list(APPEND SECUREMR_UTILS_SRCS
        ${CMAKE_CURRENT_LIST_DIR}/securemr_utils/pipeline.cpp
        ${CMAKE_CURRENT_LIST_DIR}/securemr_utils/rendercommand.cpp
        ${CMAKE_CURRENT_LIST_DIR}/securemr_utils/session.cpp
        ${CMAKE_CURRENT_LIST_DIR}/securemr_utils/tensor.cpp
    )
endif()

# Dependency: OpenXR
include("${CMAKE_CURRENT_LIST_DIR}/../external/openxr/openxr.cmake")
# Dependency: Vulkan
find_package(Vulkan)
if(Vulkan_FOUND)
    set(XR_USE_GRAPHICS_API_VULKAN TRUE)
    add_definitions(-DXR_USE_GRAPHICS_API_VULKAN)
    message(STATUS "Enabling Vulkan support")
elseif(BUILD_ALL_EXTENSIONS)
    message(FATAL_ERROR "Vulkan headers not found")
endif()

add_library(
    ${PROJECT_NAME} MODULE
    ${BASE_SRCS}
    ${SAMPLE_SRCS}
    ${VULKAN_SHADERS}
    ${SECUREMR_UTILS_SRCS}
    ${ANDROID_NDK}/sources/android/native_app_glue/android_native_app_glue.c
)

target_link_libraries(
    ${PROJECT_NAME} PRIVATE
    android
    log
    OpenXR::openxr_loader
    ${Vulkan_LIBRARY}
)
target_include_directories(${PROJECT_NAME} PRIVATE
    "${ANDROID_NDK}/sources/android/native_app_glue"
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/vulkan_shaders
    ${CMAKE_CURRENT_LIST_DIR}/oxr_utils
    ${SAMPLE_DIR}
    ${CMAKE_SOURCE_DIR}
    ${Vulkan_INCLUDE_DIRS}
)

# Shader compilation for **client**
# No shader needed for SecureMR stuff
include("${CMAKE_CURRENT_LIST_DIR}/../scripts/compile_glsl.cmake")
compile_glsl(run_glsl_compiles ${VULKAN_SHADERS})
if(GLSLANG_VALIDATOR AND NOT GLSLC_COMMAND)
    target_compile_definitions(${PROJECT_NAME} PRIVATE USE_GLSLANGVALIDATOR)
endif()
add_dependencies(${PROJECT_NAME} run_glsl_compiles)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    DEFAULT_GRAPHICS_PLUGIN_VULKAN
    XR_USE_PLATFORM_ANDROID
    XR_USE_GRAPHICS_API_VULKAN)
