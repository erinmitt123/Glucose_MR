if(NOT TARGET OpenXR::openxr_loader)
add_library(OpenXR::openxr_loader SHARED IMPORTED)
set_target_properties(OpenXR::openxr_loader PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/lib/libopenxr_loader.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/include"
    INTERFACE_LINK_LIBRARIES ""
)
endif()
