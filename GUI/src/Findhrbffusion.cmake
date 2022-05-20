###############################################################################
# Find efusion
#
# This sets the following variables:
# EFUSION_FOUND - True if EFUSION was found.
# EFUSION_INCLUDE_DIRS - Directories containing the EFUSION include files.
# EFUSION_LIBRARIES - Libraries needed to use EFUSION.

find_path(HRBFFUSION_INCLUDE_DIR HRBFFusion.h
          PATHS
            ${CMAKE_CURRENT_SOURCE_DIR}/../../Core/src
          PATH_SUFFIXES Core
)

find_library(HRBFFUSION_LIBRARY
             NAMES libhrbffusion.so
             PATHS
               ${CMAKE_CURRENT_SOURCE_DIR}/../../Core/build
               ${CMAKE_CURRENT_SOURCE_DIR}/../../Core/src/build
             PATH_SUFFIXES ${EFUSION_PATH_SUFFIXES}
)

set(HRBFFUSION_INCLUDE_DIRS ${EFUSION_INCLUDE_DIR})
set(HRBFFUSION_LIBRARIES ${EFUSION_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HRBFFUSION DEFAULT_MSG HRBFFUSION_LIBRARY HRBFFUSION_INCLUDE_DIR)

if(NOT WIN32)
  mark_as_advanced(HRBFFUSION_LIBRARY HRBFFUSION_INCLUDE_DIR)
endif()
