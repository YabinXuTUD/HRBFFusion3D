###############################################################################
# Find ORBSLAM2
#
# This sets the following variables:
# ORBSLAM2_FOUND - True if ORBSLAM2 was found.
# ORBSLAM2_INCLUDE_DIRS - Directories containing the ORBSLAM2 include files.
# ORBSLAM2_LIBRARIES - Libraries needed to use ORBSLAM2.

find_path(ORBSLAM2_INCLUDE_DIR System.h
          PATHS
            ${CMAKE_CURRENT_SOURCE_DIR}/../../Core/src/ORB_SLAM2_m/include
            ${CMAKE_CURRENT_SOURCE_DIR}/ORB_SLAM2_m/include
          PATH_SUFFIXES ORB_SLAM2
)

find_library(ORBSLAM2_LIBRARY
             NAMES libORB_SLAM2_m.so
             PATHS
               ${CMAKE_CURRENT_SOURCE_DIR}/../../Core/src/ORB_SLAM2_m/lib
               ${CMAKE_CURRENT_SOURCE_DIR}/ORB_SLAM2_m/lib
             PATH_SUFFIXES ${ORBSLAM2_PATH_SUFFIXES}
)

set(ORBSLAM2_INCLUDE_DIR ${ORBSLAM2_INCLUDE_DIR})
set(ORBSLAM2_LIBRARY ${ORBSLAM2_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ORBSLAM2 DEFAULT_MSG ORBSLAM2_LIBRARY ORBSLAM2_INCLUDE_DIR)

if(NOT WIN32)
  mark_as_advanced(EFUSION_LIBRARY EFUSION_INCLUDE_DIR)
endif()
