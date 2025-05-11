# - Find Eigen3
# Find the Eigen3 libraries
#
# This module defines the following variables:
#   EIGEN3_FOUND -True if EIGEN_INCLUDE_DIR are found
#   EIGEN3_INCLUDE_DIR - where to find Eigen.h.
#   EIGEN3_INCLUDE_DIRS - set when EIGEN_INCLUDE_DIR found

include(FindPackageHandleStandardArgs)

IF (MSVC)
SET( EIGEN3_DIRS "C:/Program Files (x86)" CACHE PATH "Directory containing Eigen3 directories" )
ELSE ()
SET( EIGEN3_DIRS "/usr/local" CACHE PATH "Directory containing Eigen3 directories" )
ENDIF (MSVC)

# Find include dir
find_path(EIGEN3_INCLUDE_DIR "Eigen/Core"
    PATHS ${EIGEN3_DIRS}/include
    PATH_SUFFIXES  eigen3
    DOC "The path to the directory that contains Eigen.h")

find_package_handle_standard_args(Eigen3 DEFAULT_MSG
    EIGEN3_INCLUDE_DIR)

if(EIGEN3_FOUND)
    set(EIGEN3_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
else(EIGEN3_FOUND)
    message(FATAL_ERROR "Could not locate the directory that contains Eigen.h ...")
endif()
