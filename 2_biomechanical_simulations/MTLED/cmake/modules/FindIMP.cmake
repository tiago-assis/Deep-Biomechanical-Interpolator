# - Find IMP
# Find the IMP libraries
#
# This module defines the following variables:
#       IMP_FOUND - True if IMP_INCLUDE_DIR are IMP_FOUND
#       IMP_LIBRARY - where to find IMP library.
#       IMP_LIBRARIES - set when IMP_LIBRARY found.
#       IMP_INCLUDE_DIR - where to find IMP headers.
#       IMP_INCLUDE_DIRS - set when IMP_INCLUDE_DIR found.

INCLUDE( FindPackageHandleStandardArgs )

# Search folder where 'lib/IMP' and 'include/IMP' exist in default paths for Windows and Linux.
IF (MSVC)
SET( IMP_DIRS "C:/Program Files (x86)" CACHE PATH "Directory containing IMP directories" )
ELSE ()
SET( IMP_DIRS "/usr/local" CACHE PATH "Directory containing IMP directories" )
ENDIF (MSVC)


# SET include directory.
FIND_PATH( IMP_INCLUDE_DIR NAMES IMP/imp.hpp  PATHS ${IMP_DIRS}/include  DOC "IMP include directory." )


# Find library.
FIND_LIBRARY( IMP_LIBRARY NAMES IMP IMP_static PATHS ${IMP_DIRS}/lib/IMP )

find_package_handle_standard_args( IMP DEFAULT_MSG IMP_LIBRARY IMP_INCLUDE_DIR )

MARK_AS_ADVANCED( IMP_INCLUDE_DIR IMP_LIBRARY )

IF( IMP_FOUND )
    SET( IMP_INCLUDE_DIRS  ${IMP_INCLUDE_DIR} )
    SET( IMP_LIBRARIES  ${IMP_LIBRARY} )
ELSE( IMP_FOUND )
    MESSAGE(FATAL_ERROR "Could not locate either the 'include/IMP' directory or the 'lib/libIMP.a' | 'lib/libIMP.so' of the IMP library..." )
ENDIF()
