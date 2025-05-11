# - Find TinyXML2
#
# Find the native TinyXML2 includes and add_library
#
# TINYXML2_FOUND - True if TinyXML2 found.
# TINYXML2_INCLUDE_DIR - Where to find tinyxml2.h, etc.
# TINYXML2_INCLUDE_DIRS - Set if found tinyxml2.h, etc.
# TINYXML2_LIBRARIES - List of libraries when using TinyXML2.
#

INCLUDE( "FindPackageHandleStandardArgs" )

IF (MSVC)
SET( TINYXML2_DIRS "C:/Program Files (x86)" CACHE PATH "Directory containing TinyXML2 directories" )
ELSE ()
SET( TINYXML2_DIRS "/usr/local" CACHE PATH "Directory containing TinyXML2 directories" )
ENDIF (MSVC)


#Find header files.
FIND_PATH( TINYXML2_INCLUDE_DIR "tinyxml2.h" PATHS ${TINYXML2_DIRS}/include PATH_SUFFIXES "tinyxml2" )

#Find libraries.
FIND_LIBRARY( TINYXML2_LIBRARY NAMES "tinyxml2" PATHS ${TINYXML2_DIRS}/lib PATH_SUFFIXES "tinyxml2" )

# handle the QUIETLY and REQUIRED arguments and set TINYXML2_FOUND to TRUE
# if all listed variables are TRUE.

FIND_PACKAGE_HANDLE_STANDARD_ARGS( "TinyXML2" DEFAULT_MSG TINYXML2_INCLUDE_DIR TINYXML2_LIBRARY )

IF(TINYXML2_FOUND)
    SET( TINYXML2_INCLUDE_DIRS  ${TINYXML2_INCLUDE_DIR} )
    SET( TINYXML2_LIBRARIES  ${TINYXML2_LIBRARY} )
ELSE(TINYXML2_FOUND)
    MESSAGE( FATAL_ERROR "Could not locate TinyXML2..." )
ENDIF()