#----------------------------------------------
CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0 FATAL_ERROR)

IF(POLICY CMP0053)
  CMAKE_POLICY(SET CMP0053 NEW) # CMake 3.1
ENDIF()

PROJECT(Contacts)


# Module source files.
SET(SOURCES  ContactPair.cpp  matrix.cpp  TriangularSurface.cpp  utils.cpp)

# Include the headers directory.
INCLUDE_DIRECTORIES(/)

# Module header files.
SET(HEADERS cdef.h ContactPair.hpp  matrix.h TriangularSurface.h  utils.h vect.h)

ADD_LIBRARY(${PROJECT_NAME} OBJECT ${SOURCES} ${HEADERS} )

SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# Install the header files to include/ExplicitSim directory under CMAKE_INSTALL_PREFIX.
INSTALL( FILES ${HEADERS} DESTINATION src/contacts )  
