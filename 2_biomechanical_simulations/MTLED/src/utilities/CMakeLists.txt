#----------------------------------------------
CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0 FATAL_ERROR)

IF(POLICY CMP0053)
  CMAKE_POLICY(SET CMP0053 NEW) # CMake 3.1
ENDIF()

PROJECT(Utilities)


# Module source files.
SET(SOURCES timer.cpp)

# Include the headers directory.
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)

# Module header files.
SET(HEADERS ${CMAKE_SOURCE_DIR}/include/ExplicitSim/utilities/utilities.hpp
            ${CMAKE_SOURCE_DIR}/include/ExplicitSim/utilities/logger.hpp
            ${CMAKE_SOURCE_DIR}/include/ExplicitSim/utilities/timer.hpp)

ADD_LIBRARY(${PROJECT_NAME} OBJECT ${SOURCES} ${HEADERS} )

SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# Install the header files to include/ExplicitSim directory under CMAKE_INSTALL_PREFIX.
INSTALL( FILES ${HEADERS} DESTINATION include/ExplicitSim/utilities )
