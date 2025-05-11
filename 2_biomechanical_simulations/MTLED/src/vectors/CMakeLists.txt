#----------------------------------------------
CMAKE_MINIMUM_REQUIRED(VERSION 3.0 FATAL_ERROR)
IF(POLICY CMP0025)
  CMAKE_POLICY(SET CMP0025 NEW) # CMake 3.0
ENDIF()
IF(POLICY CMP0053)
  CMAKE_POLICY(SET CMP0053 NEW) # CMake 3.1
ENDIF()
PROJECT(Vectors)


# Include the headers directory.
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)

# Library header files.
SET(HEADERS ${CMAKE_SOURCE_DIR}/include/ExplicitSim/vectors/vectors.hpp
            ${CMAKE_SOURCE_DIR}/include/ExplicitSim/vectors/vec3.hpp
            ${CMAKE_SOURCE_DIR}/include/ExplicitSim/vectors/vec3.tpp)

ADD_LIBRARY( ${PROJECT_NAME} OBJECT ${SOURCES} ${HEADERS} )

SET_TARGET_PROPERTIES( ${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# Install the header files to include/ExplicitSim directory under CMAKE_INSTALL_PREFIX.
INSTALL( FILES ${HEADERS} DESTINATION include/ExplicitSim/vectors )
