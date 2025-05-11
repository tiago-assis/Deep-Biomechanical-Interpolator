#----------------------------------------------
CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0 FATAL_ERROR)
IF(POLICY CMP0053)
  CMAKE_POLICY(SET CMP0053 NEW) # CMake 3.1
ENDIF()
PROJECT(MeshIO)


# Library source files.
SET(SOURCES abaqus_io.cpp  febio_io.cpp)

# Include the headers directory.
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)

# Library header files.
SET(HEADERS ${CMAKE_SOURCE_DIR}/include/ExplicitSim/mesh_io/mesh_io.hpp
            ${CMAKE_SOURCE_DIR}/include/ExplicitSim/mesh_io/abaqus_io.hpp
            ${CMAKE_SOURCE_DIR}/include/ExplicitSim/mesh_io/abaqus_io.tpp
            ${CMAKE_SOURCE_DIR}/include/ExplicitSim/mesh_io/febio_io.hpp)

ADD_LIBRARY( ${PROJECT_NAME} OBJECT ${SOURCES} ${HEADERS} )

SET_TARGET_PROPERTIES( ${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# Install the header files to include/ExplicitSim directory under CMAKE_INSTALL_PREFIX.
INSTALL( FILES ${HEADERS} DESTINATION include/ExplicitSim/mesh_io )
