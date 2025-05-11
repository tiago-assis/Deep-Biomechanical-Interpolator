#----------------------------------------------
CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0 FATAL_ERROR)

IF(POLICY CMP0053)
  CMAKE_POLICY(SET CMP0053 NEW) # CMake 3.1
ENDIF()

PROJECT(Conditions)


# Module source files.
SET(SOURCES  conditions_handler.cpp load_curve.cpp  contact_pair.cpp)

# Include the headers directory.
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

# Module header files.
SET(HEADERS ${CMAKE_SOURCE_DIR}/include/ExplicitSim/conditions/conditions.hpp
            ${CMAKE_SOURCE_DIR}/include/ExplicitSim/conditions/conditions_handler.hpp
            ${CMAKE_SOURCE_DIR}/include/ExplicitSim/conditions/load_curve.hpp
			${CMAKE_SOURCE_DIR}/include/ExplicitSim/conditions/contact_pair.hpp)

ADD_LIBRARY(${PROJECT_NAME} OBJECT ${SOURCES} ${HEADERS} )

SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# Install the header files to include/ExplicitSim directory under CMAKE_INSTALL_PREFIX.
INSTALL( FILES ${HEADERS} DESTINATION include/ExplicitSim/conditions )  
