CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0 FATAL_ERROR)

IF(POLICY CMP0053)
    CMAKE_POLICY(SET CMP0053 NEW) # CMake 3.1
ENDIF()

PROJECT(ExplicitSim VERSION 1.0.0)

# Set project's CMake modules path
SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")

# ------------------------------------------------------------------------------
# Options

OPTION(BUILD_SHARED_LIBS "Build as shared library" OFF )
OPTION(BUILD_STATIC_LIBS "Build as static library" ON )

OPTION(Boost_USE_STATIC_LIBS "Use Boost static libs" ON)

# Method to use for nearest neighbor search
SET(EXPLICITSIM_NEAREST_NEIGHBOR "CGAL" CACHE STRING
    "Method to use for nearest neighour search ('EXHAUSTIVE', 'CGAL' or 'BUCKETSEARCH')")
SET_PROPERTY(CACHE EXPLICITSIM_NEAREST_NEIGHBOR PROPERTY STRINGS EXHAUSTIVE CGAL BUCKETSEARCH)
MESSAGE(STATUS "Nearest neighbour search using: ${EXPLICITSIM_NEAREST_NEIGHBOR}")
IF(EXPLICITSIM_NEAREST_NEIGHBOR STREQUAL "EXHAUSTIVE")
    ADD_COMPILE_DEFINITIONS(EXPLICITSIM_NEAREST_NEIGHBOR_EXHAUSTIVE)
ELSEIF(EXPLICITSIM_NEAREST_NEIGHBOR STREQUAL "CGAL")
    ADD_COMPILE_DEFINITIONS(EXPLICITSIM_NEAREST_NEIGHBOR_CGAL)
ELSEIF(EXPLICITSIM_NEAREST_NEIGHBOR STREQUAL "BUCKETSEARCH")
    ADD_COMPILE_DEFINITIONS(EXPLICITSIM_NEAREST_NEIGHBOR_BUCKETSEARCH)
ENDIF()

# ------------------------------------------------------------------------------
# Compiler

# Set the version of C++
SET(CMAKE_CXX_STANDARD 14)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frounding-math -fsanitize=address -g")

# Set additional C++ flags for supported compilers.
IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Ob2")
ELSEIF(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfpmath=sse -O3 -march=native -msse2 -ffast-math -fPIC -Wall")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpmath=sse -O3 -march=native -msse2 -ffast-math -fPIC -Wall")
ELSEIF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ansi -frounding-math -fsignaling-nans -mfpmath=sse -march=native -msse2 -O3 -ffast-math -fPIC -Wall")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ansi -frounding-math -fsignaling-nans -mfpmath=sse -march=native -msse2 -O3 -ffast-math -fPIC -Wall")
ENDIF()

# ------------------------------------------------------------------------------
# Find dependencies of ExplicitSim

# CGAL
IF(EXPLICITSIM_NEAREST_NEIGHBOR MATCHES "CGAL")
    FIND_PACKAGE(CGAL REQUIRED COMPONENTS Core)
    INCLUDE(${CGAL_USE_FILE})
    MESSAGE(STATUS "CGAL: ${CGAL_VERSION} ${CGAL_INCLUDE_DIRS} ${CGAL_LIBRARIES}")
ENDIF()

# Eigen
FIND_PACKAGE(Eigen3  REQUIRED)
INCLUDE_DIRECTORIES(${EIGEN3_INCLUDE_DIRS})
MESSAGE(STATUS "Eigen: ${EIGEN3_VERSION} ${EIGEN3_INCLUDE_DIRS}")

# Boost
FIND_PACKAGE(Boost REQUIRED COMPONENTS system filesystem program_options thread)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
MESSAGE(STATUS "Boost: ${Boost_VERSION} ${Boost_INCLUDE_DIR} ${Boost_LIBRARIES}")

# TinyXML2
FIND_PACKAGE(TINYXML2 REQUIRED)
INCLUDE_DIRECTORIES(${TINYXML2_INCLUDE_DIRS})
MESSAGE(STATUS "TinyXML-2: ${TINYXML2_INCLUDE_DIRS} ${TINYXML2_LIBRARIES}")

# ------------------------------------------------------------------------------
# List of external libraries to link with ExplicitSim

LIST(APPEND LIBRARIES ${Boost_LIBRARIES} ${TINYXML2_LIBRARIES})
IF(EXPLICITSIM_NEAREST_NEIGHBOR STREQUAL "CGAL")
    LIST(APPEND LIBRARIES ${CGAL_LIBRARIES})
ENDIF()
MESSAGE(STATUS "Libraries: ${LIBRARIES}")

# ------------------------------------------------------------------------------
# Directories of ExplicitSim

# Include path to the headers of ExplicitSim.
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)

# Set the headers of the ExplicitSim library.
SET(HEADERS ${CMAKE_SOURCE_DIR}/include/ExplicitSim/explicit_sim.hpp )

ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/doc)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/approximants)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/conditions)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/elements)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/exporters)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/grid)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/integration)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/materials)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/mesh)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/mesh_io)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/models)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/options_configuration)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/sets)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/solvers)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/support_domain)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/utilities)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/vectors)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/src/contacts)
ADD_SUBDIRECTORY(${CMAKE_SOURCE_DIR}/bucketsearch)

# ------------------------------------------------------------------------------
# Compilation of the shared library.

IF(BUILD_SHARED_LIBS)
    ADD_LIBRARY(${PROJECT_NAME} SHARED ${HEADERS}
        $<TARGET_OBJECTS:Conditions>
        $<TARGET_OBJECTS:Elements>
        $<TARGET_OBJECTS:Exporters>
        $<TARGET_OBJECTS:Grid>
        $<TARGET_OBJECTS:Integration>
        $<TARGET_OBJECTS:Materials>
        $<TARGET_OBJECTS:Mesh>
        $<TARGET_OBJECTS:MeshIO>
        $<TARGET_OBJECTS:Models>
        $<TARGET_OBJECTS:OptionsConfiguration>
        $<TARGET_OBJECTS:Sets>
        $<TARGET_OBJECTS:Approximants>
        $<TARGET_OBJECTS:Solvers>
        $<TARGET_OBJECTS:SupportDomain>
        $<TARGET_OBJECTS:Utilities>
        $<TARGET_OBJECTS:Vectors>
        $<TARGET_OBJECTS:Contacts>
        $<TARGET_OBJECTS:bucketsearch>)

    # # Link with dependencies
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${LIBRARIES})

    # Set build folders.
    SET_TARGET_PROPERTIES(${PROJECT_NAME}
        PROPERTIES
        LINKER_LANGUAGE CXX
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/ExplicitSim"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/ExplicitSim"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/ExplicitSim")

    # Install the executable to 'bin' directory under CMAKE_INSTALL_PREFIX.
    INSTALL(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION lib/ExplicitSim
        LIBRARY DESTINATION lib/ExplicitSim
        RUNTIME DESTINATION bin/ExplicitSim)
ENDIF()

# ------------------------------------------------------------------------------
# Compilation of the static library.

IF(BUILD_STATIC_LIBS)
    ADD_LIBRARY(${PROJECT_NAME}_static STATIC  ${HEADERS}
        $<TARGET_OBJECTS:Conditions>
        $<TARGET_OBJECTS:Elements>
        $<TARGET_OBJECTS:Exporters>
        $<TARGET_OBJECTS:Grid>
        $<TARGET_OBJECTS:Integration>
        $<TARGET_OBJECTS:Materials>
        $<TARGET_OBJECTS:Mesh>
        $<TARGET_OBJECTS:MeshIO>
        $<TARGET_OBJECTS:Models>
        $<TARGET_OBJECTS:OptionsConfiguration>
        $<TARGET_OBJECTS:Sets>
        $<TARGET_OBJECTS:Approximants>
        $<TARGET_OBJECTS:Solvers>
        $<TARGET_OBJECTS:SupportDomain>
        $<TARGET_OBJECTS:Utilities>
        $<TARGET_OBJECTS:Vectors>
        $<TARGET_OBJECTS:Contacts>
        $<TARGET_OBJECTS:bucketsearch>)
    
    # Link with dependencies
    TARGET_LINK_LIBRARIES(${PROJECT_NAME}_static ${LIBRARIES})

    # Set build folders   
    SET_TARGET_PROPERTIES(${PROJECT_NAME}_static
        PROPERTIES
        LINKER_LANGUAGE CXX
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/ExplicitSim"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/ExplicitSim"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/ExplicitSim" )

    # Install the executable to 'bin' directory under CMAKE_INSTALL_PREFIX
    INSTALL(TARGETS ${PROJECT_NAME}_static
        ARCHIVE DESTINATION lib/ExplicitSim
        LIBRARY DESTINATION lib/ExplicitSim
        RUNTIME DESTINATION bin/ExplicitSim)
ENDIF()

# ------------------------------------------------------------------------------
# Executables

# ExplicitSim simulation executable
ADD_EXECUTABLE(IMLS_EXP  main.cpp)
IF(${BUILD_SHARED_LIBS})
    TARGET_LINK_LIBRARIES(IMLS_EXP ${PROJECT_NAME})
ELSE()
    TARGET_LINK_LIBRARIES(IMLS_EXP ${PROJECT_NAME}_static)
ENDIF()

# Development testing executable
ADD_EXECUTABLE(xps_dev  dev.cpp )
IF(${BUILD_SHARED_LIBS})
    TARGET_LINK_LIBRARIES(xps_dev ${PROJECT_NAME})
ELSE()
    TARGET_LINK_LIBRARIES(xps_dev ${PROJECT_NAME}_static)
ENDIF()

# ------------------------------------------------------------------------------
# Install

# Install the ExplicitSim simulation executable
INSTALL(TARGETS IMLS_EXP RUNTIME DESTINATION bin)

# Install header files to include/ExplicitSim directory under CMAKE_INSTALL_PREFIX
INSTALL(FILES ${HEADERS} DESTINATION include/ExplicitSim)

# Install the ExplicitSim finding module in the cmake folder under the installation directory.
#INSTALL(FILES cmake/FindExplicitSim.cmake DESTINATION cmake)
