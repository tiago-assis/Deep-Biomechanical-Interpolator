# Builing documentation option. Set ON by default.
OPTION(BUILD_DOC "Build documentation" ON)

# Check if Doxygen is installed in the OS.
FIND_PACKAGE(Doxygen)
IF (DOXYGEN_FOUND AND DOXYGEN_DOT_FOUND)
    # Set input and output files.
    SET(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/explicitsim_doxyfile.in)
    SET(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/explicitsim_doxyfile)

    # Request to configure the Doxygen documentation file.
    CONFIGURE_FILE(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    
    # Inform user how to build the Doxygen documentation.
    MESSAGE(" ")
    MESSAGE("==============================================================================")
    MESSAGE("==   You have selected to build the doxygen documentation of ExplicitSim.   ==")
    MESSAGE("==   The required dependencies where found in your Operating System.        ==")
    MESSAGE("==                                                                          ==")
    MESSAGE("==   To build the documentation of ExplicitSim execute:                     ==")
    MESSAGE("==                                                                          ==")
    MESSAGE("==   make doc                                                               ==")
    MESSAGE("==                                                                          ==")
    MESSAGE("==   after completing the configuration and generation of the CMake build.  ==")
    MESSAGE("==============================================================================")
    MESSAGE(" ")
    
    # Add custom target "doc" for documentation compilation.
    ADD_CUSTOM_TARGET(doc COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
                          WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                          COMMENT "Generating the ExplicitSim Doxygen documentation..."
                          VERBATIM )
ELSE (DOXYGEN_FOUND AND DOXYGEN_DOT_FOUND)
  MESSAGE("Doxygen and Graphviz' dot need to be installed to generate the ExplicitSim Doxygen documentation.")
ENDIF (DOXYGEN_FOUND AND DOXYGEN_DOT_FOUND)
