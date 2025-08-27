# Documentation generation with Doxygen and Sphinx

# Find Doxygen
find_package(Doxygen QUIET)

if(DOXYGEN_FOUND)
    # Configure Doxyfile
    set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR})
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/api)
    
    configure_file(
        ${PROJECT_SOURCE_DIR}/Doxyfile
        ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        @ONLY
    )
    
    # Create documentation target
    add_custom_target(docs
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM
    )
    
    # Install generated documentation
    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/api/html/
        DESTINATION ${CMAKE_INSTALL_DOCDIR}/api
        COMPONENT Documentation
        OPTIONAL
    )
    
    message(STATUS "Doxygen found: documentation target 'docs' available")
else()
    message(STATUS "Doxygen not found: documentation will not be generated")
endif()

# Find Sphinx for additional documentation
find_package(Sphinx QUIET)
if(Sphinx_FOUND)
    # Configure Sphinx
    set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
    set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
    
    # Create Sphinx target
    add_custom_target(sphinx-docs
        COMMAND ${SPHINX_EXECUTABLE} -b html ${SPHINX_SOURCE} ${SPHINX_BUILD}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating documentation with Sphinx"
        VERBATIM
    )
    
    message(STATUS "Sphinx found: 'sphinx-docs' target available")
endif()

# Install markdown documentation files
install(FILES
    ${PROJECT_SOURCE_DIR}/README.md
    DESTINATION ${CMAKE_INSTALL_DOCDIR}
    COMPONENT Documentation
)

# Install other documentation if it exists
if(EXISTS ${PROJECT_SOURCE_DIR}/docs)
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/docs/
        DESTINATION ${CMAKE_INSTALL_DOCDIR}
        COMPONENT Documentation
        FILES_MATCHING 
        PATTERN "*.md"
        PATTERN "*.rst"
        PATTERN "*.txt"
        PATTERN "images/*"
    )
endif()