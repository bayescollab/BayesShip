
# Create list of source files looking for cpp files
file(GLOB SRC_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")
#file(GLOB SWIG_SRC_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.i")


# Create object files for all source files
add_library(OBJS OBJECT ${SRC_LIST})
# Set object file options
set_property(TARGET OBJS PROPERTY  POSITION_INDEPENDENT_CODE ON)
set_property(TARGET OBJS PROPERTY CXX_STANDARD 11)
# Link to the libraries defined in main CMakeList file
target_link_libraries( OBJS PUBLIC "${LIBS}")
# Include header files defined in main CMakeList file
target_include_directories( OBJS PUBLIC "${INCLUDE_DIRS}")

# Create 2 libraries files (shared and static)
add_library(bayesship SHARED $<TARGET_OBJECTS:OBJS> ${HEADER_LIST})
add_library(bayesship_static STATIC $<TARGET_OBJECTS:OBJS> ${HEADER_LIST})
# Set properties
set_property(TARGET bayesship PROPERTY CXX_STANDARD 11)
set_property(TARGET bayesship_static PROPERTY CXX_STANDARD 11)

# Link to libraries
target_link_libraries( bayesship PUBLIC "${LIBS}")
target_link_libraries( bayesship_static PUBLIC "${LIBS}")
# Include to header files
target_include_directories( bayesship PUBLIC "${INCLUDE_DIRS}")
target_include_directories( bayesship_static PUBLIC "${INCLUDE_DIRS}")

set_target_properties(bayesship_static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set_target_properties(bayesship PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set_target_properties(bayesship_static PROPERTIES OUTPUT_NAME "bayesship")


#Install
install(TARGETS bayesship DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")
install(TARGETS bayesship_static DESTINATION "${CMAKE_INSTALL_PREFIX}/lib")


#if(USE_SWIG)
#	set (UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
#	set_source_files_properties(${SWIG_SRC_LIST} PROPERTIES CPLUSPLUS ON)
#	set_source_files_properties(${SWIG_SRC_LIST} PROPERTIES INCLUDE_DIRECTORIES ${INCLUDE_DIRS})
#
#	swig_add_library(bayesshippy TYPE SHARED LANGUAGE python SOURCES ${SWIG_SRC_LIST} )
#
#	#target_link_libraries(bayesshippy PUBLIC bayesship ${LIBS})
#	#swig_link_libraries(bayesshippy bayesship ${LIBS} )
#	target_link_libraries(bayesshippy bayesship ${LIBS} )
#
#	target_include_directories(bayesshippy PUBLIC ${INCLUDE_DIRS} )
#
#	set(PYTHON_INSTALL_FILES ${CMAKE_CURRENT_BINARY_DIR}/bayesshippy.py ${CMAKE_CURRENT_BINARY_DIR}/_bayesshippy.so )
#
#	set(SETUP_PY_IN ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in)
#	set(SETUP_PY_OUT ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
#	configure_file(${SETUP_PY_IN} ${SETUP_PY_OUT})
#
#
#	set_property(TARGET bayesshippy PROPERTY SWIG_INCLUDE_DIRECTORIES "${INCLUDE_DIRS}")
#
#	add_custom_target(install-python  COMMAND python3 ${SETUP_PY_OUT} install)
#	#install(TARGETS bayesshippy DESTINATION python)
#endif()
