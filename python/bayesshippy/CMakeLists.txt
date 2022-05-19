
# Create list of source files looking for cpp files
file(GLOB SWIG_SRC_LIST CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.i")


set (UseSWIG_TARGET_NAME_PREFERENCE STANDARD)
set_source_files_properties(${SWIG_SRC_LIST} PROPERTIES CPLUSPLUS ON)
set_source_files_properties(${SWIG_SRC_LIST} PROPERTIES INCLUDE_DIRECTORIES ${INCLUDE_DIRS})
set_source_files_properties(${SWIG_SRC_LIST} PROPERTIES SWIG_FLAGS "-threads")

swig_add_library(bayesshipSamplerpy TYPE SHARED LANGUAGE python SOURCES ${SWIG_SRC_LIST} )

target_link_libraries(bayesshipSamplerpy bayesship ${LIBS} )

target_include_directories(bayesshipSamplerpy PUBLIC ${INCLUDE_DIRS} )

#set(PYTHON_INSTALL_FILES ${CMAKE_CURRENT_BINARY_DIR}/bayesshippy.py ${CMAKE_CURRENT_BINARY_DIR}/_bayesshippy.so )

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/__init__.py" "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" COPYONLY)

set_property(TARGET bayesshipSamplerpy PROPERTY SWIG_INCLUDE_DIRECTORIES "${INCLUDE_DIRS}")

#execute_process(COMMAND ${Python_EXECUTABLE} ${SETUP_PY_OUT} install)
#install(CODE "execute_process(COMMAND ${Python_EXECUTABLE} -m pip install ${CMAKE_CURRENT_BINARY_DIR} )" )
#add_custom_target(install-python  COMMAND python3 ${SETUP_PY_OUT} install)

#if(DEFINED ENV{VIRTUAL_ENV} OR DEFINED ENV{CONDA_PREFIX})
#	set(_pip_args)
#else()
#	set(_pip_args "--user")
#endif()
#execute_process(COMMAND ${Python_EXECUTABLE} -m pip install ${_pip_args} ${_pypkg})
#if(DEBUG)
#	execute_process(COMMAND ${Python_EXECUTABLE} -m pip install ${_pip_args} -e ${CMAKE_CURRENT_LIST_DIR})
#endif()
#install(TARGETS bayesshippy DESTINATION python)

