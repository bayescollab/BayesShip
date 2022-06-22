
# Create list of source files looking for cpp files
set (UseSWIG_TARGET_NAME_PREFERENCE STANDARD)

set(bayesshipSamplerpySRC "${CMAKE_CURRENT_SOURCE_DIR}/bayesshipSamplerpy.i")
set_source_files_properties(${bayesshipSamplerpySRC} PROPERTIES CPLUSPLUS ON)
message("Include dirs: ${INCLUDE_DIRS}")
set_source_files_properties(${bayesshipSamplerpySRC} PROPERTIES INCLUDE_DIRECTORIES "${INCLUDE_DIRS}")
set_source_files_properties(${bayesshipSamplerpySRC} PROPERTIES SWIG_FLAGS "-threads")
swig_add_library(bayesshipSamplerpy TYPE SHARED LANGUAGE python SOURCES ${bayesshipSamplerpySRC} )
target_link_libraries(bayesshipSamplerpy bayesship "${LIBS}" )
target_include_directories(bayesshipSamplerpy PUBLIC "${INCLUDE_DIRS}" )
set_property(TARGET bayesshipSamplerpy PROPERTY SWIG_INCLUDE_DIRECTORIES "${INCLUDE_DIRS}")


set(dataUtilitiespySRC "${CMAKE_CURRENT_SOURCE_DIR}/dataUtilitiespy.i")
set_source_files_properties(${dataUtilitiespySRC} PROPERTIES CPLUSPLUS ON)
set_source_files_properties(${dataUtilitiespySRC} PROPERTIES INCLUDE_DIRECTORIES "${INCLUDE_DIRS}")
set_source_files_properties(${dataUtilitiespySRC} PROPERTIES SWIG_FLAGS "-threads")
swig_add_library(dataUtilitiespy TYPE SHARED LANGUAGE python SOURCES ${dataUtilitiespySRC} )
target_link_libraries(dataUtilitiespy bayesship "${LIBS}" )
target_include_directories(dataUtilitiespy PUBLIC "${INCLUDE_DIRS}" )
set_property(TARGET dataUtilitiespy PROPERTY SWIG_INCLUDE_DIRECTORIES "${INCLUDE_DIRS}")

set(utilitiespySRC "${CMAKE_CURRENT_SOURCE_DIR}/utilitiespy.i")
set_source_files_properties(${utilitiespySRC} PROPERTIES CPLUSPLUS ON)
set_source_files_properties(${utilitiespySRC} PROPERTIES INCLUDE_DIRECTORIES "${INCLUDE_DIRS}")
set_source_files_properties(${utilitiespySRC} PROPERTIES SWIG_FLAGS "-threads")
swig_add_library(utilitiespy TYPE SHARED LANGUAGE python SOURCES ${utilitiespySRC} )
target_link_libraries(utilitiespy bayesship "${LIBS}" )
target_include_directories(utilitiespy PUBLIC "${INCLUDE_DIRS}" )
set_property(TARGET utilitiespy PROPERTY SWIG_INCLUDE_DIRECTORIES "${INCLUDE_DIRS}")


configure_file("${CMAKE_CURRENT_SOURCE_DIR}/__init__.py" "${CMAKE_CURRENT_BINARY_DIR}/__init__.py" COPYONLY)

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

