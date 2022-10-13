cmake_minimum_required (VERSION 3.11)
project(BayesShip
	VERSION 0.0.0 	
	DESCRIPTION "A Parallel Tempering Reversible Jump Markov Chain Monte Carlo library"
	LANGUAGES CXX C)

option(RUN_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build example codes" ON)
option(DOCS "DOCS" OFF)
option(IMPLEMENT_HDF5 "Implement HDF5 if available" ON)
option(IMPLEMENT_OMP "Implement OMP if available" ON)
option(ENABLE_SWIG "Use SWIG to compile python modules" ON)
option(DEBUG "set debugger options" OFF)


if(DEBUG)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Wall -std=c++11")
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O2 -Wall CPPDEFINES=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++11")
	#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 CPPDEFINES=BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS")
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install/" CACHE PATH "..." FORCE)
endif()
# The version number.
set (BAYESSHIP_VERSION_MAJOR 1)
set (BAYESSHIP_VERSION_MINOR 0)

set (BAYESSHIP_ROOT_DIRECTORY "${PROJECT_SOURCE_DIR}/")

 
# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/include/bayesship/BayesShipConfig.h.in"
  "${PROJECT_BINARY_DIR}/include/bayesship/BayesShipConfig.h"
  )

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
	#Only compile if this project is the main project
	#Just leave empty for now
endif()
file(GLOB HEADER_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/bayesship/*.h")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")

include_directories("${PROJECT_BINARY_DIR}/include/bayesship")
#Required packages
find_package(HDF5 COMPONENTS CXX)
if(HDF5_FOUND AND IMPLEMENT_HDF5 )
	add_compile_definitions(_HDF5)
else()
	message ( "HDF5 is currently required for some MCMC and RJMCMC functionality" )
endif()

find_package(nlohmann_json REQUIRED)

find_package(OpenMP)
#if(OpenMP_FOUND AND IMPLEMENT_OMP)
#	add_compile_definitions(_OPENMP)
#endif()
find_package(GSL REQUIRED)
find_package(Threads REQUIRED)
#find_package(Eigen3 REQUIRED)
find_package(Armadillo REQUIRED)
find_package(FFTW3 REQUIRED)

find_package(SWIG COMPONENTS python )


#find_package(MLPACK)
#if(MLPACK_FOUND)
#	add_compile_definitions(_MLPACK)
#endif()

set(INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include;${OpenMP_INCLUDE_DIR};${HDF5_INCLUDE_DIR};${GSL_INCLUDE_DIR};${FFTW3_INCLUDE_DIR};${ARMADILLO_INCLUDE_DIRS}")
set(LIBS "OpenMP::OpenMP_CXX;${GSL_LIBRARIES};${FFTW3_LIBRARIES};${ARMADILLO_LIBRARIES}" )

if(IMPLEMENT_HDF5)
	set(LIBS "${HDF5_LIBRARIES};${LIBS}")
	set(INCLUDE_DIRS "${HDF5_INCLUDE_DIR};${INCLUDE_DIRS}")
endif()

if(SWIG_FOUND AND ENABLE_SWIG )
	find_package(PythonLibs REQUIRED)
	find_package(Python REQUIRED)
	cmake_policy(SET CMP0078 NEW)
	#cmake_policy(SET CMP0086 NEW)
	message("Using SWIG to compile python wrappers")
	include(UseSWIG)
	set(LIBS "${PYTHON_LIBRARIES};${LIBS}")
	set(INCLUDE_DIRS "${PYTHON_INCLUDE_DIR};${INCLUDE_DIRS}")
	#set(LIBS "${SWIG_DIR};${LIBS}")
	#set(INCLUDE_DIRS "${SWIG_INCLUDE_DIR};${INCLUDE_DIRS}")
	set (USE_SWIG TRUE)
else()
	set (USE_SWIG FALSE)
endif()
if(MLPACK_FOUND)
	message("Using MLPACK")
	set(LIBS "${MLPACK_LIBRARIES};${LIBS}")
	set(INCLUDE_DIRS "${MLPACK_INCLUDE_DIR};${INCLUDE_DIRS}")
endif()



#Library
add_subdirectory("${CMAKE_SOURCE_DIR}/src")

#Executables
#add_subdirectory("${CMAKE_SOURCE_DIR}/apps")

#Python
#add_subdirectory("${CMAKE_SOURCE_DIR}/gwatpy")

if(RUN_TESTS)
	#add_subdirectory("${CMAKE_SOURCE_DIR}/unit_tests")
	add_subdirectory("${CMAKE_SOURCE_DIR}/algorithm_tests")
endif()

if(BUILD_EXAMPLES)
	#add_subdirectory("${CMAKE_SOURCE_DIR}/unit_tests")
	add_subdirectory("${CMAKE_SOURCE_DIR}/examples")
endif()


find_package(Doxygen)
if(DOCS AND DOXYGEN_FOUND)
	add_subdirectory(docs)	
endif()

if(USE_SWIG)
	add_subdirectory(python)
endif()
 
# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
#include_directories("${PROJECT_BINARY_DIR}")

install(FILES ${HEADER_LIST} DESTINATION "include/bayesship")
install(FILES "${CMAKE_BINARY_DIR}/include/bayesship/BayesShipConfig.h" DESTINATION "include/bayesship")

