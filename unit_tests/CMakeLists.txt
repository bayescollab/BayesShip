file(GLOB SRC_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/unit_tests/*.cpp")
enable_testing()
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)
if(GTEST_FOUND)
	include_directories(${GTEST_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)
	add_executable(
		bayesship_test
		${SRC_LIST}
	)
	target_link_libraries(
		bayesship_test
		Threads::Threads
		GSL::gsl
		bayesship
		${GTEST_BOTH_LIBRARIES}
	)
	
	include(GoogleTest)
	gtest_discover_tests(ptrjmcmc_test)
	#install(TARGETS ptrjmcmc_test DESTINATION ${CMAKE_BINARY_DIR}/bin)
else()
	message("GoogleTests is required to run tests for this project!")
endif()
