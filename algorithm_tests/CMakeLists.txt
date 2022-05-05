add_executable(standardTests "standardTests.cpp")
add_executable(KDETests "KDETests.cpp")
add_executable(threadPoolTests "threadPoolTests.cpp")
#set_target_properties(standardTests PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/algorithm_tests")

target_link_libraries(standardTests PUBLIC bayesship)
target_link_libraries(KDETests PUBLIC bayesship)
target_link_libraries(threadPoolTests PUBLIC bayesship)

target_link_libraries( standardTests PUBLIC "${LIBS}")
target_link_libraries( KDETests PUBLIC "${LIBS}")
target_link_libraries( threadPoolTests PUBLIC "${LIBS}")

#target_link_libraries( standardTests PUBLIC "${HDF5_LIBRARIES}" OpenMP::OpenMP_CXX  "${GSL_LIBRARIES}")
#target_link_libraries( KDETests PUBLIC "${HDF5_LIBRARIES}" OpenMP::OpenMP_CXX  "${GSL_LIBRARIES}")
#target_link_libraries( threadPoolTests PUBLIC "${HDF5_LIBRARIES}" OpenMP::OpenMP_CXX  "${GSL_LIBRARIES}")

target_include_directories( standardTests PUBLIC "${INCLUDE_DIRS}")
target_include_directories( KDETests PUBLIC "${INCLUDE_DIRS}")
target_include_directories( threadPoolTests PUBLIC "${INCLUDE_DIRS}")

#target_include_directories( standardTests PUBLIC "${PROJECT_SOURCE_DIR}/include")
#target_include_directories( KDETests PUBLIC "${PROJECT_SOURCE_DIR}/include")
#target_include_directories( threadPoolTests PUBLIC "${PROJECT_SOURCE_DIR}/include")
#target_include_directories( standardTests PUBLIC "${OpenMP_INCLUDE_DIR}" "${FFTW3_INCLUDE_DIR}" "${GSL_INCLUDE_DIR}")
#target_include_directories( KDETests PUBLIC "${OpenMP_INCLUDE_DIR}" "${FFTW3_INCLUDE_DIR}" "${GSL_INCLUDE_DIR}")
#target_include_directories( threadPoolTests PUBLIC "${OpenMP_INCLUDE_DIR}" "${FFTW3_INCLUDE_DIR}" "${GSL_INCLUDE_DIR}")

#if(HDF5_Found)
#	target_link_libraries(standardTests PUBLIC "${FFTW3_LIBRARIES}" )
#	target_include_directories( standardTests PUBLIC "${HDF5_INCLUDE_DIR}" )
#	target_link_libraries(KDETests PUBLIC "${FFTW3_LIBRARIES}" )
#	target_include_directories( KDETests PUBLIC "${HDF5_INCLUDE_DIR}" )
#	target_link_libraries(threadPoolTests PUBLIC "${FFTW3_LIBRARIES}" )
#	target_include_directories( threadPoolTests PUBLIC "${HDF5_INCLUDE_DIR}" )
#endif()

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/algorithm_tests/python/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/algorithm_tests/plots/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/algorithm_tests/data/")

file(GLOB pythonScripts "python/*.py")
file(GLOB jupyterNbs "python/*.ipynb")
file(COPY ${pythonScripts}
     DESTINATION "${CMAKE_BINARY_DIR}/algorithm_tests/python/")
file(COPY ${jupyterNbs}
     DESTINATION "${CMAKE_BINARY_DIR}/algorithm_tests/python/")

#install(TARGETS standardTests DESTINATION algorithm_tests)
