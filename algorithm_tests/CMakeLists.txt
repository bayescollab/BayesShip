add_executable(KDETests "KDETests.cpp")
add_executable(threadPoolTests "threadPoolTests.cpp")

target_link_libraries(KDETests PUBLIC bayesship)
target_link_libraries(threadPoolTests PUBLIC bayesship)

target_link_libraries( KDETests PUBLIC "${LIBS}")
target_link_libraries( threadPoolTests PUBLIC "${LIBS}")


target_include_directories( KDETests PUBLIC "${INCLUDE_DIRS}")
target_include_directories( threadPoolTests PUBLIC "${INCLUDE_DIRS}")


file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/algorithm_tests/python/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/algorithm_tests/plots/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/algorithm_tests/data/")

file(GLOB pythonScripts "python/*.py")
file(GLOB jupyterNbs "python/*.ipynb")
file(COPY ${pythonScripts}
     DESTINATION "${CMAKE_BINARY_DIR}/algorithm_tests/python/")
file(COPY ${jupyterNbs}
     DESTINATION "${CMAKE_BINARY_DIR}/algorithm_tests/python/")

