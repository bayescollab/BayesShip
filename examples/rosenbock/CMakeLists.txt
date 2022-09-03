add_executable(rosenbock "src/rosenbock.cpp")

target_link_libraries(rosenbock PUBLIC bayesship)

target_link_libraries( rosenbock PUBLIC "${LIBS}")

target_include_directories( rosenbock PUBLIC "${INCLUDE_DIRS}")

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/examples/rosenbock/python/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/examples/rosenbock/plots/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/examples/rosenbock/data/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/examples/rosenbock/data/python/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/examples/rosenbock/bin/")

file(GLOB pythonScripts "python/*.py")
file(GLOB jupyterNbs "python/*.ipynb")
file(COPY ${pythonScripts}
     DESTINATION "${CMAKE_BINARY_DIR}/examples/rosenbock/python/")
file(COPY ${jupyterNbs}
     DESTINATION "${CMAKE_BINARY_DIR}/examples/rosenbock/python/")
# Put executable in "/example/rosenbock/bin" directory 
add_custom_command(TARGET rosenbock POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:rosenbock> "${PROJECT_BINARY_DIR}/examples/rosenbock/bin/")
# Delete original executable
add_custom_command(TARGET rosenbock POST_BUILD COMMAND ${CMAKE_COMMAND} -E remove $<TARGET_FILE:rosenbock> )

