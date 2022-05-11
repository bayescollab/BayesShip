add_executable(gaussian1D "src/gaussian1D.cpp")

target_link_libraries(gaussian1D PUBLIC bayesship)

target_link_libraries( gaussian1D PUBLIC "${LIBS}")

target_include_directories( gaussian1D PUBLIC "${INCLUDE_DIRS}")

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/examples/gaussian1D/python/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/examples/gaussian1D/plots/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/examples/gaussian1D/data/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/examples/gaussian1D/bin/")

file(GLOB pythonScripts "python/*.py")
file(GLOB jupyterNbs "python/*.ipynb")
file(COPY ${pythonScripts}
     DESTINATION "${CMAKE_BINARY_DIR}/examples/gaussian1D/python/")
file(COPY ${jupyterNbs}
     DESTINATION "${CMAKE_BINARY_DIR}/examples/gaussian1D/python/")
# Put executable in "/example/gaussian1D/bin" directory 
add_custom_command(TARGET gaussian1D POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gaussian1D> "${PROJECT_BINARY_DIR}/examples/gaussian1D/bin/")
# Delete original executable
add_custom_command(TARGET gaussian1D POST_BUILD COMMAND ${CMAKE_COMMAND} -E remove $<TARGET_FILE:gaussian1D> )

