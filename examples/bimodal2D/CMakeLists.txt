add_executable(bimodal2D "src/bimodal2D.cpp")

target_link_libraries(bimodal2D PUBLIC bayesship)

target_link_libraries( bimodal2D PUBLIC "${LIBS}")

target_include_directories( bimodal2D PUBLIC "${INCLUDE_DIRS}")

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/examples/bimodal2D/python/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/examples/bimodal2D/plots/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/examples/bimodal2D/data/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/examples/bimodal2D/bin/")

file(GLOB pythonScripts "python/*.py")
file(GLOB jupyterNbs "python/*.ipynb")
file(COPY ${pythonScripts}
     DESTINATION "${CMAKE_BINARY_DIR}/examples/bimodal2D/python/")
file(COPY ${jupyterNbs}
     DESTINATION "${CMAKE_BINARY_DIR}/examples/bimodal2D/python/")
# Put executable in "/example/bimodal2D/bin" directory 
add_custom_command(TARGET bimodal2D POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:bimodal2D> "${PROJECT_BINARY_DIR}/examples/bimodal2D/bin/")
# Delete original executable
add_custom_command(TARGET bimodal2D POST_BUILD COMMAND ${CMAKE_COMMAND} -E remove $<TARGET_FILE:bimodal2D> )

