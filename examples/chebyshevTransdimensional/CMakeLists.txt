add_executable(chebyshevTransdimensional "src/chebyshevTransdimensional.cpp")

target_link_libraries(chebyshevTransdimensional PUBLIC bayesship)

target_link_libraries( chebyshevTransdimensional PUBLIC "${LIBS}")

target_include_directories( chebyshevTransdimensional PUBLIC "${INCLUDE_DIRS}")

file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/examples/chebyshevTransdimensional/python/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/examples/chebyshevTransdimensional/plots/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/examples/chebyshevTransdimensional/data/")
file(MAKE_DIRECTORY "${PROJECT_BINARY_DIR}/examples/chebyshevTransdimensional/bin/")

file(GLOB pythonScripts "python/*.py")
file(GLOB jupyterNbs "python/*.ipynb")
file(COPY ${pythonScripts}
     DESTINATION "${CMAKE_BINARY_DIR}/examples/chebyshevTransdimensional/python/")
file(COPY ${jupyterNbs}
     DESTINATION "${CMAKE_BINARY_DIR}/examples/chebyshevTransdimensional/python/")
# Put executable in "/example/chebyshevTransdimensional/bin" directory 
add_custom_command(TARGET chebyshevTransdimensional POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:chebyshevTransdimensional> "${PROJECT_BINARY_DIR}/examples/chebyshevTransdimensional/bin/")
# Delete original executable
add_custom_command(TARGET chebyshevTransdimensional POST_BUILD COMMAND ${CMAKE_COMMAND} -E remove $<TARGET_FILE:chebyshevTransdimensional> )

file(COPY "data/" DESTINATION "${PROJECT_BINARY_DIR}/examples/chebyshevTransdimensional/data/")
