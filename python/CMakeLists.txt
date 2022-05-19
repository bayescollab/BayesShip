
set(SETUP_PY_IN ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in)
set(SETUP_PY_OUT ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(${SETUP_PY_IN} ${SETUP_PY_OUT})

set(README_PY_IN "${CMAKE_CURRENT_SOURCE_DIR}/README.md.in")
set(README_PY_OUT "${CMAKE_CURRENT_BINARY_DIR}/README.md")
configure_file(${README_PY_IN} ${README_PY_OUT})

add_subdirectory(bayesshippy)

install(CODE "execute_process(COMMAND ${Python_EXECUTABLE} -m pip install ${CMAKE_CURRENT_BINARY_DIR}/ )" )
