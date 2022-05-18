
set(SETUP_PY_IN ${CMAKE_CURRENT_SOURCE_DIR}/setup.py.in)
set(SETUP_PY_OUT ${CMAKE_CURRENT_BINARY_DIR}/setup.py)
configure_file(${SETUP_PY_IN} ${SETUP_PY_OUT})

add_subdirectory(bayesshippy)

install(CODE "execute_process(COMMAND ${Python_EXECUTABLE} -m pip install ${CMAKE_CURRENT_BINARY_DIR}/ )" )
#install(CODE "execute_process(COMMAND ${Python_EXECUTABLE} -m pip install . )" )
install(CODE "execute_process(COMMAND  echo ${Python_EXECUTABLE} )"  )
