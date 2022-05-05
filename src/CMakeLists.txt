
# Create list of source files looking for cpp files
file(GLOB SRC_LIST CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/src/*.cpp")


# Create object files for all source files
add_library(OBJS OBJECT ${SRC_LIST})
# Set object file options
set_property(TARGET OBJS PROPERTY  POSITION_INDEPENDENT_CODE ON)
set_property(TARGET OBJS PROPERTY CXX_STANDARD 11)
# Link to the libraries defined in main CMakeList file
target_link_libraries( OBJS PUBLIC "${LIBS}")
# Include header files defined in main CMakeList file
target_include_directories( OBJS PUBLIC "${INCLUDE_DIRS}")

# Create 2 libraries files (shared and static)
add_library(bayesship SHARED $<TARGET_OBJECTS:OBJS> ${HEADER_LIST})
add_library(bayesship_static STATIC $<TARGET_OBJECTS:OBJS> ${HEADER_LIST})
# Set properties
set_property(TARGET bayesship PROPERTY CXX_STANDARD 11)
set_property(TARGET bayesship_static PROPERTY CXX_STANDARD 11)

# Link to libraries
target_link_libraries( bayesship PUBLIC "${LIBS}")
target_link_libraries( bayesship_static PUBLIC "${LIBS}")
# Include to header files
target_include_directories( bayesship PUBLIC "${INCLUDE_DIRS}")
target_include_directories( bayesship_static PUBLIC "${INCLUDE_DIRS}")

set_target_properties(bayesship_static PROPERTIES ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set_target_properties(bayesship PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set_target_properties(bayesship_static PROPERTIES OUTPUT_NAME "bayesship")


#Install
install(TARGETS bayesship DESTINATION lib)
install(TARGETS bayesship_static DESTINATION lib)
