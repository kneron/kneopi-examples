cmake_minimum_required(VERSION 3.15)
project(main LANGUAGES C)

# Compiler options
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -Wall")

# Directories
set(DIR_Config   ${CMAKE_SOURCE_DIR}/lib/Config)
set(DIR_LCD      ${CMAKE_SOURCE_DIR}/lib/LCD)
set(DIR_FONTS    ${CMAKE_SOURCE_DIR}/lib/Fonts)
set(DIR_GUI      ${CMAKE_SOURCE_DIR}/lib/GUI)

set(DIR_Examples ${CMAKE_SOURCE_DIR}/examples)
set(DIR_BIN      ${CMAKE_BINARY_DIR}/bin)

# Collect source files
file(GLOB SRC_Config 	CONFIGURE_DEPENDS ${DIR_Config}/*.c)
file(GLOB SRC_LCD 	CONFIGURE_DEPENDS ${DIR_LCD}/*.c)
file(GLOB SRC_FONTS 	CONFIGURE_DEPENDS ${DIR_FONTS}/*.c)
file(GLOB SRC_GUI 	CONFIGURE_DEPENDS ${DIR_GUI}/*.c)
file(GLOB SRC_Examples 	CONFIGURE_DEPENDS ${DIR_Examples}/*.c)

# Create shared libraries
add_library(Config 	SHARED ${SRC_Config})
add_library(LCD 	SHARED ${SRC_LCD})
add_library(Fonts 	SHARED ${SRC_FONTS})
add_library(GUI 	SHARED ${SRC_GUI})

# Set include directories
target_include_directories(Config 	PUBLIC ${DIR_Config})
target_include_directories(LCD 		PUBLIC ${DIR_LCD} ${DIR_Config})
target_include_directories(Fonts 	PUBLIC ${DIR_FONTS})
target_include_directories(GUI 		PUBLIC ${DIR_GUI} ${DIR_Config})

# Create executable
add_executable(${PROJECT_NAME} ${SRC_Examples})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE Config LCD Fonts GUI lgpio m)

# Set output directory
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${DIR_BIN})
set_target_properties(Config PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${DIR_BIN})
set_target_properties(LCD PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${DIR_BIN})
set_target_properties(Fonts PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${DIR_BIN})
set_target_properties(GUI PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${DIR_BIN})

