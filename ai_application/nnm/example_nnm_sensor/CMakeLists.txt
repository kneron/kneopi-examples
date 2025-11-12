cmake_minimum_required(VERSION 3.22)
project(example_nnm_sensor)

find_package(PkgConfig REQUIRED)
find_package(OpenCV 4.5.0 REQUIRED)

set(CMAKE_CXX_STANDARD 11)

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

ADD_DEFINITIONS("-DVATICS_PLATFORM")
SET(TARGET_NAME example_nnm_sensor)

SET(APP_PATH            "${CMAKE_CURRENT_SOURCE_DIR}/../app_flow"       CACHE STRING "The path of app.")
SET(COMMON_PATH         "${CMAKE_CURRENT_SOURCE_DIR}/../common"         CACHE STRING "The path of common include.")
SET(FEC_PATH            "${CMAKE_CURRENT_SOURCE_DIR}/fec"               CACHE STRING "The path of fec.")
SET(VTCS_HEADER_PATH    "/usr/include/vtcs_root_leipzig"                CACHE STRING "The path of vtcs header.")
SET(VTCS_LIB_PATH       "/usr/lib/vtcs_root_leipzig"                    CACHE STRING "The path of vtcs libraries.")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${APP_PATH}/include/
                    ${COMMON_PATH}
                    ${FEC_PATH}
                    ${VTCS_HEADER_PATH}/vmf
                    ${VTCS_HEADER_PATH}/util
                    ${VTCS_HEADER_PATH}
                    ${OpenCV_INCLUDE_DIRS}
)

LINK_DIRECTORIES(${VTCS_LIB_PATH}
                 ${APP_PATH}/lib)

SET(LINK_LIST vmf
              msgbroker
              util
              iniparser
              vmf_nnm
              app_yolo
              kutils
              pthread
              m)

FILE(GLOB_RECURSE SRC_LIST "./*.c*"
                           "${FEC_PATH}/*.c"
)

ADD_EXECUTABLE(${TARGET_NAME} ${SRC_LIST})
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN/../lib")
TARGET_LINK_LIBRARIES(${TARGET_NAME} ${OpenCV_LIBS} ${LINK_LIST} )

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

add_custom_command(
    TARGET ${TARGET_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory  ${CMAKE_CURRENT_SOURCE_DIR}/../res/ini ${EXECUTABLE_OUTPUT_PATH}/ini
    COMMAND ${CMAKE_COMMAND} -E copy_directory  ${CMAKE_CURRENT_SOURCE_DIR}/../res/nef ${EXECUTABLE_OUTPUT_PATH}/nef
    COMMAND ${CMAKE_COMMAND} -E copy_directory  ${CMAKE_CURRENT_SOURCE_DIR}/../res/Resource ${EXECUTABLE_OUTPUT_PATH}/Resource
)
