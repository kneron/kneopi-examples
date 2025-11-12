cmake_minimum_required(VERSION 3.22)
project(example_nnm_image)

SET(TARGET_NAME example_nnm_image)
ADD_DEFINITIONS("-DVATICS_PLATFORM -DKL730")

SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

SET(APP_PATH            "${CMAKE_CURRENT_SOURCE_DIR}/../app_flow"       CACHE STRING "The path of app.")
SET(COMMON_PATH         "${CMAKE_CURRENT_SOURCE_DIR}/../common"         CACHE STRING "The path of common include.")
SET(VTCS_HEADER_PATH    "/usr/include/vtcs_root_leipzig"                CACHE STRING "The path of vtcs header.")
SET(VTCS_LIB_PATH       "/usr/lib/vtcs_root_leipzig"                    CACHE STRING "The path of vtcs libraries.")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${APP_PATH}/include/
                    ${APP_PATH}/pre_post_proc/include
                    ${COMMON_PATH}
                    ${VTCS_HEADER_PATH}/vmf
                    ${VTCS_HEADER_PATH}/util
                    ${VTCS_HEADER_PATH}
)

LINK_DIRECTORIES(${VTCS_LIB_PATH}
                 ${APP_PATH}/lib)

SET(LINK_LIST vmf
              util
              iniparser
              vmf_nnm
              app_yolo
              kutils
              pthread
              m)

# executable
FILE(GLOB_RECURSE SRC_LIST "./*.c"
                           "${APP_PATH}/*.c"
)

ADD_EXECUTABLE(${TARGET_NAME} ${SRC_LIST})
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN/../lib")
TARGET_LINK_LIBRARIES(${TARGET_NAME} ${LINK_LIST})

SET(OUTPUT_KP_FIRMWARE_PATH ${CMAKE_BINARY_DIR}/bin)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

add_custom_command(
    TARGET ${TARGET_NAME}
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory  ${CMAKE_CURRENT_SOURCE_DIR}/../res/ini ${EXECUTABLE_OUTPUT_PATH}/ini
    COMMAND ${CMAKE_COMMAND} -E copy_directory  ${CMAKE_CURRENT_SOURCE_DIR}/../res/nef ${EXECUTABLE_OUTPUT_PATH}/nef
    COMMAND ${CMAKE_COMMAND} -E copy_directory  ${CMAKE_CURRENT_SOURCE_DIR}/../res/images ${EXECUTABLE_OUTPUT_PATH}/images
)
