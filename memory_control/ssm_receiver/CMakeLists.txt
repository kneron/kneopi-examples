cmake_minimum_required(VERSION 3.22)
project(example_ssm)

ADD_DEFINITIONS("-DVATICS_PLATFORM")
SET(TARGET_NAME ssm_receiver)

SET(VTCS_HEADER_PATH    "/usr/include/vtcs_root_leipzig"                CACHE STRING "The path of vtcs header.")
SET(VTCS_LIB_PATH       "/usr/lib/vtcs_root_leipzig"                    CACHE STRING "The path of vtcs libraries.")

INCLUDE_DIRECTORIES(${VTCS_HEADER_PATH}/util
                    ${VTCS_HEADER_PATH}/vmf
)

LINK_DIRECTORIES(${VTCS_LIB_PATH})

SET(LINK_LIST util
	      vmf
              pthread)

FILE(GLOB_RECURSE SRC_LIST "ssm_receiver.c*"
)

ADD_EXECUTABLE(${TARGET_NAME} ${SRC_LIST})
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN/../lib")
TARGET_LINK_LIBRARIES(${TARGET_NAME} ${LINK_LIST})

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
