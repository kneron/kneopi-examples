cmake_minimum_required(VERSION 3.22)
project(srb_receiver)

SET(TARGET_NAME srb_receiver)
ADD_DEFINITIONS("-DVATICS_PLATFORM -D_GNU_SOURCE")

SET(VTCS_HEADER_PATH    "/usr/include/vtcs_root_leipzig"                CACHE STRING "The path of vtcs header.")
SET(VTCS_LIB_PATH       "/usr/lib/vtcs_root_leipzig"                    CACHE STRING "The path of vtcs libraries.")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${VTCS_HEADER_PATH}/vmf
                    ${VTCS_HEADER_PATH}/util
                    ${VTCS_HEADER_PATH}/msgbroker
)

LINK_DIRECTORIES(${VTCS_LIB_PATH})

SET(LINK_LIST vmf
              util
              iniparser
	      msgbroker
              pthread)

# executable
FILE(GLOB_RECURSE SRC_LIST "srb_receiver.cpp"
)

ADD_EXECUTABLE(${TARGET_NAME} ${SRC_LIST})
set_target_properties(${TARGET_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN/../lib")
TARGET_LINK_LIBRARIES(${TARGET_NAME} ${LINK_LIST})

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

