# build with current *.c/*.cpp plus common source files in parent folder
# executable name is current folder name.

cmake_minimum_required(VERSION 3.22)
project(plus_demo_cam_generic_image_inference_drop_frame)

find_package(PkgConfig REQUIRED)
find_package(OpenCV 4.5.0 REQUIRED)

set(CMAKE_CXX_STANDARD 11)

get_filename_component(app_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" app_name ${app_name})

SET(KPLUS_EX_COMMON_PATH    "../ex_common"                  CACHE STRING "The path of common function for examples.")
SET(KPLUS_HEADER_PATH       "/usr/include/kplus"            CACHE STRING "The path of kneron plus header.")
SET(KPLUS_LIB_PATH          "/usr/lib/kplus"                CACHE STRING "The path of kneron plus libraries.")

set(KPLUS_LIB_NAME          "kplus")
set(MATH_LIB                "m")
set(USB_LIB                 usb-1.0)

include_directories(
    ${KPLUS_EX_COMMON_PATH}
    ${KPLUS_HEADER_PATH}
    ${OpenCV_INCLUDE_DIRS}                          #openCV header
)

LINK_DIRECTORIES(${KPLUS_LIB_PATH})

file(GLOB local_src
    "*.c"
    "*.cpp"
    )

set(common_src
    ${KPLUS_EX_COMMON_PATH}/helper_functions.c
    ${KPLUS_EX_COMMON_PATH}/postprocess.c
    )

add_executable(${app_name}
    ${local_src}
    ${common_src})

target_link_libraries(${app_name} ${KPLUS_LIB_NAME} ${USB_LIB} ${MATH_LIB} ${OpenCV_LIBS} pthread)
