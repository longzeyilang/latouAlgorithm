cmake_minimum_required(VERSION 3.0)
set(CMAKE_CXX_STANDARD 11)
project(latou_algorithm)
set(CMAKE_BUILD_TYPE "Release")

# arm or x86
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
    set(PLATFORM_COMPILE_FLAGS "-mfpu=neon")
else()
    # SET(PLATFORM_COMPILE_FLAGS "-march=native")
    # some places of the algorithm are designed for 128 SIMD
    # so 128 SSE may slightly faster than 256 AVX, you may want this
    set(PLATFORM_COMPILE_FLAGS "-msse -msse2 -msse3 -msse4 -mssse3")  # SSE only
endif()

SET(PLATFORM_COMPILE_FLAGS "-DMIPP_NO_INTRINSICS")  # close SIMDs
set(COMMON_COMPILE_FLAGS "-fopenmp -Wall -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${PLATFORM_COMPILE_FLAGS} ${COMMON_COMPILE_FLAGS} $ENV{CXXFLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g2 -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

#检测内存泄漏
# set(COMMON_FLAGS "-funwind-tables -g -Wall -Wextra -Wpedantic -Wshadow -Wpointer-arith -Wcast-qual \
#                   -Wno-missing-braces -Wswitch-default -Wcast-align -Wunreachable-code -Wundef \
#                   -Wuninitialized")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99 ${COMMON_FLAGS} -Wstrict-prototypes -Wmissing-prototypes")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 ${COMMON_FLAGS}")
# add_library(heapusage SHARED heapusage/humain.c heapusage/hulog.cpp)
# target_compile_features(heapusage PRIVATE cxx_variadic_templates)
# set_target_properties(heapusage PROPERTIES LINK_FLAGS "-fPIC")
# target_link_libraries(heapusage pthread dl)


# OpenCV3.2
#set(OpenCV_DIR /usr/local/OpenCV3.2/share/OpenCV)
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "The Opencv lib is found")
endif()
include_directories(${OpenCV_INCLUDE_DIRS})
message("${OpenCV_INCLUDE_DIRS}")

#include MIPP headers
include_directories(${INCLUDE_DIRECTORIES} "${CMAKE_CURRENT_SOURCE_DIR}")

#生成liblatou.so动态链接库，并进行测试
set(SRCS latou.cpp shapeMatch.cpp Log.cpp)
set(HDRS ${INCLUDE_DIRECTORIES} latou.h)
add_library(latou SHARED ${SRCS} ${HDRS})
#add_executable(${PROJECT_NAME} testlib.cpp)
#target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} /media/gzy/f093187a-674a-48c6-8baf-b04e5e3a9a0a/latou/latouAlgorithm/lib/liblatou.so)


#直接全部测试
# add_executable(${PROJECT_NAME} latou.cpp shapeMatch.cpp Log.cpp)
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
