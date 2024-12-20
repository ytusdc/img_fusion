cmake_minimum_required(VERSION 3.1)   # 声明要求的 cmake 最低版本

# 设置项目名称 和编程语言
# 只需要在project的LANGUAGES
project(fusion LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3")
set(CMAKE_CXX_STANDARD 14)

# 设置工程编译类型
# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)

# OpenCV
find_package(OpenCV REQUIRED)
# message 向终端输出信息
message(STATUS "OpenCV Libs: \n${OpenCV_LIBS}\n")
message(STATUS "OpenCV Libraries: \n${OpenCV_LIBRARIES}\n")
message(STATUS "OpenCV Headers: \n${OpenCV_INCLUDE_DIRS}\n")

list(APPEND INCLUDE_DIRS
        ${OpenCV_INCLUDE_DIRS}
        )

list(APPEND ALL_LIBS
        ${OpenCV_LIBRARIES}
        )

# include_directories(${CMAKE_SOURCE_DIR}/include)

include_directories(${INCLUDE_DIRS})

#生成程序
add_executable(${PROJECT_NAME}
        main.cpp
        include/stitching_v0.cpp
        include/stitching_v1.cpp
        include/stitching_v2.cpp
        include/stitching.cpp
        )

# 将链接目录添加到target
target_link_directories(${PROJECT_NAME} PUBLIC ${ALL_LIBS})
# 将库文件链接到可执行程序上
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

if (${OpenCV_VERSION} VERSION_GREATER_EQUAL 4.7.0)
    message(STATUS "Build with -DBATCHED_NMS")
    add_definitions(-DBATCHED_NMS)
endif ()
