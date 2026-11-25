cmake_minimum_required(VERSION 3.10)
project(Projem)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(ZXing REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB SOURCES "*.cpp" "*.h" "*.hpp")
add_executable(${PROJECT_NAME} ${SOURCES})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ZXing::ZXing)
