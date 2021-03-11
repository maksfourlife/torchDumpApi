cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(DumpApi)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(DumpApi src/main.cpp src/dump.cpp src/network.cpp)
target_link_libraries(DumpApi "${TORCH_LIBRARIES}")
set_property(TARGET DumpApi PROPERTY CXX_STANDARD 14)
