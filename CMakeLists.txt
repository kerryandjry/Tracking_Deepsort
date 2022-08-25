cmake_minimum_required(VERSION 3.12)
project(tracking_deepsort)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_PREFIX_PATH "/usr/include/libtorch")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_library(Torch SHARED IMPORTED)
set_property(TEST PROPERTY ENVIRONMENT LD_PRELOAD=/usr/lib/libgomp.so)

add_executable(main src/main.cpp)
target_include_directories(main PRIVATE include)
target_link_libraries(main ${TORCH_LIBRARIES} ${OpenCV_LIBS})
#if(Torch_FOUND)
  #message(STATUS "libtorch ${libtorch_VERSION} - found")
  #macro(add_example name)
    #add_executable(main src/main.cpp)
    #target_include_directories(main PRIVATE include)
    #target_link_libraries(main ${TORCH_LIBRARIES} ${OpenCV_LIBS})
  #endmacro()

#endif()


