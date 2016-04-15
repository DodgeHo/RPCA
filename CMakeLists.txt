cmake_minimum_required(VERSION 3.3)
project(rpca)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(OpenCV 3 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories( ${Boost_INCLUDE_DIRS} )

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY bin)
set(SOURCE_FILES src/InexactRPCASolver.cpp src/RPCA.h src/RPInexactRPCASolver.cpp src/Converter.cpp src/Converter.h
        main.cpp)
add_executable(rpca ${SOURCE_FILES})
target_link_libraries(rpca ${OpenCV_LIBS} ${ARMADILLO_LIBRARIES} ${Boost_SYSTEM_LIBRARY} ${Boost_FILESYSTEM_LIBRARY} )
