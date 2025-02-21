cmake_minimum_required(VERSION 3.10)
project(ObstacleFreePolytopes)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PCL 1.10 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenMP REQUIRED)

# Define the source files
set(SOURCE_FILES
    src/ObstacleFree.cpp  # Main file
    src/corridor_finder.cpp
    src/ciri.cpp
    src/kdtree.c
    src/utils/banded_system.cpp
    src/utils/ellipsoid.cpp
    src/utils/geometry_utils.cpp
    src/utils/lbfgs.cpp
    src/utils/mvie.cpp
    src/utils/optimization_utils.cpp
    src/utils/piece.cpp
    src/utils/polynomial_interpolation.cpp
    src/utils/polytope.cpp
    src/utils/quickhull.cpp
    src/utils/root_finder.cpp
    src/utils/sdlp.cpp
    src/utils/trajectory.cpp
)

# Define the executable
add_executable(obstaclefree ${SOURCE_FILES})

target_include_directories(obstaclefree PUBLIC 
    ${PCL_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/data_structure
    ${CMAKE_SOURCE_DIR}/data_structure/base  # ADD THIS LINE
    ${CMAKE_SOURCE_DIR}/utils
    ${CMAKE_SOURCE_DIR}/utils/optimization
    ${CMAKE_SOURCE_DIR}/utils/geometry
    ${CMAKE_SOURCE_DIR}/utils/header
)

# Link necessary libraries
target_link_libraries(obstaclefree
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    Threads::Threads
)

set(ENV{VTK_SILENCE_WARNINGS} "1")

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(obstaclefree OpenMP::OpenMP_CXX)
endif()

# Export compile commands for debugging
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)