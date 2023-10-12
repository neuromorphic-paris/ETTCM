cmake_minimum_required(VERSION 3.16.3)

set(LIB_NAME ETTCM)
project(${LIB_NAME} LANGUAGES CXX)

# C++ version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(${LIB_NAME}_DOC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/doc)
set(${LIB_NAME}_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(${LIB_NAME}_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

# CXX flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Werror -march=native -mtune=native -fPIC")

# Default to Release
set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Options
option(${LIB_NAME}_BUILD_DOC "Build documentation" ON)

# Dependencies
include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

# pontella
FetchContent_Declare(
  pontella
  GIT_REPOSITORY https://github.com/neuromorphic-paris/pontella
  GIT_TAG        master
)
FetchContent_MakeAvailable(pontella)

# sepia
FetchContent_Declare(
  sepia
  GIT_REPOSITORY https://github.com/neuromorphic-paris/sepia
  GIT_TAG        master
)
FetchContent_MakeAvailable(sepia)

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# OpenCV
find_package(OpenCV)
if(${OPENCV_FOUND})
  message(STATUS "OpenCV version found - ${OpenCV_VERSION}")
else()
  message(STATUS "OpenCV not found")
endif()

# Source
add_subdirectory(${${LIB_NAME}_SOURCE_DIR})

# Documentation
if(${LIB_NAME}_BUILD_DOC)
  add_subdirectory(${${LIB_NAME}_DOC_DIR})
endif()
