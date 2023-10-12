# Library
add_library(${LIB_NAME} INTERFACE)

target_include_directories(${LIB_NAME} INTERFACE ${${LIB_NAME}_INCLUDE_DIR}
  ${pontella_SOURCE_DIR}/source ${sepia_SOURCE_DIR}/source ${tarsier_SOURCE_DIR}/source)
target_link_libraries(${LIB_NAME} INTERFACE Eigen3::Eigen pthread stdc++fs)

# OpenCV
if(${OPENCV_FOUND})
  target_compile_definitions(${LIB_NAME} INTERFACE ${LIB_NAME}_OPENCV_FOUND)

  target_include_directories(${LIB_NAME} INTERFACE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${LIB_NAME} INTERFACE ${OpenCV_LIBS})
endif()
  
# Option for fast exp computation
option(${LIB_NAME}_FAST_EXP "Enable fast exp" ON)
if(${LIB_NAME}_FAST_EXP)
  target_compile_definitions(${LIB_NAME} INTERFACE ${LIB_NAME}_FAST_EXP)
endif()

# Function to add new executables
function(add_new_executable ARG)
  add_executable(${ARG} ${ARG}.cpp)
  target_link_libraries(${ARG} ${LIB_NAME})
endfunction()

# List of executables
add_new_executable(global_divergence_scaling)
add_new_executable(global_divergence_translation)
add_new_executable(global_divergence_six_dof)
