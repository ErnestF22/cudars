cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cudars LANGUAGES CXX CUDA)
#add_definitions(-std=c++0x)

add_compile_options(-std=c++14)

# Set the output directory for the build executables and libraries
SET(cudars_RUNTIME_OUTPUT_DIRECTORY ${cudars_SOURCE_DIR}/bin CACHE PATH "Target for the binaries")
SET(cudars_LIBRARY_OUTPUT_DIRECTORY ${cudars_SOURCE_DIR}/lib CACHE PATH "Target for the libraries")
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${cudars_LIBRARY_OUTPUT_DIRECTORY})
SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${cudars_LIBRARY_OUTPUT_DIRECTORY})
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${cudars_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_CUDA_FLAGS_DEBUG "-g -G")
#add_compile_options(-lineinfo)

#Compute capability for NVIDIA RTX 3090 -> 86
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75)


include_directories(include)
include_directories(src)

include(CTest)


#list(APPEND ars_LINK_LIBS boost_iostreams boost_system boost_filesystem)
find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})


# Eigen library v. 3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})


# PCL library 
#message(STATUS "pcl dir: ${PCL_DIR}")
find_package(PCL 1.8 REQUIRED COMPONENTS search KdTree)
if(${PCL_FOUND})
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})
endif()


find_package(rofl REQUIRED)
if(${rofl_FOUND})
    include_directories(${rofl_INCLUDE_DIRS})
    link_directories(${rofl_LIBRARY_DIRS})
endif()



# Library files
add_library(cudars
    src/ars2d.cpp
    src/ars2d.cu
    
    src/definitions.cpp
    src/utils.cpp
    src/functions.cpp

    src/ArsKernelIsotropic2d.cpp
    src/BBOptimizer1d.cpp
    src/Profiler.cpp

    src/mpeg7RW.cpp
    src/mpeg7_io.cpp
)

add_library(cudars_bbtransl
    src/BBTranslation.cpp
    src/Box.cpp
)

add_library(cudars_ani STATIC
    src/ars2d.cpp
    src/ars2d.cu
    
    src/definitions.cpp
    src/utils.cpp
    src/functions.cpp


    src/GaussianMixtureEstimator.cpp
    src/DisjointSet.cpp
    src/MortonSort.cpp


    src/ArsKernelIsotropic2d.cpp
    src/ArsKernelAnisotropic2d.cpp
    src/BBOptimizer1d.cpp
    src/Profiler.cpp
)


####################################################################################
### Isotropic cudArs

add_executable(gpu_testIsotropicArs test/gpu_testIsotropicArs.cu)
target_link_libraries(gpu_testIsotropicArs cudars rofl_common ${PCL_LIBRARIES})

#target_compile_features(gpu_testIsotropicArs PUBLIC cxx_std_11)
set_target_properties(gpu_testIsotropicArs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_testIsotropicArs PROPERTIES CUDA_ARCHITECTURES "75")

####################################################################################
 
add_executable(gpu_testArsIsoMpeg7 test/gpu_testArsIsoMpeg7.cu)
target_link_libraries(gpu_testArsIsoMpeg7 cudars ${PCL_LIBRARIES} rofl_common -lstdc++fs -lboost_filesystem)

target_compile_features(gpu_testArsIsoMpeg7 PUBLIC cxx_std_11)
set_target_properties(gpu_testArsIsoMpeg7 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_testArsIsoMpeg7 PROPERTIES CUDA_ARCHITECTURES "75")

####################################################################################

add_executable(gpu_testArsIsoPcl test/gpu_testArsIsoPcl.cu)
target_link_libraries(gpu_testArsIsoPcl  cudars rofl_common ${PCL_LIBRARIES} -lstdc++fs)

target_compile_features(gpu_testArsIsoPcl PUBLIC cxx_std_11)
set_target_properties(gpu_testArsIsoPcl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_testArsIsoPcl PROPERTIES CUDA_ARCHITECTURES "75")

####################################################################################
 
add_executable(gpu_fullArsIsoMpeg7 app/gpu_fullArsIsoMpeg7.cu)
target_link_libraries(gpu_fullArsIsoMpeg7  cudars rofl_common ${PCL_LIBRARIES} -lstdc++fs)

target_compile_features(gpu_fullArsIsoMpeg7 PUBLIC cxx_std_11)
set_target_properties(gpu_fullArsIsoMpeg7 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_fullArsIsoMpeg7 PROPERTIES CUDA_ARCHITECTURES "75")

### + transl (full pose)
add_executable(gpu_fullArsIsoPoseMpeg7 app/gpu_fullArsIsoPoseMpeg7.cu)
target_link_libraries(gpu_fullArsIsoPoseMpeg7 cudars cudars_bbtransl rofl_common ${PCL_LIBRARIES} -lstdc++fs)

# target_compile_features(gpu_fullArsIsoPoseMpeg7 PUBLIC cxx_std_11)
set_target_properties(gpu_fullArsIsoPoseMpeg7 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_fullArsIsoPoseMpeg7 PROPERTIES CUDA_ARCHITECTURES "75")

#####################################################################################

add_executable(gpu_fullArsIsoPcl app/gpu_fullArsIsoPcl.cu)
target_link_libraries(gpu_fullArsIsoPcl  cudars rofl_common ${PCL_LIBRARIES} -lstdc++fs)

target_compile_features(gpu_fullArsIsoPcl PUBLIC cxx_std_11)
set_target_properties(gpu_fullArsIsoPcl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_fullArsIsoPcl PROPERTIES CUDA_ARCHITECTURES "75")

#####################################################################################

### Anisotropic cudArs

#add_executable(gpu_testAnisotropicArs test/gpu_testAnisotropicArs.cu )
#target_link_libraries(gpu_testAnisotropicArs  cudars_ani rofl_common -lboost_iostreams -lboost_system -lboost_filesystem)
#
##target_compile_features(gpu_testAnisotropicArs PUBLIC cxx_std_11)
#set_target_properties(gpu_testAnisotropicArs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(gpu_testAnisotropicArs PROPERTIES CUDA_ARCHITECTURES "75")

#####################################################################################

### translation - rofl grid

add_executable(gpu_testConsensusTranslationEstimator test/gpu_testConsensusTranslationEstimator.cu)
target_link_libraries(gpu_testConsensusTranslationEstimator  cudars rofl_common ${PCL_LIBRARIES} -lstdc++fs)

####################################################################################
 
add_executable(gpu_testArsIsoPoseMpeg7 test/gpu_testArsIsoPoseMpeg7.cu)
target_link_libraries(gpu_testArsIsoPoseMpeg7  cudars ${PCL_LIBRARIES} rofl_common -lstdc++fs -lboost_filesystem)

target_compile_features(gpu_testArsIsoPoseMpeg7 PUBLIC cxx_std_11)
set_target_properties(gpu_testArsIsoPoseMpeg7 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_testArsIsoPoseMpeg7 PROPERTIES CUDA_ARCHITECTURES "75")

####################################################################################
 
add_executable(gpu_testArsIsoPosePcl test/gpu_testArsIsoPosePcl.cu)
target_link_libraries(gpu_testArsIsoPosePcl  cudars ${PCL_LIBRARIES} rofl_common -lstdc++fs -lboost_filesystem)

target_compile_features(gpu_testArsIsoPosePcl PUBLIC cxx_std_11)
set_target_properties(gpu_testArsIsoPosePcl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_testArsIsoPosePcl PROPERTIES CUDA_ARCHITECTURES "75")

### translation - Teaser TLS

add_executable(testTlsScalarCons test/testTlsScalarCons.cu)
target_link_libraries(testTlsScalarCons cudars rofl_common ${PCL_LIBRARIES} -lstdc++fs)

### translation - B&B Transl

add_executable(testBBTransl test/testBBTransl.cpp)
target_link_libraries(testBBTransl cudars cudars_bbtransl rofl_common ${PCL_LIBRARIES} -lstdc++fs)

add_executable(gpu_testBBTransl test/gpu_testBBTransl.cu src/BBTranslation.cu src/prioqueue.cu)
target_link_libraries(gpu_testBBTransl cudars rofl_common ${PCL_LIBRARIES} -lstdc++fs)

target_compile_features(gpu_testBBTransl PUBLIC cxx_std_11)
set_target_properties(gpu_testBBTransl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_testBBTransl PROPERTIES CUDA_ARCHITECTURES "75")

add_executable(fullBBTransl app/fullBBTransl.cpp src/Box.cpp src/BBTranslation.cpp)
target_link_libraries(fullBBTransl cudars rofl_common ${PCL_LIBRARIES} -lstdc++fs)

target_compile_features(fullBBTransl PUBLIC cxx_std_11)
set_target_properties(fullBBTransl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(fullBBTransl PROPERTIES CUDA_ARCHITECTURES "75")

### prioqueue
add_executable(testPrioqueue test/testPrioqueue.cpp src/definitions.cpp src/utils.cpp)
# target_link_libraries(testPrioqueue cudars rofl_common ${PCL_LIBRARIES} -lstdc++fs)

add_executable(testPrioqueueBox test/testPrioqueueBox.cu src/definitions.cpp src/utils.cpp src/prioqueue.cu)

####################################################################################



# add_library(cudars STATIC)
# set_target_properties(cudars PROPERTIES PUBLIC_HEADER)
# install(TARGETS cudars
#         PUBLIC_HEADER
#           DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cudars
# )

INSTALL(TARGETS cudars 
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

# Option "make install": copy headers
FILE(GLOB headers "${CMAKE_CURRENT_SOURCE_DIR}/include/cudars/*.h")
FILE(GLOB cuheaders "${CMAKE_CURRENT_SOURCE_DIR}/include/cudars/*.cuh")
FILE(GLOB headers_3rdparty "${CMAKE_CURRENT_SOURCE_DIR}/include/cudars/thirdparty/*.h")
INSTALL(FILES ${headers} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/cudars/)
INSTALL(FILES ${cuheaders} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/cudars/)
INSTALL(FILES ${headers_3rdparty} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/cudars/thirdparty/)

# Option "make install": copy cmake script
FILE(GLOB cmake_script "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/cudarsConfig.cmake")
message(STATUS "cmake_script " ${cmake_script})
INSTALL(FILES ${cmake_script} DESTINATION ${CMAKE_INSTALL_PREFIX}/share/cudars/)

# uninstall target
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_modules/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_modules/cmake_uninstall.cmake)
endif()
