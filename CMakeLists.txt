cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cudars LANGUAGES CXX CUDA)
#add_definitions(-std=c++0x)

add_compile_options(-std=c++14)

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
add_library(iso_cuars STATIC
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

add_library(ani_cuars STATIC
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
### Isotropic cuArs

add_executable(gpu_testIsotropicArs test/gpu_testIsotropicArs.cu )
target_link_libraries(gpu_testIsotropicArs  iso_cuars rofl_common ${PCL_LIBRARIES})

#target_compile_features(gpu_testIsotropicArs PUBLIC cxx_std_11)
set_target_properties(gpu_testIsotropicArs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_testIsotropicArs PROPERTIES CUDA_ARCHITECTURES "75")

####################################################################################
 
add_executable(gpu_testArsIsoMpeg7 test/gpu_testArsIsoMpeg7.cu)
target_link_libraries(gpu_testArsIsoMpeg7  iso_cuars ${PCL_LIBRARIES} rofl_common -lstdc++fs -lboost_filesystem)

target_compile_features(gpu_testArsIsoMpeg7 PUBLIC cxx_std_11)
set_target_properties(gpu_testArsIsoMpeg7 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_testArsIsoMpeg7 PROPERTIES CUDA_ARCHITECTURES "75")

####################################################################################

add_executable(gpu_testArsIsoPcl test/gpu_testArsIsoPcl.cu)
target_link_libraries(gpu_testArsIsoPcl  iso_cuars rofl_common ${PCL_LIBRARIES} -lstdc++fs)

target_compile_features(gpu_testArsIsoPcl PUBLIC cxx_std_11)
set_target_properties(gpu_testArsIsoPcl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_testArsIsoPcl PROPERTIES CUDA_ARCHITECTURES "75")

####################################################################################
 
add_executable(gpu_fullArsIsoMpeg7 app/gpu_fullArsIsoMpeg7.cu)
target_link_libraries(gpu_fullArsIsoMpeg7  iso_cuars rofl_common ${PCL_LIBRARIES} -lstdc++fs)

target_compile_features(gpu_fullArsIsoMpeg7 PUBLIC cxx_std_11)
set_target_properties(gpu_fullArsIsoMpeg7 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_fullArsIsoMpeg7 PROPERTIES CUDA_ARCHITECTURES "75")

#####################################################################################

add_executable(gpu_fullArsIsoPcl app/gpu_fullArsIsoPcl.cu)
target_link_libraries(gpu_fullArsIsoPcl  iso_cuars rofl_common ${PCL_LIBRARIES} -lstdc++fs)

target_compile_features(gpu_fullArsIsoPcl PUBLIC cxx_std_11)
set_target_properties(gpu_fullArsIsoPcl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_fullArsIsoPcl PROPERTIES CUDA_ARCHITECTURES "75")

#####################################################################################

### Anisotropic cuArs

#add_executable(gpu_testAnisotropicArs test/gpu_testAnisotropicArs.cu )
#target_link_libraries(gpu_testAnisotropicArs  ani_cuars rofl_common -lboost_iostreams -lboost_system -lboost_filesystem)
#
##target_compile_features(gpu_testAnisotropicArs PUBLIC cxx_std_11)
#set_target_properties(gpu_testAnisotropicArs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#set_target_properties(gpu_testAnisotropicArs PROPERTIES CUDA_ARCHITECTURES "75")

#####################################################################################

### translation

add_executable(gpu_testConsensusTranslationEstimator test/gpu_testConsensusTranslationEstimator.cu)
target_link_libraries(gpu_testConsensusTranslationEstimator  iso_cuars rofl_common ${PCL_LIBRARIES} -lstdc++fs)

####################################################################################
 
add_executable(gpu_testArsIsoPoseMpeg7 test/gpu_testArsIsoPoseMpeg7.cu)
target_link_libraries(gpu_testArsIsoPoseMpeg7  iso_cuars ${PCL_LIBRARIES} rofl_common -lstdc++fs -lboost_filesystem)

target_compile_features(gpu_testArsIsoPoseMpeg7 PUBLIC cxx_std_11)
set_target_properties(gpu_testArsIsoPoseMpeg7 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gpu_testArsIsoPoseMpeg7 PROPERTIES CUDA_ARCHITECTURES "75")

