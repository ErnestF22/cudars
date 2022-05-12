# CUDARS: Angular Radon Spectrum - CUDA Version
#### Copyright (C) 2017-2020 Dario Lodi Rizzini.
#### Copyright (C) 2021- Dario Lodi Rizzini, Ernesto Fontana.


OVERVIEW
-------------------------------------------------

Library **cudars** implements the Angular Radon Spectrum method 
for estimation of rotation. 
It has been kept to a minimal design. 

If you use this library, please cite the following paper: 

D. Lodi Rizzini. 
Angular Radon Spectrum for Rotation Estimation. 
Pattern Recognition, Volume 84, Dec. 2018, Pages 182-196, 
DOI [10.1016/j.patcog.2018.07.017](https://doi.org/10.1016/j.patcog.2018.07.017).

````
  @article{lodirizzini2018pr,
    author={Lodi Rizzini, D.},
    title={{Angular Radon Spectrum for Rotation Estimation}}
    journal={Pattern Recognition},
    volume={84},
    pages={182--196},
    month={dec},
    year={2018},
    publisher={Elsevier},
    issn = {},
    doi = {10.1016/j.patcog.2018.07.017},
    note = {DOI 10.1016/j.patcog.2018.07.017, EID 2-s2.0-85050072081},
  }
````

or the most relevant associated publications by visiting: 
http://rimlab.ce.unipr.it/


DEPENDENCIES
-------------------------------------------------

The software depends on the following external libraries

- Boost (submodule lexical_cast)
- Point Cloud Library 1.8+
- NVCC CUDA Compiler; arch >= 52 (tested mainly on 75, 86). 
Please note that part of the algorithms have to run on a NVidia GPU.

Other dependencies are placed in directory thirdparty. 
Some examples require the external application "gnuplot" to display 
results. 
Necessary functions of common libraries (used also in the original ARS) 
such Eigen have been reimplemented in CUDA compatible code, generally
usable in both host and device.


HOW TO COMPILE
-------------------------------------------------

Let ${cudars_ROOT} be the install directory of your local copy 
of library cudars. 
The following standard commands are required to compile it:

-  cd ${cudars_ROOT}
-  mkdir build
-  cd build
-  cmake ..
-  make

You can also install the library into a system directory. 
To change the install directory you must set cmake environment
variable ${CMAKE_INSTALL_PREFIX} (e.g. using command "ccmake .."
before calling "cmake .."). 
Its default value on UNIX-like/Linux systems is "/usr/local".
After compiling library cudars, run the command:

-  sudo make install

The command "sudo" is required only if ${CMAKE_INSTALL_PREFIX} 
is a system diretory managed by administrator user root.
Such command copies:
- header files of ${cudars_ROOT}/include/cudars to
   ${CMAKE_INSTALL_PREFIX}/include/cudars/
- library files ${cudars_ROOT}/lib/libcudars.a to
   ${CMAKE_INSTALL_PREFIX}/lib/
- cmake script ${cudars_ROOT}/cmake_modules/cudarsConfig.cmake to
   ${CMAKE_INSTALL_PREFIX}/share/cudars/


HOW TO USE LIBRARY cudars IN YOUR PROJECT
-------------------------------------------------

If library cudars has been installed in system directory "/usr/local",
then it is straighforward to use it in your projects.
You need to add the following lines to your project as in this example:


> CMAKE_MINIMUM_REQUIRED(VERSION 3.8)  
> PROJECT(foobar)  
> 
> find_package(cudars REQUIRED)  
> message(STATUS "cudars_FOUND ${cudars_FOUND}")  
> message(STATUS "cudars_INCLUDE_DIRS ${cudars_INCLUDE_DIRS}")  
> message(STATUS "cudars_LIBRARY_DIRS ${cudars_LIBRARY_DIRS}")  
> message(STATUS "cudars_LIBRARIES ${cudars_LIBRARIES}")  
>  
> if(${cudars_FOUND})   
>   include_directories(${cudars_INCLUDE_DIRS})  
>   link_directories(${cudars_LIBRARY_DIRS})  
> endif()  
> 
> add_executable(foobar foobar.cpp)  
> target_link_libraries(foobar ${cudars_LIBRARIES})  

The above example uses the variables defined in cudarsConfig.cmake:

-  cudars_FOUND - system has cudars module
-  cudars_INCLUDE_DIRS - the cudars include directories
-  cudars_LIBRARY_DIRS - the cudars library directories
-  cudars_LIBRARIES - link these to use cudars


