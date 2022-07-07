# - Try to find Library cudars
# Once done, this will define
#
#  cudars_FOUND - system has cudars module
#  cudars_INCLUDE_DIRS - the cudars include directories
#  cudars_LIBRARY_DIRS - the cudars library directories
#  cudars_LIBRARIES - link these to use cudars


# Uses  directory to search mrf_segmentation directory!
set(cudars_PREFIX_DIR /usr/local)
message(STATUS "Searching cudars in directory ${cudars_PREFIX_DIR}." )

# Searches include directory /usr/local/include/cudars
find_path(cudars_INCLUDE_DIR cudars ${cudars_PREFIX_DIR}/include)
message(STATUS "    cudars_INCLUDE_DIR ${cudars_INCLUDE_DIR}." )
set(cudars_INCLUDE_DIRS ${cudars_INCLUDE_DIR})
  
# Searches library librimagraph.a in /usr/local/lib
find_path(cudars_LIBRARY_DIR libcudars.a ${cudars_PREFIX_DIR}/lib)
message(STATUS "    cudars_LIBRARY_DIR ${cudars_LIBRARY_DIR}." )
set(cudars_LIBRARY_DIRS ${cudars_PREFIX_DIR}/lib)

# Sets the names of library components (actually A name and A component)
find_library(cudars_LIBRARY cudars ${cudars_LIBRARY_DIRS})
message(STATUS "    cudars_LIBRARY ${cudars_LIBRARY}." )
set(cudars_LIBRARIES ${cudars_LIBRARY})

if(("${cudars_INCLUDE_DIR}" STREQUAL "cudars_INCLUDE_DIR-NOTFOUND") OR
   ("${cudars_LIBRARY_DIRS}" STREQUAL "cudars_LIBRARY_DIRS-NOTFOUND") OR
   ("${cudars_LIBRARY}" STREQUAL "cudars_LIBRARY-NOTFOUND")
  )
  message(STATUS "Library cudars NOT found")
  unset(cudars_FOUND)
  unset(cudars_INCLUDE_DIR)
  unset(cudars_LIBRARY_DIR)
  unset(cudars_LIBRARY)
  unset(cudars_LIBRARIES)
endif()

mark_as_advanced(cudars_INCLUDE_DIRS cudars_LIBRARY_DIRS cudars_LIBRARIES)

