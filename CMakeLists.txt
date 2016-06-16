#Very simple CMake to compile everything into 1 library

cmake_minimum_required(VERSION 2.8.9)

project (QUADIS)

set(PROJ_DIRS "src")
set(EXE_DIR "main" )

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11" )

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_INSTALL_LIBDIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
#set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR})


foreach(dir ${PROJ_DIRS})
  list(APPEND HEADERS "${dir}/*.h")
  list(APPEND SOURCES "${dir}/*.cpp")
endforeach(dir)


file(GLOB PROJHDR ${HEADERS})
file(GLOB PROJSRC ${SOURCES})

include_directories(${PROJ_DIRS} )



add_library(layerdnn SHARED ${PROJSRC} )
target_link_libraries(layerdnn  )

set ( PROJECT_LINK_LIBS liblayerdnn.so )
link_directories (${CMAKE_BINARY_DIR}/lib)

file(COPY ${PROJHDR} DESTINATION "include/")

#Generate executables for each .cc file in the EXE_DIR
foreach(dir ${EXE_DIR})
  file(GLOB EXESRC "${dir}/*.cpp")
  foreach(exe ${EXESRC})
    get_filename_component(EXENAME ${exe} NAME_WE)#file name without extension
    add_executable(${EXENAME} ${exe})
    target_link_libraries(${EXENAME}  ${PROJECT_LINK_LIBS} ${ROOT_LIBRARIES} )
  endforeach()
endforeach()




