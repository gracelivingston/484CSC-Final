cmake_minimum_required(VERSION 3.0)
project(geomean)

# Configure the MPI tools.
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

add_executable(geomean geomean.c)
target_compile_options(geomean PRIVATE "-Wall" "-g3")
target_link_libraries(geomean
  PRIVATE
  ${MPI_C_LIBRARIES} "m")

target_include_directories(geomean
  PRIVATE
  ${MPI_C_INCLUDE_PATH})
