cmake_minimum_required(VERSION 3.12)
project(izhikevich_simulation)

find_program(ISPC_EXECUTABLE ispc)
if(NOT ISPC_EXECUTABLE)
    message(FATAL_ERROR "ISPC compiler not found. Please install ISPC.")
endif()

set(ISPC_FLAGS --target=host --opt=fast-math)
set(ISPC_SRC "${CMAKE_CURRENT_SOURCE_DIR}/izhikevich.ispc")
set(ISPC_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(ISPC_OUTPUT_BASENAME "izhikevich")

add_custom_command(
    OUTPUT "${ISPC_OUTPUT_DIR}/${ISPC_OUTPUT_BASENAME}_ispc.h" "${ISPC_OUTPUT_DIR}/${ISPC_OUTPUT_BASENAME}_ispc.o"
    COMMAND ${ISPC_EXECUTABLE} ${ISPC_FLAGS} -o "${ISPC_OUTPUT_DIR}/${ISPC_OUTPUT_BASENAME}_ispc.o" -h "${ISPC_OUTPUT_DIR}/${ISPC_OUTPUT_BASENAME}_ispc.h" ${ISPC_SRC}
    DEPENDS ${ISPC_SRC}
    COMMENT "Compiling ISPC file ${ISPC_SRC}"
)

add_library(izhikevich_ispc STATIC "${ISPC_OUTPUT_DIR}/${ISPC_OUTPUT_BASENAME}_ispc.o")
set_target_properties(izhikevich_ispc PROPERTIES LINKER_LANGUAGE CXX)

add_executable(izhikevich_simulation izhikevich_main.cpp)
target_include_directories(izhikevich_simulation PRIVATE ${ISPC_OUTPUT_DIR})
target_link_libraries(izhikevich_simulation izhikevich_ispc)

set_property(TARGET izhikevich_simulation PROPERTY CXX_STANDARD 11)
set_property(TARGET izhikevich_simulation PROPERTY CXX_STANDARD_REQUIRED ON) 