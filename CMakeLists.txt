cmake_minimum_required(VERSION 3.10)

project(misa-kmc)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(version.cmake)
include(config.cmake)
include(dependency.cmake)

set(BGHT_INCLUDE_DIR "/work1/aicao/duhao/BGHT_HashTable/BGHT-hip")
include_directories(${BGHT_INCLUDE_DIR})

if (KMC_HIP_ENABLE_FLAG)
    enable_language(HIP)
    option(DEBUG_MODE_GPU "Enable debug mode (print debug info)" OFF)
    # ON or OFF
    message("debug mode is $DEBUG_MODE_GPU")
    if(DEBUG_MODE_GPU)
        #add_compile_definitions(DEBUG_MODE_GPU)
    endif()
    add_subdirectory(gpu)
endif ()

add_subdirectory(models)
add_subdirectory(src)
add_subdirectory(profiles)
add_subdirectory(profile_serial)
# tools
if (KMC_TOOLS_BUILD_FLAG)
    add_subdirectory(tools)
endif ()

#if (KMC_TOOLS_BUILD_ENABLE_FLAG)
#    add_subdirectory(tools)
#endif ()
################################
## test configure
################################
if (KMC_TEST_BUILD_ENABLE_FLAG)
    MESSAGE(STATUS "TEST is enabled")
    enable_testing()
    # add_subdirectory(tests)
endif ()
