cmake_minimum_required (VERSION 3.14)
project (snpehelper)
set (APP "snpehelper")

set( APP_SOURCES
    "dllmain.cpp"
    "snpehelper.cpp"
    "SNPERuntime.cpp"
)

set (CHIPSET "QCS6490")
set (SNPE_ROOT "/data/sdk/v2.26.0.240828/qairt/2.26.0.240828")
set (SNPE_INCLUDE_DIR "${SNPE_ROOT}/include/SNPE")
set (SNPE_LIB_PREFIX "${SNPE_ROOT}/lib")
set (_dtuple_POSTFIX "ubuntu-gcc9.4")
set (PYTHON_BASE_DIR "/usr/")
set (PYTHON_INCLUDE_DIR "${PYTHON_BASE_DIR}/include/python3.12")
set (PYTHON_LIB_DIR "${PYTHON_BASE_DIR}/lib/python3.12")
set (PYBIND11_INCLUDE_DIR "${PYTHON_BASE_DIR}/lib/python3/dist-packages/pybind11/include")

#set(RELEASE "Release/")
set(CMAKE_CXX_STANDARD 17)

message("Linking with LE1.0 SNPE")
# Prefer Ubuntu libsnpe1 (/usr/lib/libSNPE.so): its version must match the
# Hexagon skels in /usr/lib/rfsa/adsp. Linking the 2.26 SDK host library
# against 2.43 DSP skels causes setTargetRuntime to fall back to CPU.
if(EXISTS "/usr/lib/libSNPE.so")
	get_filename_component(SNPE_LIB_PATH "/usr/lib/libSNPE.so" REALPATH)
	message(STATUS "Using system libSNPE at ${SNPE_LIB_PATH} (libsnpe1 / DSP skel compatible)")
else()
	get_filename_component(SNPE_LIB_PATH "${SNPE_LIB_PREFIX}/aarch64-${_dtuple_POSTFIX}/libSNPE.so" REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
	message(STATUS "Using SDK libSNPE at ${SNPE_LIB_PATH}")
endif()
get_filename_component(SNPE_HTP_PATH "${SNPE_LIB_PREFIX}/aarch64-${_dtuple_POSTFIX}/libSnpeHtpPrepare.so" REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if(CHIPSET STREQUAL "QCS8550")
	message("QCS8550 is selected")
	get_filename_component(SNPE_STUB_PATH "${SNPE_LIB_PREFIX}/aarch64-${_dtuple_POSTFIX}/libSnpeHtpV73Stub.so" REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
	get_filename_component(SNPE_SKEL_PATH "${SNPE_LIB_PREFIX}/hexagon-v73/unsigned/libSnpeHtpV73Skel.so" REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
else()
	message("Default is selected")
	get_filename_component(SNPE_STUB_PATH "${SNPE_LIB_PREFIX}/aarch64-${_dtuple_POSTFIX}/libSnpeHtpV68Stub.so" REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
	get_filename_component(SNPE_SKEL_PATH "${SNPE_LIB_PREFIX}/hexagon-v68/unsigned/libSnpeHtpV68Skel.so" REALPATH BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
endif()

add_library(SNPE SHARED IMPORTED)
set_target_properties(SNPE PROPERTIES
	INTERFACE_INCLUDE_DIRECTORIES ${SNPE_INCLUDE_DIR}  
    IMPORTED_LOCATION ${SNPE_LIB_PATH}
)

include_directories(${SNPE_INCLUDE_DIR})
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${PYBIND11_INCLUDE_DIR})
link_directories(${PYTHON_LIB_DIR})

add_library(${APP} MODULE ${APP_SOURCES})

#target_compile_definitions(${APP} PUBLIC -DDEBUG)

target_link_libraries (${APP} PUBLIC SNPE)
# At runtime, resolve libSNPE.so from /usr/lib first (system skel handshake).
set_target_properties(${APP} PROPERTIES
	BUILD_RPATH "/usr/lib"
	INSTALL_RPATH "/usr/lib"
	BUILD_WITH_INSTALL_RPATH TRUE
)
