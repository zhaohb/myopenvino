# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(ProcessorCount)

#
# Disables deprecated warnings generation
# Defines ie_c_cxx_deprecated varaible which contains C / C++ compiler flags
#
macro(disable_deprecated_warnings)
    if(WIN32)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(ie_c_cxx_deprecated "/Qdiag-disable:1478,1786")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set(ie_c_cxx_deprecated "/wd4996")
        endif()
    else()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(ie_c_cxx_deprecated "-diag-disable=1478,1786")
        else()
            set(ie_c_cxx_deprecated "-Wno-deprecated-declarations")
        endif()
    endif()

    if(NOT ie_c_cxx_deprecated)
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ie_c_cxx_deprecated}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ie_c_cxx_deprecated}")
endmacro()

#
# Don't threat deprecated warnings as errors
# Defines ie_c_cxx_deprecated_no_errors varaible which contains C / C++ compiler flags
#
macro(ie_deprecated_no_errors)
    if(WIN32)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(ie_c_cxx_deprecated "/Qdiag-warning:1478,1786")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set(ie_c_cxx_deprecated "/wd4996")
        endif()
    else()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(ie_c_cxx_deprecated_no_errors "-diag-warning=1478,1786")
        else()
            set(ie_c_cxx_deprecated_no_errors "-Wno-error=deprecated-declarations")
        endif()

        if(NOT ie_c_cxx_deprecated_no_errors)
            message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
        endif()
    endif()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${ie_c_cxx_deprecated_no_errors}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${ie_c_cxx_deprecated_no_errors}")
endmacro()

#
# Provides SSE4.2 compilation flags depending on an OS and a compiler
#
function(ie_sse42_optimization_flags flags)
    if(WIN32)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            # No such option for MSVC 2019
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(${flags} "/arch:SSE4.2 /QxSSE4.2" PARENT_SCOPE)
        else()
            message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
        endif()
    else()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(${flags} "-msse4.2 -xSSE4.2" PARENT_SCOPE)
        else()
            set(${flags} "-msse4.2" PARENT_SCOPE)
        endif()
    endif()
endfunction()

#
# Provides AVX2 compilation flags depending on an OS and a compiler
#
function(ie_avx2_optimization_flags flags)
    if(WIN32)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(${flags} "/QxCORE-AVX2" PARENT_SCOPE)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set(${flags} "/arch:AVX2" PARENT_SCOPE)
        else()
            message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
        endif()
    else()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(${flags} "-march=core-avx2 -xCORE-AVX2 -mtune=core-avx2" PARENT_SCOPE)
        else()
            set(${flags} "-mavx2 -mfma" PARENT_SCOPE)
        endif()
    endif()
endfunction()

#
# Provides common AVX512 compilation flags for AVX512F instruction set support
# depending on an OS and a compiler
#
function(ie_avx512_optimization_flags flags)
    if(WIN32)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(${flags} "/QxCOMMON-AVX512" PARENT_SCOPE)
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            set(${flags} "/arch:AVX512" PARENT_SCOPE)
        else()
            message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
        endif()
    else()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            set(${flags} "-xCOMMON-AVX512" PARENT_SCOPE)
        endif()
        if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
            set(${flags} "-mavx512f -mfma" PARENT_SCOPE)
        endif()
        if(CMAKE_CXX_COMPILER_ID MATCHES "^(Clang|AppleClang)$")
            set(${flags} "-mavx512f -mfma" PARENT_SCOPE)
        endif()
    endif()
endfunction()

function(ie_arm_neon_optimization_flags flags)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        message(WARNING "Unsupported CXX compiler ${CMAKE_CXX_COMPILER_ID}")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # nothing
    elseif(ANDROID)
        if(ANDROID_ABI STREQUAL "arm64-v8a")
            set(${flags} "-mfpu=neon" PARENT_SCOPE)
        elseif(ANDROID_ABI STREQUAL "armeabi-v7a-hard with NEON")
            set(${flags} "-march=armv7-a -mfloat-abi=hard -mhard-float -D_NDK_MATH_NO_SOFTFP=1 -mfpu=neon" PARENT_SCOPE)
        elseif((ANDROID_ABI STREQUAL "armeabi-v7a with NEON") OR
               (ANDROID_ABI STREQUAL "armeabi-v7a" AND
                DEFINED CMAKE_ANDROID_ARM_NEON AND CMAKE_ANDROID_ARM_NEON))
            set(${flags} "-march=armv7-a -mfloat-abi=softfp -mfpu=neon" PARENT_SCOPE)
        endif()
    else()
        if(AARCH64)
            set(${flags} "-O2 -ftree-vectorize" PARENT_SCOPE)
        elseif(ARM)
            set(${flags} "-mfpu=neon" PARENT_SCOPE)
        endif()
    endif()
endfunction()

#
# Enables Link Time Optimization compilation
#
macro(ie_enable_lto)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endmacro()

#
# Adds compiler flags to C / C++ sources
#
macro(ie_add_compiler_flags)
    foreach(flag ${ARGN})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${flag}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${flag}")
    endforeach()
endmacro()

#
# Forced includes certain header file to all target source files
#
function(ov_force_include target scope header_file)
    if(MSVC)
        target_compile_options(${target} ${scope} /FI"${header_file}")
    else()
        target_compile_options(${target} ${scope} -include "${header_file}")
    endif()
endfunction()

#
# Compilation and linker flags
#

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)

# to allows to override CMAKE_CXX_STANDARD from command line
if(NOT DEFINED CMAKE_CXX_STANDARD)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(CMAKE_CXX_STANDARD 14)
    else()
        set(CMAKE_CXX_STANDARD 11)
    endif()
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(ENABLE_COVERAGE)
    ie_add_compiler_flags(--coverage)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif()

if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    ie_add_compiler_flags(-fsigned-char)
endif()

# Honor visibility properties for all target types
set(CMAKE_POLICY_DEFAULT_CMP0063 NEW)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

if(WIN32)
    ie_add_compiler_flags(-D_CRT_SECURE_NO_WARNINGS -D_SCL_SECURE_NO_WARNINGS)
    ie_add_compiler_flags(/EHsc) # no asynchronous structured exception handling
    ie_add_compiler_flags(/Gy) # remove unreferenced functions: function level linking
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LARGEADDRESSAWARE")

    if (TREAT_WARNING_AS_ERROR)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
            ie_add_compiler_flags(/WX)
            ie_add_compiler_flags(/Qdiag-warning:47,1740,1786)
        elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
           # ie_add_compiler_flags(/WX) # Too many warnings
        endif()
    endif()

    # Compiler specific flags

    ie_add_compiler_flags(/bigobj)
    ie_add_compiler_flags(/MP)

    # Disable noisy warnings

    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        # C4251 needs to have dll-interface to be used by clients of class
        ie_add_compiler_flags(/wd4251)
        # C4275 non dll-interface class used as base for dll-interface class
        ie_add_compiler_flags(/wd4275)
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        # 161: unrecognized pragma
        # 177: variable was declared but never referenced
        # 556: not matched type of assigned function pointer
        # 1744: field of class type without a DLL interface used in a class with a DLL interface
        # 1879: unimplemented pragma ignored
        # 2586: decorated name length exceeded, name was truncated
        # 2651: attribute does not apply to any entity
        # 3180: unrecognized OpenMP pragma
        # 11075: To get full report use -Qopt-report:4 -Qopt-report-phase ipo
        # 15335: was not vectorized: vectorization possible but seems inefficient. Use vector always directive or /Qvec-threshold0 to override
        ie_add_compiler_flags(/Qdiag-disable:161,177,556,1744,1879,2586,2651,3180,11075,15335)
    endif()

    # Debug information flags, by default CMake adds /Zi option
    # but provides no way to specify CMAKE_COMPILE_PDB_NAME on root level
    # In order to avoid issues with ninja we are replacing default flag instead of having two of them
    # and observing warning D9025 about flag override
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REPLACE "/Zi" "/Z7" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
    string(REPLACE "/Zi" "/Z7" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
else()
    # TODO: enable for C sources as well
    # ie_add_compiler_flags(-Werror)
    if(TREAT_WARNING_AS_ERROR)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
    endif()

    ie_add_compiler_flags(-ffunction-sections -fdata-sections)
    ie_add_compiler_flags(-fdiagnostics-show-option)
    ie_add_compiler_flags(-Wundef)
    ie_add_compiler_flags(-Wreturn-type)
    ie_add_compiler_flags(-Wunused-variable)

    # Disable noisy warnings

    if (CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        ie_add_compiler_flags(-Wswitch)
    elseif(UNIX)
        ie_add_compiler_flags(-Wuninitialized -Winit-self)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            ie_add_compiler_flags(-Wno-error=switch)
        else()
            ie_add_compiler_flags(-Wmaybe-uninitialized)
        endif()
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        ie_add_compiler_flags(-diag-disable=remark)
        # noisy warnings from Intel Compiler 19.1.1.217 20200306
        ie_add_compiler_flags(-diag-disable=2196)
    endif()

    # Linker flags

    if(APPLE)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-dead_strip")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-dead_strip")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-dead_strip")
    elseif(LINUX)
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections -Wl,--exclude-libs,ALL")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--gc-sections -Wl,--exclude-libs,ALL")
    endif()
endif()
