# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(CheckCXXCompilerFlag)

if (ENABLE_SANITIZER OR ENABLE_THREAD_SANITIZER)
    # This is workaround for https://gitlab.kitware.com/cmake/cmake/-/issues/16609.
    # It ensures pthread is searched without ASAN linking.
    # Line bellow must be before adding -fsanitize=address or -fsanitize=thread to
    # build options for the trick to work.
    find_package(Threads REQUIRED)
endif()

if (ENABLE_SANITIZER)
    set(SANITIZER_COMPILER_FLAGS "-g -fsanitize=address -fno-omit-frame-pointer")
    CHECK_CXX_COMPILER_FLAG("-fsanitize-recover=address" SANITIZE_RECOVER_SUPPORTED)
    if (SANITIZE_RECOVER_SUPPORTED)
        set(SANITIZER_COMPILER_FLAGS "${SANITIZER_COMPILER_FLAGS} -fsanitize-recover=address")
    endif()

    set(SANITIZER_LINKER_FLAGS "-fsanitize=address")
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -fuse-ld=gold")
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$" AND NOT WIN32)
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.0)
            set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -fuse-ld=lld")
        endif()
    endif()

    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SANITIZER_COMPILER_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_COMPILER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SANITIZER_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${SANITIZER_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_LINKER_FLAGS}")
endif()

if (ENABLE_THREAD_SANITIZER)
    set(SANITIZER_COMPILER_FLAGS "-g -fsanitize=thread -fno-omit-frame-pointer")
    set(SANITIZER_LINKER_FLAGS "-fsanitize=thread")
    if(CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$" AND NOT WIN32)
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 8.0)
            set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -fuse-ld=lld")
        else()
            set(SANITIZER_LINKER_FLAGS "${SANITIZER_LINKER_FLAGS} -static-libsan")
        endif()
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${SANITIZER_COMPILER_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_COMPILER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${SANITIZER_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${SANITIZER_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${SANITIZER_LINKER_FLAGS}")
endif()
