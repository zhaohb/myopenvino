# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

if(WIN32)
    set(PROGRAMFILES_ENV "ProgramFiles(X86)")
    file(TO_CMAKE_PATH $ENV{${PROGRAMFILES_ENV}} PROGRAMFILES)
    set(UWP_SDK_PATH "${PROGRAMFILES}/Windows Kits/10/bin/${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION}/x64")

    message(STATUS "Trying to find apivalidator in: ${UWP_SDK_PATH}")
    find_host_program(UWP_API_VALIDATOR
                      NAMES apivalidator
                      PATHS "${UWP_SDK_PATH}"
                      DOC "ApiValidator for UWP compliance")

    if(UWP_API_VALIDATOR)
        message(STATUS "Found apivalidator: ${UWP_API_VALIDATOR}")
    endif()
endif()

function(_ie_add_api_validator_post_build_step_recursive)
    cmake_parse_arguments(API_VALIDATOR "" "TARGET" "" ${ARGN})

    list(APPEND API_VALIDATOR_TARGETS ${API_VALIDATOR_TARGET})
    set(API_VALIDATOR_TARGETS ${API_VALIDATOR_TARGETS} PARENT_SCOPE)

    get_target_property(IS_IMPORTED ${API_VALIDATOR_TARGET} IMPORTED)
    if(IS_IMPORTED)
        return()
    endif()

    get_target_property(LIBRARY_TYPE ${API_VALIDATOR_TARGET} TYPE)
    if(LIBRARY_TYPE STREQUAL "EXECUTABLE" OR LIBRARY_TYPE STREQUAL "SHARED_LIBRARY")
        get_target_property(LINKED_LIBRARIES ${API_VALIDATOR_TARGET} LINK_LIBRARIES)
        if(LINKED_LIBRARIES)
            foreach(ITEM IN LISTS LINKED_LIBRARIES)
                if(NOT TARGET ${ITEM})
                    continue()
                endif()
                get_target_property(LIBRARY_TYPE_DEPENDENCY ${ITEM} TYPE)
                if(LIBRARY_TYPE_DEPENDENCY STREQUAL "SHARED_LIBRARY")
                    _ie_add_api_validator_post_build_step_recursive(TARGET ${ITEM})
                endif()
            endforeach()
        endif()
    endif()

    set(API_VALIDATOR_TARGETS ${API_VALIDATOR_TARGETS} PARENT_SCOPE)
endfunction()

set(VALIDATED_LIBRARIES "" CACHE INTERNAL "")

function(_ie_add_api_validator_post_build_step)
    set(UWP_API_VALIDATOR_APIS "${PROGRAMFILES}/Windows Kits/10/build/universalDDIs/x64/UniversalDDIs.xml")
    set(UWP_API_VALIDATOR_EXCLUSION "${UWP_SDK_PATH}/BinaryExclusionlist.xml")

    if((NOT UWP_API_VALIDATOR) OR (WINDOWS_STORE OR WINDOWS_PHONE))
        return()
    endif()

    cmake_parse_arguments(API_VALIDATOR "" "TARGET" "" ${ARGN})

    if(NOT API_VALIDATOR_TARGET)
        message(FATAL_ERROR "RunApiValidator requires TARGET to validate!")
    endif()

    if(NOT TARGET ${API_VALIDATOR_TARGET})
        message(FATAL_ERROR "${API_VALIDATOR_TARGET} is not a TARGET in the project tree.")
    endif()

    # collect targets

    _ie_add_api_validator_post_build_step_recursive(TARGET ${API_VALIDATOR_TARGET})

    # remove targets which were tested before

    foreach(item IN LISTS VALIDATED_LIBRARIES)
        list(REMOVE_ITEM API_VALIDATOR_TARGETS ${item})
    endforeach()

    list(REMOVE_DUPLICATES API_VALIDATOR_TARGETS)

    if(NOT API_VALIDATOR_TARGETS)
        return()
    endif()

    # apply check

    macro(api_validator_get_target_name)
        get_target_property(IS_IMPORTED ${target} IMPORTED)
        if(IS_IMPORTED)
            get_target_property(target_location ${target} LOCATION)  
            get_filename_component(target_name "${target_location}" NAME_WE)
        else()
            set(target_name ${target})
        endif()
    endmacro()

    foreach(target IN LISTS API_VALIDATOR_TARGETS)
        api_validator_get_target_name()
        set(output_file "${CMAKE_BINARY_DIR}/api_validator/${target_name}.txt")

        add_custom_command(TARGET ${API_VALIDATOR_TARGET} POST_BUILD
            COMMAND ${CMAKE_COMMAND}
                -D UWP_API_VALIDATOR=${UWP_API_VALIDATOR}
                -D UWP_API_VALIDATOR_TARGET=$<TARGET_FILE:${target}>
                -D UWP_API_VALIDATOR_APIS=${UWP_API_VALIDATOR_APIS}
                -D UWP_API_VALIDATOR_EXCLUSION=${UWP_API_VALIDATOR_EXCLUSION}
                -D UWP_API_VALIDATOR_OUTPUT=${output_file}
                -D CMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
                -P "${IEDevScripts_DIR}/api_validator/api_validator_run.cmake"
            BYPRODUCTS ${output_file}
            COMMENT "[apiValidator] Check ${target_name} for OneCore compliance"
            VERBATIM)
    endforeach()

    # update list of validated libraries

    list(APPEND VALIDATED_LIBRARIES ${API_VALIDATOR_TARGETS})
    set(VALIDATED_LIBRARIES "${VALIDATED_LIBRARIES}" CACHE INTERNAL "" FORCE)
endfunction()

#
# ie_add_api_validator_post_build_step(TARGET <name>)
#
macro(ie_add_api_validator_post_build_step)
    _ie_add_api_validator_post_build_step(${ARGV})
endmacro()
