#
# Copyright (c) 2020-2021 UNITN, NYU
#
# This file is part of consim
# consim is free software: you can redistribute it
# and/or modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
# consim is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Lesser Public License for more details. You should have
# received a copy of the GNU Lesser General Public License along with
# consim If not, see
# <http://www.gnu.org/licenses/>.

# --- MACROS --- #
MACRO(SYMLINK_AND_INSTALL_HEADERS HEADERS SUBPATH)
  FOREACH(header ${HEADERS})
    GET_FILENAME_COMPONENT(headerName ${header} NAME)
    GET_FILENAME_COMPONENT(headerPath ${header} PATH)
    EXECUTE_PROCESS(COMMAND ${CMAKE_COMMAND} -E ${LINK}
      ${CMAKE_CURRENT_SOURCE_DIR}/${header}
      ${${PROJECT_NAME}_BINARY_DIR}/include/${PROJECT_NAME}/${SUBPATH}/${header})

    INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/${header}
      DESTINATION ${CMAKE_INSTALL_PREFIX}/include/${PROJECT_NAME}/${SUBPATH}/${headerPath}
            PERMISSIONS OWNER_READ GROUP_READ WORLD_READ OWNER_WRITE)
  ENDFOREACH(header)
ENDMACRO(SYMLINK_AND_INSTALL_HEADERS HEADERS SUBPATH)

FUNCTION(REMOVE_PATH_FROM_LIST list_name path_name dest_list)
  SET(list_name_)
  FOREACH(header ${list_name})
    STRING(REGEX REPLACE "${path_name}" "" header ${header}) 
    LIST(APPEND list_name_ ${header})
  ENDFOREACH(header ${list_name_})
  SET(${dest_list} ${list_name_} PARENT_SCOPE)
ENDFUNCTION(REMOVE_PATH_FROM_LIST)

FILE(GLOB ${PYWRAP}_SOURCES
    explicit_euler.cpp
    implicit_euler.cpp
    rk4.cpp
    base.cpp
    exponential.cpp
    rigid_euler.cpp
    bindings.cpp
    contacts.cpp
    stop_watch.cpp
)

# --- PYTHON TARGET --- #
ADD_CUSTOM_TARGET(python)
SET_TARGET_PROPERTIES(python PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD True)

SET(PYWRAP ${PROJECT_NAME}_pywrap)
SET(PYWRAP ${PYWRAP} PARENT_SCOPE)

# --- DEPENDENCIES --- #
SET(PKG_CONFIG_PYWRAP_REQUIRES "eigenpy >= 1.4.0")
SET(PKG_CONFIG_PYWRAP_REQUIRES "pinocchio >= 2.0.0")
SET(PKG_CONFIG_PYWRAP_REQUIRES "eiquadprog >= 1.2.0")
SET(PKG_CONFIG_PYWRAP_REQUIRES "expokit")
FOREACH(dep ${PKG_CONFIG_PYWRAP_REQUIRES})
  ADD_COMPILE_DEPENDENCY(${dep})
ENDFOREACH(dep ${PKG_CONFIG_PYWRAP_REQUIRES})

# --- LIBRARY --- #
SET(${PYWRAP}_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${PYTHON_SITELIB}/${PROJECT_NAME})

FILE(GLOB_RECURSE ${PROJECT_NAME}_PYTHON_HEADERS
     ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
     ) 

FILE(GLOB_RECURSE ${PROJECT_NAME}_PYTHON_SOURCES
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp 
     ) 

REMOVE_PATH_FROM_LIST("${${PROJECT_NAME}_PYTHON_HEADERS}" "${CMAKE_CURRENT_SOURCE_DIR}/" ${PROJECT_NAME}_PYTHON_HEADERS)
REMOVE_PATH_FROM_LIST("${${PROJECT_NAME}_PYTHON_SOURCES}" "${CMAKE_CURRENT_SOURCE_DIR}/" ${PROJECT_NAME}_PYTHON_SOURCES)

LIST(APPEND HEADERS ${${PROJECT_NAME}_PYTHON_HEADERS})

# Headers of the python binding
MAKE_DIRECTORY("${${PROJECT_NAME}_BINARY_DIR}/include/consim/bindings/python")

SYMLINK_AND_INSTALL_HEADERS("${${PROJECT_NAME}_PYTHON_HEADERS}" "bindings/python")

# --- COMPILE WRAPPER
MAKE_DIRECTORY("${${PROJECT_NAME}_BINARY_DIR}/bindings/python/${PROJECT_NAME}")
# SET(${PYWRAP}_SOURCES ${${PROJECT_NAME}_PYTHON_SOURCES})
# SET(${PYWRAP}_HEADERS ${${PROJECT_NAME}_PYTHON_HEADERS})

ADD_LIBRARY(${PYWRAP} SHARED ${${PYWRAP}_SOURCES} ${${PYWRAP}_HEADERS})
ADD_DEPENDENCIES(python ${PYWRAP})

IF(BUILD_WITH_COMMIT_VERSION)
  TAG_LIBRARY_VERSION(${PYWRAP})
ENDIF(BUILD_WITH_COMMIT_VERSION)
ADD_HEADER_GROUP(${PYWRAP}_HEADERS)
ADD_SOURCE_GROUP(${PYWRAP}_SOURCES)
PKG_CONFIG_USE_DEPENDENCY(${PYWRAP} eigenpy)
PKG_CONFIG_USE_DEPENDENCY(${PYWRAP} pinocchio)
PKG_CONFIG_USE_DEPENDENCY(${PYWRAP} eiquadprog)
PKG_CONFIG_USE_DEPENDENCY(${PYWRAP} expokit)

TARGET_LINK_LIBRARIES(${PYWRAP} ${PROJECT_NAME})
TARGET_LINK_BOOST_PYTHON(${PYWRAP})

IF(APPLE)
  # We need to change the extension for python bindings
  SET_TARGET_PROPERTIES(${PYWRAP} PROPERTIES SUFFIX ".so")
ENDIF(APPLE)

SET_TARGET_PROPERTIES(${PYWRAP} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bindings/python/${PROJECT_NAME}")

INSTALL(TARGETS ${PYWRAP} DESTINATION ${${PYWRAP}_INSTALL_DIR})

# --- INSTALL SCRIPTS 
SET(PYTHON_FILES
  __init__.py
  )

FOREACH(python ${PYTHON_FILES})
  INSTALL(FILES
    "${${PROJECT_NAME}_SOURCE_DIR}/bindings/python/${python}"
    DESTINATION ${${PYWRAP}_INSTALL_DIR})
ENDFOREACH(python)
