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

# --- MAIN LIBRARY --- #
SET(LIBRARY_NAME ${PROJECT_NAME})

SET(${LIBRARY_NAME}_SOURCES
    contact.cpp
    object.cpp
    simulators/common.cpp
    simulators/base.cpp
    simulators/explicit_euler.cpp
    simulators/rk4.cpp
    simulators/implicit_euler.cpp
    simulators/exponential.cpp
    # utils/stop-watch.cpp
  )

ADD_LIBRARY(${LIBRARY_NAME} SHARED ${HEADERS_FULL_PATH} ${${LIBRARY_NAME}_SOURCES})
SET_TARGET_PROPERTIES(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
ADD_HEADER_GROUP(HEADERS_FULL_PATH)
ADD_SOURCE_GROUP(${LIBRARY_NAME}_SOURCES)

PKG_CONFIG_USE_DEPENDENCY(${LIBRARY_NAME} eigen3)
PKG_CONFIG_USE_DEPENDENCY(${LIBRARY_NAME} pinocchio)
PKG_CONFIG_USE_DEPENDENCY(${LIBRARY_NAME} expokit)
PKG_CONFIG_USE_DEPENDENCY(${LIBRARY_NAME} eiquadprog)

INSTALL(TARGETS ${PROJECT_NAME} DESTINATION lib)

