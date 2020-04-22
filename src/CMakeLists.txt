#
# Copyright (c) 2017-2018 CNRS
#
# This file is part of tsid
# tsid is free software: you can redistribute it
# and/or modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
# tsid is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Lesser Public License for more details. You should have
# received a copy of the GNU Lesser General Public License along with
# tsid If not, see
# <http://www.gnu.org/licenses/>.

# --- MAIN LIBRARY --- #
SET(LIBRARY_NAME ${PROJECT_NAME})

SET(${LIBRARY_NAME}_MATH_SOURCES
    math/utils.cpp
  )

SET(${LIBRARY_NAME}_SOLVERS_SOURCES
    solvers/eiquadprog-fast.cpp
    solvers/solver-HQP-base.cpp
    solvers/solver-HQP-factory.cpp
    solvers/solver-HQP-eiquadprog.cpp
    solvers/solver-HQP-eiquadprog-fast.cpp
    solvers/solver-HQP-qpoases.cpp
    solvers/utils.cpp
  )

SET(${LIBRARY_NAME}_ROBOTS_SOURCES
    robots/robot-wrapper.cpp
  )

SET(${LIBRARY_NAME}_SOURCES
    contact.cpp
    #object.cpp
    #simulator.cpp
#    utils/statistics.cpp
    utils/stop-watch.cpp
#    ${${LIBRARY_NAME}_MATH_SOURCES}
#    ${${LIBRARY_NAME}_SOLVERS_SOURCES}
#    ${${LIBRARY_NAME}_ROBOTS_SOURCES}
  )

ADD_LIBRARY(${LIBRARY_NAME} SHARED ${HEADERS_FULL_PATH} ${${LIBRARY_NAME}_SOURCES})
SET_TARGET_PROPERTIES(${LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE CXX)
ADD_HEADER_GROUP(HEADERS_FULL_PATH)
ADD_SOURCE_GROUP(${LIBRARY_NAME}_SOURCES)

PKG_CONFIG_USE_DEPENDENCY(${LIBRARY_NAME} eigen3)
PKG_CONFIG_USE_DEPENDENCY(${LIBRARY_NAME} pinocchio)
PKG_CONFIG_USE_DEPENDENCY(${LIBRARY_NAME} expokit)

INSTALL(TARGETS ${PROJECT_NAME} DESTINATION lib)

