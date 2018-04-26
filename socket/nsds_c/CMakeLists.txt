cmake_minimum_required(VERSION 3.1.10)

project(nsds_client)

set(NSDS_CLIENT_VERSION 1.0)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY $(CMAKE_BINARY_DIR)/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $(CMAKE_BINARY_DIR)/bin)

#SET(CMAKE_BUILE_TYPE DEBUG)

#find Qt
unset(QT_QTCORE_LIBRARY CACHE)
unset(QT_QTNETWORK_LIBRARY CACHE)


if(NOT QT_QMAKE_EXECUTABLE)
   find_program(QT_QMAKE_EXECUTABLE NAMES "qmake" "qmake-qt5" "qmake.exe")
endif()
if(NOT QT_QMAKE_EXECUTABLE)
    message(FATAL_ERROR "you have to set the path to the Qt5 qmake executable")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
get_filename_component(QT_QMAKE_PATH ${QT_QMAKE_EXECUTABLE} PATH)

set(QT_ROOT ${QT_QMAKE_PATH}/)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${QT_QMAKE_PATH}\\..\\lib\\cmake\\Qt5)

unset(Qt5Network_FOUND CACHE)
find_package(Qt5 REQUIRED Core Network)

set(QT_INCLUDES
        ${Qt5Core_INCLUDE_DIRS}
        ${Qt5Network_INCLUDE_DIRS}
        )

# gather information for building
include_directories (BEFORE SYSTEM
    ${QT_INCLUDES}
 )
include_directories (
    ${CMAKE_CURRENT_BINARY_DIR}
)

file(GLOB NSDS_CLIENT_SOURCES "*.cpp")
file(GLOB NSDS_CLIENT_HEADERS "*.h")

file(COPY ${NSDS_CLIENT_SOURCES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${NSDS_CLIENT_HEADERS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

#qt5_use_modules(${DLL_CORE_NAME} Core Network)
#set_target_properties(${DLL_CORE_NAME} PROPERTIES DEBUG_OUTPUT_NAME ${DLL_CORE_NAME}d)
#set_target_properties(${DLL_CORE_NAME} PROPERTIES RELEASE_OUTPUT_NAME ${DLL_CORE_NAME})

#SET(SOURCE_FILES main.cpp filesender.cpp)
#find_package(Qt5Network)
#find_package(Qt5Core)
#install(FILES "${SOURCE_FILES}" DESTINATION "${CMAKE_BINARY_DIR}")

add_executable(${PROJECT_NAME} ${NSDS_CLIENT_SOURCES})
target_link_libraries(${PROJECT_NAME} Qt5::Core Qt5::Network)
