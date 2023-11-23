cmake_minimum_required(VERSION 3.1)
 
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 指定spdlog库的路径，这里假设spdlog库已经安装在指定目录下
set(SPDLOG_INCLUDE_DIR "~/project/matrixlib_test/include")

link_directories("/usr/local/lib")
include_directories("/usr/local/include")
 
project(matrixlib_test LANGUAGES CXX)

find_package(OsqpEigen)
find_package(Eigen3)
 
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
 
#MPCExample
add_executable(osqp_test1 src/osqp/osqp_test1.cpp)
 
target_link_libraries(osqp_test1 OsqpEigen::OsqpEigen)

add_executable(osqp-eigen_test1 src/osqp-eigen/osqp-eigen_test1.cpp)
 
target_link_libraries(osqp-eigen_test1 OsqpEigen::OsqpEigen)

add_executable(mpc src/osqp/mpc.cpp)
 
target_link_libraries(mpc 
                      OsqpEigen::OsqpEigen
                      libqpOASES.a)
target_include_directories(mpc PRIVATE ${SPDLOG_INCLUDE_DIR})

add_executable(test src/osqp/test.cpp)
 
target_link_libraries(test 
                      OsqpEigen::OsqpEigen
                      libqpOASES.a)

add_executable(test2 src/osqp/test2.cpp)
 
target_link_libraries(test2 OsqpEigen::OsqpEigen)