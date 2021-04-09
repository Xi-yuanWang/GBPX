cmake_minimum_required(VERSION 2.8.12)
project(GBP_utils)
add_subdirectory(pybind11)
pybind11_add_module(GBP_utils randomwalk.cpp)
