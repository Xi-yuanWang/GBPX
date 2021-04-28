cmake_minimum_required(VERSION 2.8.12)
project(PPRPush)
add_subdirectory(pybind11)
pybind11_add_module(PPRPush push.cpp)
