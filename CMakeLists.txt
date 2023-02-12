cmake_minimum_required(VERSION 3.10)

project(jinx)

find_package(pybind11 REQUIRED)

pybind11_add_module(example example.cpp)
