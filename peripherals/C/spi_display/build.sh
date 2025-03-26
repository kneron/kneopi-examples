#!/bin/bash

mkdir build
cd build
cmake ..
make -j

# change the executiton path to access example images
cp bin/* ../
