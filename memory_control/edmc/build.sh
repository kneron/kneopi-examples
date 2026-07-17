#!/bin/bash

rm -rf build
rm -rf bin
mkdir build

cd build
cmake ../ || exit 1
make || exit 1

cp -rf "./bin" "../"
cd ..
rm -rf build
