#!/bin/bash

rm -rf build
rm -rf bin
mkdir build
mkdir bin

cd build
cmake "../" || exit 1
make || exit 1
cp -rf "./bin" "../"
cd ".."