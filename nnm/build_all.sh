#!/bin/bash

BGREEN='\033[1;32m'
BWHITE='\033[1;37m'
NC='\033[0m'

BUILD_ARR=("none" "example_nnm_image" "example_nnm_webcam" "example_nnm_sensor" "example_nnm_rtsp" "All examples")

rm -rf build
rm -rf bin
mkdir build
mkdir bin

if [ "$1" = "-a" ]
then
	BUILD_NUM=5
else
	echo ""
	echo "Please select the example(s) to be build:"
	echo "---------------- Basic Examples ----------------"
	echo "[1] example_nnm_image"
	echo "---------------- OpenCV Example ----------------"
	echo "[2] example_nnm_webcam"
	echo "[3] example_nnm_sensor"
	echo "[4] example_nnm_rtsp"
	echo "---------------- All Examples ------------------"
	echo "[5] All examples"
	echo -e "$BWHITE"
	read -p "Please enter 1-5: " BUILD_NUM
	echo -e "$NC"
fi

case $BUILD_NUM in
	"1"|"2"|"3"|"4")
		echo -e "\n$BGREEN[BUILD] ${BUILD_ARR[$BUILD_NUM]}$NC\n"
		rm -rf build
		mkdir build
		cd build
		cmake "../${BUILD_ARR[$BUILD_NUM]}" || exit 1
		make || exit 1
		cp -rf "./bin" "../"
		cd ".."
		;;
	"5")
		for i in {1..4}
		do
			echo -e "\n$BGREEN[BUILD] ${BUILD_ARR[$i]}$NC\n"
			rm -rf build
			mkdir build
			cd build
			cmake "../${BUILD_ARR[$i]}" || exit 1
			make || exit 1
			cp -rf "./bin" "../"
			cd ".."
		done
		;;
	*)
		echo "The input is not valid"
		exit 1
		;;
esac

rm -rf build
