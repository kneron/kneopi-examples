#!/bin/bash

BGREEN='\033[1;32m'
BWHITE='\033[1;37m'
NC='\033[0m'

BUILD_ARR=("none" "scan_devices" "kl730_demo_generic_data_inference" "kl730_demo_generic_image_inference" "All basic examples" "kl730_demo_cam_generic_image_inference_drop_frame" "All examples")

rm -rf build
rm -rf bin
mkdir build
mkdir bin

if [ $1 = "-a" ]
then
	BUILD_NUM=6
else
	echo ""
	echo "Please select the example(s) to be build:"
	echo "---------------- Basic Examples ----------------"
	echo "[1] scan_devices"
	echo "[2] kl730_demo_generic_data_inference"
	echo "[3] kl730_demo_generic_image_inference"
	echo "[4] All basic examples"
	echo "---------------- OpenCV Example ----------------"
	echo "[5] kl730_demo_cam_generic_image_inference_drop_frame"
	echo "---------------- All Examples ------------------"
	echo "[6] All examples"
	echo -e "$BWHITE"
	read -p "Please enter 1-6: " BUILD_NUM
	echo -e "$NC"
fi

case $BUILD_NUM in
	"1"|"2"|"3"|"5")
		echo -e "\n$BGREEN[BUILD] ${BUILD_ARR[$BUILD_NUM]}$NC\n"
		rm -rf build
		mkdir build
		cd build
		cmake "../${BUILD_ARR[$BUILD_NUM]}" || exit 1
		make || exit 1
		mv ${BUILD_ARR[$BUILD_NUM]} "../bin"
		cd ".."
		;;
	"4")
		for i in {1..3}
		do
			echo -e "\n$BGREEN[BUILD] ${BUILD_ARR[$i]}$NC\n"
			rm -rf build
			mkdir build
			cd build
			cmake "../${BUILD_ARR[$i]}" || exit 1
			make || exit 1
			mv ${BUILD_ARR[$i]} "../bin"
			cd ".."
		done
		;;
	"6")
		for i in {1..3}
		do
			echo -e "\n$BGREEN[BUILD] ${BUILD_ARR[$i]}$NC\n"
			rm -rf build
			mkdir build
			cd build
			cmake "../${BUILD_ARR[$i]}" || exit 1
			make || exit 1
			mv ${BUILD_ARR[$i]} "../bin"
			cd ".."
		done

		echo -e "\n$BGREEN[BUILD] ${BUILD_ARR[5]}$NC\n"
		rm -rf build
		mkdir build
		cd build
		cmake "../${BUILD_ARR[5]}" || exit 1
		make || exit 1
		mv ${BUILD_ARR[5]} "../bin"
		cd ".."
		;;
	*)
		echo "The input is not valid"
		exit 1
		;;
esac

rm -rf build
