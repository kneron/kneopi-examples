#/bin/bash

cd package

pip uninstall -y KneronPLUS_YOLO-3.0.0-py3-none-any.whl --break-system-packages --quiet 2>/dev/null
pip uninstall -y KneronPLUS-3.0.0-py3-none-any.whl --break-system-packages --quiet 2>/dev/null

pip install KneronPLUS-3.0.0-py3-none-any.whl --break-system-packages
pip install KneronPLUS_YOLO-3.0.0-py3-none-any.whl --break-system-packages

