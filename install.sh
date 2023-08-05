#!/bin/bash

# Get system information
OS=$(uname -s)

# Check if system is Linux
if [ "${OS}" = "Linux" ]; then
    echo "This is a Linux system."
    # Check if NVIDIA graphics card is installed
    if lspci | grep -i nvidia > /dev/null; then
        echo "NVIDIA graphics card is installed."
        pip install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
        pip install torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
        pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13/index.html
    else
        echo "NVIDIA graphics card is not installed."
        pip install torch==1.13.0 -f https://download.pytorch.org/whl/torch_stable.html
        pip install torchvision==0.14.0 -f https://download.pytorch.org/whl/torch_stable.html
        pip install mmcv-full==1.7.0
    fi
# Check if system is macOS
elif [ "${OS}" = "Darwin" ]; then
    echo "This is a macOS system."
    pip install torch==1.13.0
    pip install torchvision==0.14.0
    pip install mmcv-full==1.7.0

# Check if system is Windows
elif [ "${OS}" = "Windows_NT" ]; then
    echo "This is a Windows system, please customize the installation of the code base."
else
    echo "Unknown system"
fi

pip install git+https://github.com/deepmind/surface-distance.git
pip install -e .
