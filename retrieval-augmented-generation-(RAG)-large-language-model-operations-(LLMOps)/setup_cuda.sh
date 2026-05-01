#!/bin/bash

# Remove conflicting NVIDIA packages
sudo apt-get remove --purge libnvidia-extra-525 libnvidia-extra-550 libnvidia-fbc1-525 libnvidia-fbc1-550 -y

# update dependencies
sudo apt-get install -f -y
sudo apt update -f -y

# Clean up apt cache
sudo apt-get clean
sudo apt-get autoclean

# Python package cleanup
pip uninstall deepspeed -y
sudo rm -rf /usr/local/lib/python3.11/dist-packages/deepspeed*
