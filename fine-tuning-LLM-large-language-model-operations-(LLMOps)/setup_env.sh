#!/bin/bash
# Environment setup script for CUDA and dependencies

echo "Setting up CUDA environment..."

# Remove conflicting NVIDIA packages
echo "Removing conflicting NVIDIA packages..."
sudo apt-get remove --purge libnvidia-extra-525 libnvidia-extra-550 libnvidia-fbc1-525 libnvidia-fbc1-550 -y 2>/dev/null

# Fix broken dependencies
echo "Fixing broken dependencies..."
sudo apt-get install -f -y
sudo apt update -f -y

# Clean up apt cache
echo "Cleaning apt cache..."
sudo apt-get clean
sudo apt-get autoclean

# Python package cleanup
echo "Cleaning Python packages..."
pip uninstall deepspeed -y 2>/dev/null
sudo rm -rf /usr/local/lib/python3.11/dist-packages/deepspeed* 2>/dev/null

echo "Environment setup completed!"
echo "CUDA version:"
nvidia-smi | grep "CUDA Version" || echo "CUDA not found"