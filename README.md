# TensorFlow GPU Installation Guide for ASUS ROG GL 552V (GTX 960M)

This guide will help you install TensorFlow with GPU support on Windows Subsystem for Linux (Ubuntu 20.04) with Python 3.8.

## Prerequisites
- Windows 11 with WSL2 Ubuntu 20.04 installed
- NVIDIA GeForce GTX 960M
- Python 3.8 (default on Ubuntu 20.04)

## Step 0: System Preparation
```bash
# Update package list and upgrade existing packages
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y wget build-essential python3-dev python3-pip python3-venv

# Restart WSL to ensure all updates are applied
wsl.exe --shutdown # Run this from Windows PowerShell 
# Then restart your WSL terminal
```

## Step 1: Create and Activate Virtual Environment
```bash
# Create a new virtual environment
python3 -m venv ~/tensorflow-gpu

# Activate the virtual environment
source ~/tensorflow-gpu/bin/activate

# Upgrade pip
pip install --upgrade pip
```

## Step 2: Remove Existing CUDA Installation
```bash
# Remove old CUDA installation
sudo apt remove --purge nvidia-cuda-toolkit
sudo apt autoremove
sudo apt clean
```

## Step 3: Install CUDA 11.8
```bash
# Download CUDA 11.8 installer
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb

# Install CUDA repository
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/

# Update package list and install CUDA
sudo apt update
sudo apt install -y cuda-11-8
```

## Step 4: Configure CUDA Environment
```bash
# Add CUDA paths to .bashrc
echo 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

## Step 5: Install cuDNN 8.6

1. Register/Login at NVIDIA Developer website:
    - Go to [https://developer.nvidia.com/cudnn](https://developer.nvidia.com/cudnn)
    - Create an account or login if you already have one
    - Accept the terms and conditions
2.  Download cuDNN v8.6.0 for CUDA 11.x:
    - Find "Download cuDNN v8.6.0 (October 3rd, 2022), for CUDA 11.x"
    - Download "Local Installer for Ubuntu20.04 x86_64 (Deb)"
    - The file should be named similar to: `cudnn-local-repo-ubuntu2004-8.6.0.163_1.0-1_amd64.deb`
3.  Install cuDNN:
```bash
# Install the downloaded package (replace with your actual downloaded file path)
# After downloading, replace <filename> with actual filename: # sudo dpkg -i <filename>.deb
sudo dpkg -i cudnn-local-repo-ubuntu2004-8.6.0.163_1.0-1_amd64.deb

# Copy the repository key
sudo cp /var/cudnn-local-repo-ubuntu2004-8.6.0.163/cudnn-local-*-keyring.gpg /usr/share/keyrings/

# Update package manager
sudo apt update

# Install cuDNN packages
sudo apt install -y libcudnn8=8.6.0.163-1+cuda11.8
sudo apt install -y libcudnn8-dev=8.6.0.163-1+cuda11.8

# Verify cuDNN installation
dpkg -l | grep cudnn

# Install cuDNN libraries (Try the following if the above does not work)
# sudo apt install -y libcudnn8 libcudnn8-dev
```

## Step 6: Verify CUDA Installation
```bash
# Check NVIDIA driver and CUDA version
nvidia-smi
nvcc --version
```

## Step 7: Install TensorFlow
```bash
# Ensure virtual environment is activated
source ~/tensorflow-gpu/bin/activate

# Remove any existing TensorFlow installations
pip uninstall -y tensorflow tensorflow-gpu

# Install TensorFlow 2.10.0 (compatible with CUDA 11.8)
pip install tensorflow==2.10.0
```

## Step 8: Verify TensorFlow GPU Setup
```bash
# Check TensorFlow version and GPU availability
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU Available:', tf.config.list_physical_devices('GPU'))"
```

## Step 9: Test GPU Performance
### Basic GPU Test
```python
# Create a test script (save as basic_gpu_test.py)
import tensorflow as tf

print('Is GPU available:', tf.config.list_physical_devices('GPU'))
print('Is built with CUDA:', tf.test.is_built_with_cuda())

# Simple matrix multiplication test
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    c = tf.matmul(a, b)
    print('Matrix multiplication result:', c)
```

### Performance Comparison Test
```python
# Save as gpu_test.py
import os
# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import time

# Print system info
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("GPU Device Name:", tf.test.gpu_device_name())

# Create large matrices
size = 2000
a = tf.random.normal([size, size])
b = tf.random.normal([size, size])

# Time CPU computation
with tf.device('/CPU:0'):
    start = time.time()
    c = tf.matmul(a, b)
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.2f} seconds")

# Time GPU computation
with tf.device('/GPU:0'):
    start = time.time()
    c = tf.matmul(a, b)
    gpu_time = time.time() - start
    print(f"GPU time: {gpu_time:.2f} seconds")

print(f"GPU speedup: {cpu_time/gpu_time:.2f}x")
```

## Troubleshooting

1. If `nvidia-smi` doesn't work:
   - Ensure WSL2 is using the latest kernel
   - Check if NVIDIA drivers are properly installed in Windows
   - Restart WSL with `wsl --shutdown` from PowerShell

2. If TensorFlow doesn't detect GPU:
   - Verify CUDA paths in `.bashrc`
   - Ensure all versions are compatible (CUDA 11.8, cuDNN 8.6, TF 2.10.0)
   - Check system logs for any error messages

3. If you get CUDA-related errors:
   - Make sure all CUDA paths are correctly set
   - Verify that the NVIDIA driver version is compatible with CUDA 11.8
   - Try rebooting your WSL instance

4. If performance test shows poor GPU speedup:
   - Check if other processes are using the GPU
   - Monitor GPU temperature and clock speeds
   - Try running the test with different matrix sizes
   - Ensure power settings are set to "High Performance" in Windows

## Notes
- Keep the virtual environment activated while working with TensorFlow
- After restarting WSL, you'll need to reactivate the virtual environment
- Monitor GPU memory usage with `nvidia-smi` while running intensive tasks
- The performance test results will vary based on your specific hardware and system load
- For the GTX 960M, expect moderate speedup compared to CPU due to its older architecture


# Alternative considerations

## Step 7: Install TensorFlow and Additional Dependencies
```bash
# Ensure virtual environment is activated
source ~/tensorflow-gpu/bin/activate

# Remove any existing TensorFlow installations
pip uninstall -y tensorflow tensorflow-gpu

# Install TensorFlow 2.10.0 (compatible with CUDA 11.8)
pip install tensorflow==2.10.0

# Install TensorRT dependencies
# The version numbers are specific to TensorFlow 2.10.0
sudo apt update
sudo apt install -y software-properties-common
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install -y libnvinfer8=8.4.3-1+cuda11.6 libnvinfer-plugin8=8.4.3-1+cuda11.6 \
                    libnvinfer-dev=8.4.3-1+cuda11.6 libnvinfer-plugin-dev=8.4.3-1+cuda11.6

# Pin the TensorRT packages to prevent unwanted upgrades
sudo apt-mark hold libnvinfer8 libnvinfer-plugin8 libnvinfer-dev libnvinfer-plugin-dev
```

## Understanding Common Warnings

1. **oneDNN Warning**:
```
This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)...
```
- This is informational and not an error
- You can suppress it by adding this before your Python code:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

2. **cuBLAS Warning**:
```
Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS...
```
- This is a known warning that doesn't affect functionality
- Can be safely ignored

3. **NUMA Warning**:
```
could not open file to read NUMA node: /sys/bus/pci/devices/0000:01:00.0/numa_node
Your kernel may have been built without NUMA support.
```
- This is normal in WSL2 environment
- Doesn't affect GPU functionality
- Can be safely ignored

## Updated GPU Test Script
```python
# Save as gpu_test.py
import os
# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import time

# Print system info
print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("GPU Device Name:", tf.test.gpu_device_name())

# Create large matrices
size = 2000
a = tf.random.normal([size, size])
b = tf.random.normal([size, size])

# Time CPU computation
with tf.device('/CPU:0'):
    start = time.time()
    c = tf.matmul(a, b)
    cpu_time = time.time() - start
    print(f"CPU time: {cpu_time:.2f} seconds")

# Time GPU computation
with tf.device('/GPU:0'):
    start = time.time()
    c = tf.matmul(a, b)
    gpu_time = time.time() - start
    print(f"GPU time: {gpu_time:.2f} seconds")

print(f"GPU speedup: {cpu_time/gpu_time:.2f}x")
```

## Additional Troubleshooting Notes

1. If you still see TensorRT warnings:
   - The warnings about `libnvinfer.so.7` are resolved by installing the correct TensorRT packages
   - These warnings don't affect basic GPU functionality
   - They're only needed if you plan to use TensorRT optimizations

2. For optimal performance:
   - Make sure your Windows power plan is set to "High Performance"
   - Monitor GPU temperature with `nvidia-smi -l 1` in a separate terminal
   - Close other GPU-intensive applications

3. If GPU is not detected:
   ```bash
   # Check NVIDIA driver status in WSL
   nvidia-smi
   
   # Verify CUDA installation
   nvcc --version
   
   # Check CUDA paths
   echo $PATH | grep cuda
   echo $LD_LIBRARY_PATH | grep cuda
   ```

   ```bash
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
   sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
   ```