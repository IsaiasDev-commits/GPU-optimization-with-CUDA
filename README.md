# GPU Optimization - CUDA Project

This project implements matrix multiplication using CUDA to leverage GPU capabilities and optimize performance for computation-heavy operations.

## Description

The project uses the NVIDIA CUDA API to perform `N x N` matrix multiplication on the GPU. The goal is to enhance the performance of linear algebra operations by utilizing GPU parallelization.

## Requirements

- **CUDA Toolkit**: Ensure that the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) is installed on your system.
- **NVIDIA GPU** with CUDA support.
- **Visual Studio Code** with the necessary extensions for CUDA development (e.g., C/C++ extension).
- **Git** for version control and source code management.

## Installation

### 1. Clone the repository

To clone the repository to your local machine, run the following command:

```bash
git clone https://github.com/IsaiasDev-commits/GPU-optimization-with-CUDA.git

2. Install Visual Studio Code Extensions
Make sure you have the following extensions installed in Visual Studio Code:

C/C++: Provides IntelliSense, debugging, and code browsing for C/C++.

CUDA (optional): If you want syntax highlighting and other CUDA-specific features.

3. Compile the code
To compile the CUDA code on your machine, open the terminal in the project folder and run the following command:
nvcc matmul_cuda.cu -o matmul_cuda.exe

4. Run the program
To run the program and perform matrix multiplication, use the following command:
.\matmul_cuda.exe
You should see an output similar to:
C[0][0] = 2048.000000
