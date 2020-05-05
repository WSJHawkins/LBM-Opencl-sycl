# LBM-Opencl-sycl
Contains code to implement a d2q9-bgk lattice boltzmann scheme. Two versions are present to compare. One in OpenCl and one in Sycl.
The traces from running these codes on an Intel NUC (Iris Pro 580) with the OpenCl Intercept Layer are also included.

## How to run
There are Makefiles to compile with LLVM SYCL or ComputeCPP.
To compile with hipSYCL use the hipSYCL verison of this code. It is the same bar the recip functions have been removed and manually typed and select operations have been replaced with ternary operations. Use the following command:
``` syclcc-clang -std=c++17 -O3 --hipsycl-gpu-arch=gfx906 --hipsycl-platform=rocm  d2q9-bgk_hipSycl.cpp -o d2q9-bgk ```
changing the platform and architecture where appropriate.


After compiling using the given makefile run the code using the following command:  
``` ./d2q9-bgk Inputs/input_128x128.params Obstacles/obstacles_128x128.dat ```  
The 128x128 can be changed to 256x256 or 1024x1024 to see how it performs at different scales.

This repository has been superceeded by: https://github.com/WSJHawkins/ExploringSycl
Any updates to the code will be placed there.
