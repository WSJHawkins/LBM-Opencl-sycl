# LBM-Opencl-sycl
Contains code to implement a d2q9-bgk lattice boltzmann scheme. Two versions are present to compare. One in OpenCl and one in Sycl

## How to run
After compiling using the given makefile run the code using the following command:  
``` ./d2q9-bgk Inputs/input_128x128.params Obstacles/obstacles_128x128.dat ```  
The 128x128 can be changed to 256x256 or 1024x1024 to see how it performs at different scales.
