# CuRidge

Ridge Regression written in CUDA

## Usage

First make sure you have installed CUDA.

To use the regression just copy the files in `src/` into your project and use it like shown in the examples.

The folder `examples/` provides two usage examples:

1. The first examples just includes the project-file into an cuda file/project.

   Do the following to run it:

   Compile:

   ```sh
   nvcc -std=c++11  -o out first_example.cu -lcublas -lcusolver
   ```

   And run:

   ```sh
   ./out
   ```

2. The second example uses forward declaration in an c++ file. This means that we can call the cuda code in c++ projects.

   Do the following to run it:

   Compile:

   ```sh
   g++ -c second_example.cpp
   nvcc -c path_to_ridge/ridge.cu
   nvcc -std=c++11  -o out second_example.o ridge.o -lcublas -lcusolver
   ```

   And run:

   ```sh
   ./out
   ```
