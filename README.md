# Izhikevich Neuron Simulation with ISPC

This project implements the Izhikevich neuron model using ISPC (Intel SPMD Program Compiler), ported from a CUDA implementation.

## Prerequisites

- ISPC compiler
- CMake
- A C++ compiler with C++11 support

## Building the Project (For Windows):

1. Install ISPC if not already installed.

2. Open **PowerShell** and navigate to the project folder:  
   ```powershell
   cd ...path...\izhikevich_simulation

3. Create a build directory and build the project:
   ```bash
   mkdir build
   cd build
   cmake ..
   cmake --build .
   ```

## Running the Simulation

From the build directory, run:
```bash
.\Debug\izhikevich_simulation.exe
```

This will create two output files:
- `spikes.csv`: Contains the spike times and neuron indices
- `voltages.csv`: Contains the membrane potentials of each neuron over time

## Visualizing the Results

Use the given Python script to visualize the results:
```bash
python ../plot_izhikevich.py
```

This will display a plot showing:
- The membrane potentials of the four neuron types (Regular, Fast, Chattering, and Bursting)
- The spike times for each neuron

## Implementation Details

The implementation simulates four different types of Izhikevich neurons:
1. Regular spiking
2. Fast spiking
3. Chattering
4. Bursting

Each neuron type has different parameters (a, b, c, d) that determine its firing characteristics.

## Differences from CUDA Version

The ISPC implementation differs from the CUDA version in the following ways:
- Instead of GPU parallelism, ISPC uses SIMD parallelism on the CPU
- The shared memory and atomic operations in CUDA are replaced with a simpler approach in ISPC
- Memory allocation is done using standard C++ rather than CUDA's memory management functions 