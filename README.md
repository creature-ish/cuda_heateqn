# cuda_heateqn: A solution to the 2-dimensional heat equation using CUDA Thrust.

An exercise for SciNet's HPC133 Intro to GPU Programming course. Simulates heat diffusion in 2D over time. My first time coding in Thrust, so bear with me! Note that this is a Visual Studio 2022 project (GPU resources on Cedar and Graham are in high demand right now and I don't have Linux running natively quite yet). 

Adjust ``params.cpp`` to change the simulation parameters, and change the equation in ``dens_init``'s ``operator()`` to change the initial state.

## Todos

* Plot natively in C++ rather than straight .csv export OR find a faster data export library (export is the major performance bottleneck as it stands)
* Find a way to better initialize the permutation iterators (doing the same thing four times sucks!)