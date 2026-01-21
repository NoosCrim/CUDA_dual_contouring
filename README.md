# CUDA dual contouring

Library with parallelized implementation of dual contouring, algorithm that's a hybrid of cube-based and dual-based contouring methods.

For now, interface consists of function template `RunDualContouring` that takes in functors for density and gradient values, size of grid and scale of output mesh.

## Example

Provided example is built to `out/cuda_dc_demo` using `make example`. Usage instructions with flag `-h`
