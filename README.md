# Arrodes

Bringing out of MuKumari.jl the objective inference code, making it a proper package instead of a massive one-file script. Implements the various elements of a particle filter to complete Open-Ended SIPS, the random fourier features necessary to make arbitrary objective functions, and the network-based approaches to both learning Q-functions for MDPs and inferring Q-functions from ExperienceBuffers of state-action data.

For installation, if need be - particularly if `Crux` keeps failing to install due to issues finding OpenSSL:
```julia
using Pkg; Pkg.add(name="OpenSSL_jll", version="3.0")
```
