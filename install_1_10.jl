#!/usr/bin/env julia

ENV["JULIA_PKG_PRECOMPILE_AUTO"] = ("--precompile" in ARGS) ? "1" : "0"

using Pkg

Pkg.activate(@__DIR__)

local_packages = Pkg.PackageSpec[
    Pkg.PackageSpec(path=joinpath(@__DIR__, "..", "MuKumari")),
    Pkg.PackageSpec(path=joinpath(@__DIR__, "..", "SCRIBE")),
    Pkg.PackageSpec(path=joinpath(@__DIR__, "..", "VulcanJ")),
]

git_packages = Pkg.PackageSpec[
    Pkg.PackageSpec(url="https://github.com/probcomp/DynamicForwardDiff.jl", rev="main"),
    Pkg.PackageSpec(url="https://github.com/probcomp/GenTraceKernelDSL.jl", rev="main"),
    Pkg.PackageSpec(url="https://github.com/probcomp/GenSMCP3.jl", rev="main"),
]

Pkg.add(vcat(local_packages, git_packages))
Pkg.develop(local_packages)
Pkg.resolve()
Pkg.instantiate()
