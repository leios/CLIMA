#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using JuliaFormatter

headbranch = get(ARGS, 1, "sb/julia-formatter")

for filename in
    readlines(`git diff --name-only --diff-filter=AM $headbranch...`)
    endswith(filename, ".jl") || continue

    format(
        filename,
        verbose = true,
        indent = 4,
        margin = 80,
        always_for_in = true,
        whitespace_typedefs = true,
        whitespace_ops_in_indices = true,
        remove_extra_newlines = false,
    )
end
