module ColumnwisePreconditionedConjugateGradientSolver

export ColumnwisePreconditionedConjugateGradient

using ..LinearSolvers
const LS = LinearSolvers
using ..MPIStateArrays: device, realview

using LinearAlgebra
using LazyArrays
using StaticArrays
using GPUifyLoops

struct ColumnwisePreconditionedConjugateGradient{T, D} <: LS.AbstractIterativeLinearSolver

  rtol::T
  atol::T
  max_iter::T

  r0::MArray{T}
  z0::MArray{T}
  p0::MArray{T}

  r1::MArray{T}
  z1::MArray{T}
  p1::MArray{T}
  Lp::MArray{T}

  dims::D


  function ColumnwisePreconditionedConjugateGradient(Q::AT; max_iter=size(Q)[3]*size(Q)[5],
          rtol=eps(eltype(AT)), atol=eps(eltype(AT)), dims=:) where AT
      # FIXME: Need to revisit this (assuming uniform vertical resolution in each column)
      return new{eltype(Q), eltype(dims)}(rtol, atol, max_iter, dims)
  end

end

function LS.initialize!(linearoperator!, Q, Qrhs,
                        solver::ColumnwisePreconditionedConjugateGradient,
                        args...)
    # Initialize as 'not converged'
    return false
end

function LS.doiteration!(linearoperator!, Q, Qrhs,
                         solver::ColumnwisePreconditionedConjugateGradient{M},
                         args...) where M

    rtol = solver.rtol
    atol = solver.atol
    residual_norm = typemax(eltype(Q))
    dims = solver.dims
    converged = false

    max_iter = solver.max_iter

    r0 = solver.r0
    z0 = solver.z0
    p0 = solver.p0

    r1 = solver.r1
    z1 = solver.z1
    p1 = solver.p1

    # Smack residual by linear operator
    linearoperator!(r0, Q)
    r0 .= Qrhs - r0
    applyPC!(z0, r0)
    p0 = copy(z0)

    Lp = solver.Lp

    absolute_residual = norm(r0, 2, false, dims=dims)
    relative_residual = absolute_residual / norm(Qrhs, 2, false, dims=dims)

    # TODO: Need to review termination criterion
    converged = false
    if (absolute_residual <= atol) || (relative_residual <= rtol)
        converged = true
        return (converged, absolute_residual)
    end

    for j in 1:max_iter

        linearoperator!(Lp, p0)
        α = sum(r0 .* z0, dims=dims) ./ sum(p0 .* Lp, dims=dims)

        # Update along preconditioned direction
        @. Q += α * p0
        @. r1 = r0 - α * p0

        # TODO: Probably need to perform MPI call for allreduce
        if maximum(norm(r1, 2, false, dims=dims)) < atol
            converged = true
            break
        end

        applyPC!(z1, r1)

        β = sum(z1 .* r1, dims=dims) ./ sum(z0 .* r0, dims=dims)

        # Update
        @. p0 = z1 + β * p0
        @. z0 = z1
        @. r0 = r1

    end

end

end
