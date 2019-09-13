module DifferentialEquationsMethods

using DifferentialEquations
using ..ODESolvers
ODEs = ODESolvers
using ..SpaceMethods

export DifferentialEquationsMethod

struct DifferentialEquationsMethod{T, RT, S} <: ODEs.AbstractODESolver
  "time step"
  dt::Array{RT,1}
  "time"
  t::Array{RT,1}
  "rhs function"
  rhs!
  "DifferentialEquations.jl solver"
  deqs_solver::S
end

function DifferentialEquationsMethod(deqs_solver, rhs!, Q::AT; dt=nothing, t0=0) where {AT<:AbstractArray}
  T = eltype(Q)
  RT = real(T)
  dt = [dt]
  t0 = [t0]
  
  S = typeof(deqs_solver)
  DifferentialEquationsMethod{T, RT, S}(dt, t0, rhs!, deqs_solver)
end

function DifferentialEquationsMethod(deqs_solver, spacedisc::AbstractSpaceMethod,
                                Q::AT; dt=nothing, t0=0) where {AT<:AbstractArray}
  rhs! = (x...; increment) -> SpaceMethods.odefun!(spacedisc, x..., increment = increment)
  DifferentialEquationsMethod(deqs_solver, rhs!, Q; dt=dt, t0=t0)
end

function ODEs.solve!(Q, solver::DifferentialEquationsMethod, param=nothing; timeend::Real=Inf,
                     adjustfinalstep=true, numberofsteps::Integer=0, callbacks=())

  @assert isfinite(timeend) || numberofsteps > 0

  # TODO: handle callbacks
  @assert isempty(callbacks) 

  T = eltype(Q)
  deqs_solver = solver.deqs_solver
  rhs! = solver.rhs!
  dt = solver.dt[1]
  t = solver.t[1]

  tspan = (t, T(timeend))
  problem = ODEProblem(Q, tspan) do du, u, p, t
    rhs!(du, u, param, t; increment=false)
  end

  # TODO: use integrator interface ?
  solve(problem, deqs_solver; alias_u0=true, save_on=false,
        adaptive=false, dt=dt, unstable_check=(_...)->false)
end

end
