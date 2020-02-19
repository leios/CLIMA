module MultistateMultirateRungeKuttaMethod
using ..ODESolvers
using ..AdditiveRungeKuttaMethod
using ..LowStorageRungeKuttaMethod
using ..StrongStabilityPreservingRungeKuttaMethod
using ..MPIStateArrays: device, realview

using GPUifyLoops
include("MultistateMultirateRungeKuttaMethod_kernels.jl")

export MultistateMultirateRungeKutta

ODEs = ODESolvers
LSRK2N = LowStorageRungeKutta2N
SSPRK = StrongStabilityPreservingRungeKutta

"""
    MultistateMultirateRungeKutta(slow_solver, fast_solver; dt, t0 = 0)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q_fast} = f_fast(Q_fast, Q_slow, t) 
  \\dot{Q_slow} = f_slow(Q_slow, Q_fast, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

The constructor builds a multistate multirate Runge-Kutta scheme using two different RK
solvers and two different MPIStateArrays. This is based on

Currently only the low storage RK methods can be used as slow solvers

  - [`LowStorageRungeKuttaMethod`](@ref)

### References
"""
mutable struct MultistateMultirateRungeKutta{SS, FS, RT} <: ODEs.AbstractODESolver
  "slow solver"
  slow_solver::SS
  "fast solver"
  fast_solver::FS
  "time step"
  dt::RT
  "time"
  t::RT

  function MultistateMultirateRungeKutta(slow_solver::LSRK2N,
                                         fast_solver,
                                         Q=nothing;
                                         dt=ODEs.getdt(slow_solver), t0=slow_solver.t
                                         ) where {AT<:AbstractArray}
    SS = typeof(slow_solver)
    FS = typeof(fast_solver)
    RT = real(eltype(slow_solver.dQ))
    return new{SS, FS, RT}(slow_solver, fast_solver, RT(dt), RT(t0))
  end
end
MSMRRK = MultistateMultirateRungeKutta

function ODEs.dostep!(Qvec, msmrrk::MSMRRK, param,
                      timeend::AbstractFloat, adjustfinalstep::Bool)
  time, dt = msmrrk.t, msmrrk.dt
  @assert dt > 0
  if adjustfinalstep && time + dt > timeend
    dt = timeend - time
    @assert dt > 0
  end

  ODEs.dostep!(Qvec, msmrrk, param, time, dt)

  if dt == mrrk.dt
    msmrrk.t += dt
  else
    msmrrk.t = timeend
  end
  return msmrrk.t
end

function ODEs.dostep!(Qvec, msmrrk::MSMRRK{SS}, param,
                      time::AbstractFloat, dt::AbstractFloat,
                      in_slow_δ = nothing, in_slow_rv_dQ = nothing,
                      in_slow_scaling = nothing) where {SS <: LSRK2N}
  slow = msmrrk.slow_solver
  fast = msmrrk.fast_solver

  Qslow = Qvec.slow
  Qfast = Qvec.fast  
    
  slow_rv_dQ = realview(slow.dQ)

  threads = 256
  blocks = div(length(realview(Qslow)) + threads - 1, threads)

  for slow_s = 1:length(slow.RKA)
    # Currnent slow state time
    slow_stage_time = time + slow.RKC[slow_s] * dt

    # Evaluate the slow mode
    slow.rhs!(slow.dQ, Qslow, param, slow_stage_time, increment = true)

    if in_slow_δ !== nothing
      slow_scaling = nothing
      if slow_s == length(slow.RKA)
        slow_scaling = in_slow_scaling
      end
      # update solution and scale RHS
      @launch(device(Qslow), threads=threads, blocks=blocks,
              update!(slow_rv_dQ, in_slow_rv_dQ, in_slow_δ, slow_scaling))
    end

    # Fractional time for slow stage
    if slow_s == length(slow.RKA)
      γ = 1 - slow.RKC[slow_s]
    else
      γ = slow.RKC[slow_s + 1] - slow.RKC[slow_s]
    end

    # RKB for the slow with fractional time factor remove (since full
    # integration of fast will result in scaling by γ)
    slow_δ = slow.RKB[slow_s] / (γ)

    # RKB for the slow with fractional time factor remove (since full
    # integration of fast will result in scaling by γ)
    nsubsteps = ODEs.getdt(fast) > 0 ? ceil(Int, γ * dt / ODEs.getdt(fast)) : 1
    fast_dt = γ * dt / nsubsteps

    # reconcile slow equation using fast equation
    @launch(device(Qfast), threads=threads, blocks=blocks,
            do_integrals_and_pass_from_slow_to_fast!(Qfast, slow_rv_dQ, fast.rhs!.bl, slow.rhs!.bl))
      
    for substep = 1:nsubsteps
      slow_rka = nothing
      if substep == nsubsteps
        slow_rka = slow.RKA[slow_s%length(slow.RKA) + 1]
      end
      fast_time = slow_stage_time + (substep - 1) * fast_dt
      ODEs.dostep!(Qfast, fast, param, fast_time, fast_dt, slow_δ, slow_rv_dQ,
                   slow_rka)
    end

    # reconcile slow equation using fast equation
    @launch(device(Qslow), threads=threads, blocks=blocks,
            reconcile_fast_to_slow!(slow_rv_dQ, Qfast, slow.rhs!.bl, fast.rhs!.bl))
  end
  return nothing
end

end
