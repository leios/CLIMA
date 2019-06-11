module StrongStabilityPreservingRungeKuttaMethod
export StrongStabilityPreservingRungeKutta, updatedt!
export StrongStabilityPreservingRungeKutta33
export StrongStabilityPreservingRungeKutta34

using GPUifyLoops
include("StrongStabilityPreservingRungeKuttaMethod_kernels.jl")

using ..ODESolvers
ODEs = ODESolvers
using ..SpaceMethods

"""
    StrongStabilityPreservingRungeKutta(f, Q; dt, t0 = 0)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

        Q̇ = f(Q)

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

This uses either 3-stage or 4-stage 3rd order Strong-Stability-Preserving (SS) time-integrators.

### References
S. Gottlieb and C.-W. Shu, Total variation diminishing Runge-Kutta schemes, Math .Comp .,67 (1998), pp .73–85
R.J. Spiteri and S.J. Ruuth, A new class of optimal high-order strong-stability-preserving time discretization methods, SIAM J. Numer. Anal., 40 (2002), pp .469–491
"""

struct StrongStabilityPreservingRungeKutta{T, AT, Nstages} <: ODEs.AbstractODESolver
"time step"
dt::Array{T,1}
"time"
t::Array{T,1}
"rhs function"
rhs!::Function
"Storage for RHS during the StrongStabilityPreservingRungeKutta update"
Rstage::AT
"Storage for the stage state during the StrongStabilityPreservingRungeKutta update"
Qstage::AT
"RK coefficient vector A (rhs scaling)"
RKA::Array{T,2}
"RK coefficient vector B (rhs add in scaling)"
RKB::Array{T,1}
"RK coefficient vector C (time scaling)"
RKC::Array{T,1}

function StrongStabilityPreservingRungeKutta(rhs!::Function, RKA, RKB, RKC, Q::AT; dt=nothing, t0=0) where {AT<:AbstractArray}

    @assert dt != nothing

    T = eltype(Q)
    dt = [T(dt)]
    t0 = [T(t0)]
    new{T, AT, length(RKB)}(dt, t0, rhs!, similar(Q), similar(Q), RKA, RKB, RKC)
end
end

function StrongStabilityPreservingRungeKutta(spacedisc::AbstractSpaceMethod, RKA, RKB, RKC,Q; dt=nothing, t0=0)
    rhs! = (x...) -> SpaceMethods.odefun!(spacedisc, x...)
    StrongStabilityPreservingRungeKutta(rhs!, RKA, RKB, RKC, Q; dt=dt, t0=t0)
end

function StrongStabilityPreservingRungeKutta33(F::Union{Function, AbstractSpaceMethod}, Q; dt=nothing, t0=0)
    T=eltype(Q)
    RKA = [ T(1) T(0); T(3//4) T(1//4); T(1//3) T(2//3) ]
    RKB = [T(1), T(1//4), T(2//3)]
    RKC = [ T(0), T(1), T(1//2) ]
    StrongStabilityPreservingRungeKutta(F, RKA, RKB, RKC, Q; dt=dt, t0=t0)
end

function StrongStabilityPreservingRungeKutta34(F::Union{Function, AbstractSpaceMethod}, Q; dt=nothing,t0=0)
    T=eltype(Q)
    RKA = [ T(1) T(0); T(0) T(1); T(2//3) T(1//3); T(0) T(1) ]
    RKB = [ T(1//2); T(1//2); T(1//6); T(1//2) ]
    RKC = [ T(0); T(1//2); T(1); T(1//2) ]
    StrongStabilityPreservingRungeKutta(F, RKA, RKB, RKC, Q; dt=dt, t0=t0)
end


"""
    updatedt!(ssp::StrongStabilityPreservingRungeKutta, dt)

Change the time step size to `dt` for `ssp.
"""
updatedt!(ssp::StrongStabilityPreservingRungeKutta, dt) = ssp.dt[1] = dt

function ODEs.dostep!(Q, ssp::StrongStabilityPreservingRungeKutta, timeend, adjustfinalstep)
    time, dt = ssp.t[1], ssp.dt[1]
    if adjustfinalstep && time + dt > timeend
        dt = timeend - time
        @assert dt > 0
    end
    RKA, RKB, RKC = ssp.RKA, ssp.RKB, ssp.RKC
    rhs! = ssp.rhs!
    Rstage, Qstage = ssp.Rstage, ssp.Qstage
    
    rv_Q = ODEs.realview(Q)
    rv_Rstage = ODEs.realview(Rstage)
    rv_Qstage = ODEs.realview(Qstage)

    threads = 1024
    blocks = div(length(rv_Q) + threads - 1, threads)

    rv_Qstage .= rv_Q
    for s = 1:length(RKB)
        rv_Rstage .= 0
        rhs!(Rstage, Qstage, time + RKC[s] * dt)
      
        @launch(ODEs.device(Q), threads = threads, blocks = blocks,
                update!(rv_Rstage, rv_Q, rv_Qstage, RKA[s,1], RKA[s,2], RKB[s], dt))
    end
    rv_Q .= rv_Qstage

    if dt == ssp.dt[1]
        ssp.t[1] += dt
    else
        ssp.t[1] = timeend
    end

end

end