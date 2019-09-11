using StaticArrays
using CLIMA.VariableTemplates
import CLIMA.DGmethods: BalanceLaw,
                        vars_aux, vars_state, vars_gradient, vars_diffusive,
                        flux_nondiffusive!, flux_diffusive!,
                        source!, gradvariables!, diffusive!,
                        init_aux!, init_state!,
                        boundarycondition_state!, boundarycondition_diffusive!,
                        wavespeed, LocalGeometry

abstract type AdvectionDiffusionProblem end
struct AdvectionDiffusion{dim, P} <: BalanceLaw
  problem::P
  function AdvectionDiffusion{dim}(problem::P) where {dim, P <: AdvectionDiffusionProblem}
    new{dim, P}(problem)
  end
end

# Stored in the aux state are:
#   `coord` coordinate points (needed for BCs)
#   `u` advection velocity
#   `D` Diffusion tensor
vars_aux(::AdvectionDiffusion, T) = @vars(coord::SVector{3, T},
                                           u::SVector{3, T},
                                           D::SMatrix{3, 3, T, 9})
#
# Density is only state
vars_state(::AdvectionDiffusion, T) = @vars(ρ::T)

# Take the gradient of density
vars_gradient(::AdvectionDiffusion, T) = @vars(ρ::T)

# The DG auxiliary variable: D ∇ρ
vars_diffusive(::AdvectionDiffusion, T) = @vars(σ::SVector{3,T})

"""
    flux_nondiffusive!(m::AdvectionDiffusion, flux::Grad, state::Vars,
                       aux::Vars, t::Real)

Computes non-diffusive flux `F` in:

```
∂ρ
-- = - ∇ • (u ρ - σ) = - ∇ • F
∂t
```
Where

 - `u` is the advection velocity
 - `ρ` is the advected quantity
 - `σ` is DG auxiliary variable (`σ = D ∇ ρ` with D being the diffusion tensor)
"""
function flux_nondiffusive!(m::AdvectionDiffusion, flux::Grad, state::Vars,
                            aux::Vars, t::Real)
  ρ = state.ρ
  u = aux.u
  flux.ρ += u * ρ
end

"""
    flux_diffusive!(m::AdvectionDiffusion, flux::Grad, state::Vars,
                     auxDG::Vars, aux::Vars, t::Real)

Computes diffusive flux `F` in:

```
∂ρ
-- = - ∇ • (u ρ - σ) = - ∇ • F
∂t
```
Where

 - `u` is the advection velocity
 - `ρ` is the advected quantity
 - `σ` is DG auxiliary variable (`σ = D ∇ ρ` with D being the diffusion tensor)
"""
function flux_diffusive!(m::AdvectionDiffusion, flux::Grad, state::Vars,
                         auxDG::Vars, aux::Vars, t::Real)
  σ = auxDG.σ
  flux.ρ += -σ
end

"""
    gradvariables!(m::AdvectionDiffusion, transform::Vars, state::Vars,
                   aux::Vars, t::Real)

Set the variable to take the gradient of (`ρ` in this case)
"""
function gradvariables!(m::AdvectionDiffusion, transform::Vars, state::Vars,
                        aux::Vars, t::Real)
  transform.ρ = state.ρ
end

"""
    diffusive!(m::AdvectionDiffusion, transform::Vars, state::Vars, aux::Vars,
               t::Real)

Set the variable to take the gradient of (`ρ` in this case)
"""
function diffusive!(m::AdvectionDiffusion, auxDG::Vars, gradvars::Grad,
                    state::Vars, aux::Vars, t::Real)
  ∇ρ = gradvars.ρ
  D = aux.D
  auxDG.σ = D * ∇ρ
end

"""
    source!(m::AdvectionDiffusion, _...)

There is no source in the advection-diffusion model
"""
source!(m::AdvectionDiffusion, _...) = nothing

"""
    wavespeed(m::AdvectionDiffusion, nM, state::Vars, aux::Vars, t::Real)

Wavespeed with respect to vector `nM`
"""
function wavespeed(m::AdvectionDiffusion, nM, state::Vars, aux::Vars, t::Real)
  u = aux.u
  abs(dot(nM, u))
end

"""
    init_aux!(m::AdvectionDiffusion, aux::Vars, geom::LocalGeometry)

initialize the auxiliary state
"""
function init_aux!(m::AdvectionDiffusion, aux::Vars, geom::LocalGeometry)
  aux.coord = geom.coord
  init_velocity_diffusion!(m.problem, aux, geom)
end

function init_state!(m::AdvectionDiffusion, state::Vars, aux::Vars,
                     coords, t::Real)
  initial_condition!(m.problem, state, aux, coords, t)
end

function boundarycondition_state!(m::AdvectionDiffusion,
                                  stateP::Vars, auxP::Vars, nM,
                                  stateM::Vars, auxM::Vars,
                                  bctype, t, _...)
  # Dirichlet
  if bctype == 1
    init_state!(m, stateP, auxP, auxP.coord, t)
  end
end

function boundarycondition_diffusive!(m::AdvectionDiffusion,
                                      stateP::Vars, diffP::Vars, auxP::Vars, nM,
                                      stateM::Vars, diffM::Vars, auxM::Vars,
                                      bctype, t, _...)
  if bctype == 2
    # TODO add Neumann
  end
end
