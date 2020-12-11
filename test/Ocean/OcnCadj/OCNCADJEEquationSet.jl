"""
    module OCNCADJEEquationSet
   
 Defines an equation set for an explicit convective adjustment scheme for ocean.

 Defines kernels to evaluates RHS of

  d ϕ / dt = - d/dz κ( ϕ_init ) d/dz ϕ

 subject to specified prescribed gradient or flux bc's

 - [`OCNCADJEEquations`](@ref)
     balance law struct created by this module 

"""
module OCNCADJEEquationSet

export OCNCADJEEquations
export PenaltyNumFluxDiffusive

using ClimateMachine.BalanceLaws:
      Auxiliary,
      BalanceLaw,
      Gradient,
      GradientFlux,
      Prognostic

import ClimateMachine.BalanceLaws:
       boundary_conditions,
       boundary_state!,
       compute_gradient_argument!,
       compute_gradient_flux!,
       flux_first_order!,
       flux_second_order!,
       init_state_prognostic!,
       nodal_init_state_auxiliary!,
       source!,
       vars_state,
       wavespeed

using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.MPIStateArrays

using ClimateMachine.DGMethods.NumericalFluxes:
      CentralNumericalFluxGradient,
      CentralNumericalFluxSecondOrder,
      NumericalFluxFirstOrder,
      NumericalFluxSecondOrder,
      RusanovNumericalFlux

import ClimateMachine.DGMethods.NumericalFluxes:
       numerical_boundary_flux_second_order!,
       numerical_flux_second_order!


using  ClimateMachine.VariableTemplates

using LinearAlgebra
using StaticArrays

"""
    OCNCADJEEquations
Type that holds specification for an explicit convection balance law instance.
"""
struct OCNCADJEEquations{FT, BLP} <: BalanceLaw
        bl_prop :: BLP
        function OCNCADJEEquations{FT}(;
          bl_prop::BLP=nothing,
        ) where {FT, BLP}
          return new{FT,BLP}(
                 bl_prop,
                 )
        end
end

eq_type=OCNCADJEEquations

"""
  Extend the NumericalFluxSecondOrder to include a penalty term numerical
  flux formulation.
"""
struct PenaltyNumFluxDiffusive <: NumericalFluxSecondOrder end

"""
  Set a default set of properties and their default values
  - init_aux_geom   :: function to initialize geometric terms stored in aux.
  - init_theta      :: function to set initial θ values.
  - source_theta    :: function to add a source term to θ.
  - calc_kappa_diff :: function to set diffusion coeffiecient(s).
  - get_wavespeed   :: function to return a wavespeed for Rusanov computations (there aren't any in this model)
  - get_penalty_tau :: function to set timescale on which to bring state+ and state- together
"""
function prop_defaults()
  bl_prop=NamedTuple()
  bl_prop=( bl_prop...,init_aux_geom=nothing)
  bl_prop=( bl_prop...,   init_theta=nothing)
  bl_prop=( bl_prop..., source_theta=nothing)
  bl_prop=( bl_prop..., calc_kappa_diff=nothing)
  bl_prop=( bl_prop..., get_wavespeed=(0.) )
  bl_prop=( bl_prop..., get_penalty_tau=(1.) )
end

"""
  Declare single prognostic state variable, θ
"""
function vars_state(e::eq_type, ::Prognostic, FT)
  @vars begin
    θ::FT
  end
end

"""
  Auxiliary state variable symbols for array index and real world coordinates and for
  θ value at reference time used to compute κ.
"""
function vars_state(e::eq_type, ::Auxiliary, FT)
  @vars begin
       npt::Int
     elnum::Int
        xc::FT
        yc::FT
        zc::FT
     θⁱⁿⁱᵗ::FT
  end
end

"""
  Gradient computation stage input (and output) variable symbols
"""
function vars_state(e::eq_type, ::Gradient, FT)
  @vars begin
        ∇θ::FT
    ∇θⁱⁿⁱᵗ::FT
  end
end

"""
  Flux due to gradient accumulation stage variable symbols
"""
function vars_state(e::eq_type, ::GradientFlux, FT)
  @vars begin
   κ∇θ::SVector{3,FT}
  end
end

"""
  Initialize prognostic state variables
"""
function init_state_prognostic!(e::eq_type, Q::Vars, A::Vars, geom::LocalGeometry, FT)
  npt=getproperty(geom,:n)
  elnum=getproperty(geom,:e)
  x=geom.coord[1]
  y=geom.coord[2]
  z=geom.coord[3]
  Q.θ=e.bl_prop.init_theta(x,y,z,npt,elnum)
  nothing
end

"""
  Initialize auxiliary state variables
"""
function nodal_init_state_auxiliary!(e::eq_type, A::Vars, tmp::Vars, geom::LocalGeometry, _...)
  npt=getproperty(geom,:n)
  elnum=getproperty(geom,:e)
  x=geom.coord[1]
  y=geom.coord[2]
  z=geom.coord[3]
  A.npt, A.elnum, A.xc, A.yc, A.zc = e.bl_prop.init_aux_geom(npt,elnum,x,y,z)
  A.θⁱⁿⁱᵗ=0
  nothing
end

"""
  Set any source terms for prognostic state external sources
"""
# function source!(e::eq_type,S::Vars,Q::Vars,G::Vars,A::Vars,t)
function source!(e::eq_type,S::Vars,Q::Vars,G::Vars,A::Vars,_...)
  S.θ=e.bl_prop.source_theta(S.θ,A.npt,A.elnum,A.xc,A.yc,A.zc)
  nothing
end

"""
  No flux first order for diffusion equation. but we must define a stub
"""
function flux_first_order!( e::eq_type, _...)
  nothing
end

"""
  Set values to have gradients computed
"""
function compute_gradient_argument!( e::eq_type, G::Vars, Q::Vars, A::Vars, t )
  G.∇θ = Q.θ
  G.∇θⁱⁿⁱᵗ=A.θⁱⁿⁱᵗ
  nothing
end

"""
  Compute diffusivity tensor times computed gradient to give net gradient flux.
"""
function compute_gradient_flux!( e::eq_type, GF::Vars, G::Grad, Q::Vars, A::Vars, t )
  # "Non-linear" form (for time stepped)
  κ¹,κ²,κ³=e.bl_prop.calc_kappa_diff(G.∇θ,A.npt,A.elnum,A.xc,A.yc,A.zc)
  # "Linear" form (for implicit)
  κ¹,κ²,κ³=e.bl_prop.calc_kappa_diff(G.∇θⁱⁿⁱᵗ,A.npt,A.elnum,A.xc,A.yc,A.zc)
  # Maybe I should pass both G.∇θ and G.∇θⁱⁿⁱᵗ?
  GF.κ∇θ = Diagonal(@SVector([κ¹,κ²,κ³]))*G.∇θ
  nothing
end

"""
  Pass flux components for second order term into update kernel.
"""
function flux_second_order!( e::eq_type, F::Grad, Q::Vars, GF::Vars, H::Vars, A::Vars, t )
  F.θ += GF.κ∇θ 
  nothing
end

"""
  Define boundary condition flags/types to iterate over
"""
function boundary_conditions( e::eq_type, _...)
 ( 1, )
end

"""
  Zero normal gradient boundary condition
"""
function boundary_state!(nF::Union{CentralNumericalFluxGradient}, bc, e::eq_type, Q⁺::Vars, A⁺::Vars,n,Q⁻::Vars,A⁻::Vars,t,_...)
 Q⁺.θ=Q⁻.θ
 nothing
end

"""
  Set any first order numerical flux to null
  NumericalFluxFirstOrder is an abstract type that currently generalizes
  RusanovNumericalFlux, CentralNumericalFluxFirstOrder, RoeNumericalFlux, HLLCNumericalFlux.
"""
# No first order fluxes so numerical flux needed. NumericalFluxFirstOrder
function boundary_state!(nF::NumericalFluxFirstOrder, bc, e::eq_type, Q⁺::Vars, A⁺::Vars,n,Q⁻::Vars,A⁻::Vars,t,_...)
 nothing
end

function boundary_state!(nF::Union{NumericalFluxSecondOrder}, bc, e::eq_type, Q⁺::Vars, GF⁺::Vars, A⁺::Vars,n⁻,Q⁻::Vars,GF⁻::Vars,A⁻::Vars,t,_...)
 Q⁺.θ=Q⁻.θ
 GF⁺.κ∇θ= n⁻ * -0
 nothing
end

function wavespeed(e::eq_type, _...)
 # Used in Rusanov term. 
 # Only active if there is a flux first order term?
 e.bl_prop.get_wavespeed()
end

"""
  Penalty flux formulation of second order numerical flux. This formulation
  computes the CentralNumericalFluxSecondOrder term first (which is just the average
  of the + and - fluxes and an edge), and then adds a "penalty" flux that relaxes 
  the edge state + and - toward each other.
"""
function numerical_flux_second_order!(
    ::PenaltyNumFluxDiffusive,
    bl::eq_type,
    fluxᵀn::Vars{S},
    n::SVector,
    state⁻::Vars{S},
    diff⁻::Vars{D},
    hyperdiff⁻::Vars{HD},
    aux⁻::Vars{A},
    state⁺::Vars{S},
    diff⁺::Vars{D},
    hyperdiff⁺::Vars{HD},
    aux⁺::Vars{A},
    t,
) where {S, HD, D, A}

    numerical_flux_second_order!(
        CentralNumericalFluxSecondOrder(),
        bl,
        fluxᵀn,
        n,
        state⁻,
        diff⁻,
        hyperdiff⁻,
        aux⁻,
        state⁺,
        diff⁺,
        hyperdiff⁺,
        aux⁺,
        t,
    )

    Fᵀn = parent(fluxᵀn)
    FT = eltype(Fᵀn)
    tau = e.bl_prop.get_penalty_tau()
    Fᵀn .-= tau * (parent(state⁻) - parent(state⁺))
end

# We are assuming zero gradient bc for now - so there is no numerical second order 
# flux from boundary
numerical_boundary_flux_second_order!(nf::PenaltyNumFluxDiffusive, _...) = nothing

end
