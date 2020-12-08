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
      NumericalFluxSecondOrder


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
  Set a default set of properties and their default values
"""
function prop_defaults()
  bl_prop=NamedTuple()
  bl_prop=( bl_prop...,init_aux_geom=nothing)
  bl_prop=( bl_prop...,   init_theta=nothing)
  bl_prop=( bl_prop..., source_theta=nothing)
  bl_prop=( bl_prop..., calc_kappa_diff=nothing)
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
  κ¹,κ²,κ³=e.bl_prop.calc_kappa_diff(G.∇θ,A.npt,A.elnum,A.xc,A.yc,A.zc)
  # κ¹=κ²=κ³=-0.1
  GF.κ∇θ = Diagonal([κ¹,κ²,κ³])*G.∇θ
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
  Not sure if I have to have this to trigger boundary_state! call - possibly, need to check
"""
function boundary_conditions( e::eq_type, _...)
 ( nothing, )
end

"""
  Zero normal gradient boundary condition
"""
function boundary_state!(nF::Union{NumericalFluxSecondOrder}, bc, e::eq_type, Q⁺::Vars, GF⁺::Vars, A⁺::Vars,n,Q⁻::Vars,GF⁻::Vars,A⁻::Vars,t,_...)
 Q⁺.θ=Q⁻.θ
 GF⁺.κ∇θ= n⁻ * -0
 nothing
end

function wavespeed(e::eq_type, _...)
 0
end

# *** note to me - remember to add numerical_flux_second_order(CentralNumericalFluxSecondOrder) at 
# interfaces, this is a term that averages flux⁺ and flux- (it may already be there?). 

end
