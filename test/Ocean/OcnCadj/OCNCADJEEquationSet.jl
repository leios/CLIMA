"""
    module OCNCADJEEquationSet
   
 Defines an equation set for an explicit convective adjustment scheme for ocean.

 Defines kernels to evaluates RHS of

  d ϕ / dt = - d/dz κ d/dz ϕ

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
       flux_first_order!,
       flux_second_order!,
       init_state_prognostic!,
       nodal_init_state_auxiliary!,
       source!,
       vars_state,
       wavespeed

using ClimateMachine.Mesh.Geometry: LocalGeometry
using ClimateMachine.MPIStateArrays

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

function prop_defaults()
  bl_prop=NamedTuple()
  bl_prop=( bl_prop...,init_aux_geom=nothing)
  bl_prop=( bl_prop...,   init_theta=nothing)
  bl_prop=( bl_prop..., source_theta=nothing)
end

function vars_state(e::eq_type, ::Prognostic, FT)
  @vars begin
    θ::FT
  end
end
function vars_state(e::eq_type, ::Auxiliary, FT)
  @vars begin
     npt::Int
   elnum::Int
    xc::FT
    yc::FT
    zc::FT
  end
end

function init_state_prognostic!(e::eq_type, Q::Vars, A::Vars, geom::LocalGeometry, FT)
  npt=getproperty(geom,:n)
  elnum=getproperty(geom,:e)
  x=geom.coord[1]
  y=geom.coord[2]
  z=geom.coord[3]
  Q.θ=e.bl_prop.init_theta(x,y,z,npt,elnum)
  nothing
end
function nodal_init_state_auxiliary!(e::eq_type, A::Vars, tmp::Vars, geom::LocalGeometry, _...)
  npt=getproperty(geom,:n)
  elnum=getproperty(geom,:e)
  x=geom.coord[1]
  y=geom.coord[2]
  z=geom.coord[3]
  A.npt, A.elnum, A.xc, A.yc, A.zc = e.bl_prop.init_aux_geom(npt,elnum,x,y,z)
  nothing
end

# function source!(e::eq_type,S::Vars,Q::Vars,G::Vars,A::Vars,t)
function source!(e::eq_type,S::Vars,Q::Vars,G::Vars,A::Vars,_...)
  S.θ=e.bl_prop.source_theta(A.npt,A.elnum,A.xc,A.yc,A.zc)
  nothing
end

function flux_first_order!( e::eq_type, _...)
  nothing
end

function flux_second_order!( e::eq_type, _...)
  nothing
end

function boundary_conditions( e::eq_type, _...)
 ( nothing, )
end

function boundary_state!(nft_x, nft_y, e::eq_type, _...)
 nothing
end

function wavespeed(e::eq_type, _...)
 0
end

end
