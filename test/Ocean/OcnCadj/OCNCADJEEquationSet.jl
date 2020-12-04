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
      Prognostic,
      init_state_prognostic!,
      source!,
      vars_state

import ClimateMachine.BalanceLaws:
       init_state_prognostic!,
       source!,
       vars_state

using ClimateMachine.Mesh.Geometry: LocalGeometry

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
  bl_prop=( bl_prop..., init_theta=nothing)
end

function vars_state(e::eq_type, ::Prognostic, FT)
  @vars begin
    θ::FT
  end
end

function init_state_prognostic!(e::eq_type, Q::Vars, A::Vars, geom::LocalGeometry, FT)
  x=geom.coords[1]
  y=geom.coords[1]
  z=geom.coords[1]
  Q.θ=e.bl_prop.init_theta(x,y,z)
  nothing
end

function source!(e::eq_type, ::Prognostic, FT)
  nothing
end

end
