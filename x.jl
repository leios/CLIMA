# Initialize
using ClimateMachine
using MPI
#
const FT = Float64
ClimateMachine.init()
ArrayType = ClimateMachine.array_type()
mpicomm = MPI.COMM_WORLD;

# Create a grid and save grid parameters
xOrderedEdgeList=[0,1,2,3]
yOrderedEdgeList=[0,1,2,3]
f(x)=0.5*(x[1]+x[end])
xmid=f(xOrderedEdgeList)
ymid=f(yOrderedEdgeList)
f(x)=x[end]-x[1]
Lx=f(xOrderedEdgeList)
Ly=f(yOrderedEdgeList)

using ClimateMachine.Mesh.Topologies
brickrange=( xOrderedEdgeList, yOrderedEdgeList )
topl = BrickTopology(
       mpicomm,
       brickrange;
       periodicity=(true,true),
)

Np=4
using ClimateMachine.Mesh.Grids
mgrid = DiscontinuousSpectralElementGrid(
     topl,
     FloatType = FT,
     DeviceArray = ArrayType,
     polynomialorder = Np,
)

# Import an equation set template
include("test/Ocean/OcnCadj/OCNCADJEEquationSet.jl")
using ..OCNCADJEEquationSet

# Set up custom function and parameter options as needed
"""
  θ(t=0)=0
  θ(t=0)=exp(-((x-x0)/L0x)^2).exp(-((y-y0)/L0y)^2)
"""
const xDecayLength=FT(Lx/6)
const yDecayLength=FT(Ly/6)
function init_theta(x::FT,y::FT,z::FT,n,e)
 xAmp=exp(  -( ( (x - xmid)/xDecayLength )^2 )  )
 yAmp=exp(  -( ( (y - ymid)/yDecayLength )^2 )  )
 return FT(xAmp*yAmp)
end

"""
  θˢᵒᵘʳᶜᵉ(1,1)=1
"""
function source_theta(θ,npt,elnum,x,y,z)
 if Int(npt) == 1 && Int(elnum) == 1
   return FT(0)
 end
 return FT(0)
end

"""
  Save array indexing and real world coords
"""
function init_aux_geom(npt,elnum,x,y,z)
 return npt,elnum,x,y,z
end

# Add customizations to properties
bl_prop=OCNCADJEEquationSet.prop_defaults()
bl_prop=(bl_prop...,init_aux_geom=init_aux_geom)
bl_prop=(bl_prop...,   init_theta=init_theta   )
bl_prop=(bl_prop..., source_theta=source_theta )

# Create an equation set with the cutomized function and parameter properties
oml=OCNCADJEEquations{Float64}(;bl_prop=bl_prop)

# Instantiate a DG model with the customized equation set
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
oml_dg = DGModel(oml,mgrid,RusanovNumericalFlux(),CentralNumericalFluxSecondOrder(),CentralNumericalFluxGradient())
oml_Q = init_ode_state(oml_dg, FT(0); init_on_cpu = true)
dQ = init_ode_state(oml_dg, FT(0); init_on_cpu = true)

# Execute the DG model
oml_dg(dQ,oml_Q, nothing, 0; increment=false)

println(oml_Q.θ)
println(dQ.θ)


# Try some timestepping


