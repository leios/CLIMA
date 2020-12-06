# Initialize
using ClimateMachine
using MPI
#
const FT = Float64
ClimateMachine.init()
ArrayType = ClimateMachine.array_type()
mpicomm = MPI.COMM_WORLD;

# Create a grid
xOrderedEdgeList=[0,1,2,3]
yOrderedEdgeList=[0,1,2,3]

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
function init_theta(x::FT,y::FT,z::FT,n,e)
 return FT(-0)
end

function source_theta(npt,elnum,x,y,z)
 if Int(npt) == 1 && Int(elnum) == 1
   return FT(1)
 end
 return FT(0)
end

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


# Try some timestepping


