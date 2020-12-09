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
# yOrderedEdgeList=[0,1,2,3]
yOrderedEdgeList=[0,3]
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
 yAmp=1.
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

"""
  Compute and set diffusivity in each direction
"""
function calc_kappa_diff(∇θ,npt,elnum,x,y,z)
  return +0.1, +0.1, +0.1
  # return -0., -0., -0.
end

function get_wavespeed()
  return FT(0.)
end

# Add customizations to properties
bl_prop=OCNCADJEEquationSet.prop_defaults()
bl_prop=(bl_prop...,init_aux_geom=init_aux_geom)
bl_prop=(bl_prop...,   init_theta=init_theta   )
bl_prop=(bl_prop..., source_theta=source_theta )
bl_prop=(bl_prop..., calc_kappa_diff=calc_kappa_diff )
bl_prop=(bl_prop..., get_wavespeed=get_wavespeed )

# Create an equation set with the cutomized function and parameter properties
oml=OCNCADJEEquations{Float64}(;bl_prop=bl_prop)

# Instantiate a DG model with the customized equation set
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
# oml_dg = DGModel(oml,mgrid,RusanovNumericalFlux(),CentralNumericalFluxSecondOrder(),CentralNumericalFluxGradient())
oml_dg = DGModel(oml,mgrid,RusanovNumericalFlux(),PenaltyNumFluxDiffusive(),CentralNumericalFluxGradient())
oml_Q = init_ode_state(oml_dg, FT(0); init_on_cpu = true)
dQ = init_ode_state(oml_dg, FT(0); init_on_cpu = true)

# Execute the DG model
oml_dg(dQ,oml_Q, nothing, 0; increment=false)

println(oml_Q.θ)
println(dQ.θ)

dt=(mgrid.vgeo[2,13,1])^2/0.1*0.5*0.20
println("tic")
for iter=1:100
# println(iter)
oml_Q.θ.=oml_Q.θ-dt.*dQ.θ

# Make some plots
using Plots
using ClimateMachine.Mesh.Elements: interpolationmatrix
nelem = size(dQ)[end]
# fld=dQ.θ
fld=oml_Q.θ
dim = dimensionality(mgrid)
N = polynomialorders(mgrid)
Nq = N .+ 1
sp=range(-0.98;length=40,stop=0.98)
FT=eltype(fld)
ξ = ntuple(
     i -> N[i] == 0 ? FT.([-1, 1]) : referencepoints(mgrid)[i],
     dim,
    )
ξdst=sp
I1d = ntuple(i -> interpolationmatrix(ξ[dim - i + 1], ξdst), dim)
I = kron(I1d...)
global fldsp=I*fld[:,1,:]
global X=ntuple(i -> I*mgrid.x_vtk[i], length(mgrid.x_vtk) )
i=1;plot(reshape(X[1][:,i],(40,40))[:,20],reshape(fldsp[:,i],(40,40))[:,20] )
i=2;plot!(reshape(X[1][:,i],(40,40))[:,20],reshape(fldsp[:,i],(40,40))[:,20] )
i=3;plot!(reshape(X[1][:,i],(40,40))[:,20],reshape(fldsp[:,i],(40,40))[:,20] )
# scatter(X[1],X[2],fldsp,camera=(0,90),zcolor=fldsp,size=(1200,800),label="",markerstrokewidth=0,markershape=:rect)
oml_dg(dQ,oml_Q, nothing, 0; increment=false);
end
println("toc")

i=1;plot(reshape(X[1][:,i],(40,40))[:,20],reshape(fldsp[:,i],(40,40))[:,20] )
i=2;plot!(reshape(X[1][:,i],(40,40))[:,20],reshape(fldsp[:,i],(40,40))[:,20] )
i=3;plot!(reshape(X[1][:,i],(40,40))[:,20],reshape(fldsp[:,i],(40,40))[:,20] )


# Try some timestepping


