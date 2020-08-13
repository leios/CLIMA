"""
    shallow_boundary_state!(::NumericalFluxFirstOrder, ::Impenetrable{FreeSlip}, ::SWModel)

apply free slip boundary condition for velocity
sets reflective ghost point
"""
@inline function shallow_boundary_state!(
    ::NumericalFluxFirstOrder,
    bc::Impenetrable{FreeSlip},
    m::SWModel,
    ::TurbulenceClosure,
    q⁺,
    a⁺,
    n⁻,
    q⁻,
    a⁻,
    t,
)
    q⁺.η = q⁻.η

    V⁻ = @SVector [q⁻.U[1], q⁻.U[2], -0]
    V⁺ = V⁻ - 2 * n⁻ ⋅ V⁻ .* SVector(n⁻)
    q⁺.U = @SVector [V⁺[1], V⁺[2]]

    return nothing
end

"""
    shallow_boundary_state!(::Union{NumericalFluxGradient, NumericalFluxSecondOrder}, ::Impenetrable{FreeSlip}, ::SWModel)

no second order flux computed for linear drag
"""
shallow_boundary_state!(
    ::Union{NumericalFluxGradient, NumericalFluxSecondOrder},
    ::VelocityBC,
    m::SWModel,
    ::LinearDrag,
    _...,
) = nothing

"""
    shallow_boundary_state!(::NumericalFluxGradient, ::Impenetrable{FreeSlip}, ::SWModel)

apply free slip boundary condition for velocity
sets non-reflective ghost point
"""
function shallow_boundary_state!(
    ::NumericalFluxGradient,
    bc::Impenetrable{FreeSlip},
    m::SWModel,
    ::ConstantViscosity,
    Q⁺,
    A⁺,
    n⁻,
    Q⁻,
    A⁻,
    t,
)
    V⁻ = @SVector [Q⁻.U[1], Q⁻.U[2], -0]
    V⁺ = V⁻ - n⁻ ⋅ V⁻ .* SVector(n⁻)
    Q⁺.U = @SVector [V⁺[1], V⁺[2]]

    return nothing
end

"""
    shallow_normal_boundary_flux_second_order!(::NumericalFluxSecondOrder, ::Impenetrable{FreeSlip}, ::SWModel)

apply free slip boundary condition for velocity
apply zero numerical flux in the normal direction
"""
function shallow_boundary_state!(
    ::NumericalFluxSecondOrder,
    bc::Impenetrable{FreeSlip},
    m::SWModel,
    ::ConstantViscosity,
    Q⁺,
    D⁺,
    A⁺,
    n⁻,
    Q⁻,
    D⁻,
    A⁻,
    t,
)
    Q⁺.U = Q⁻.U
    D⁺.ν∇U = n⁻ * (@SVector [-0, -0])'

    return nothing
end

"""
    shallow_boundary_state!(::NumericalFluxFirstOrder, ::Impenetrable{NoSlip}, ::SWModel)

apply no slip boundary condition for velocity
sets reflective ghost point
"""
@inline function shallow_boundary_state!(
    ::NumericalFluxFirstOrder,
    bc::Impenetrable{NoSlip},
    m::SWModel,
    ::TurbulenceClosure,
    q⁺,
    α⁺,
    n⁻,
    q⁻,
    α⁻,
    t,
)
    q⁺.η = q⁻.η
    q⁺.U = -q⁻.U

    return nothing
end

"""
    shallow_boundary_state!(::NumericalFluxGradient, ::Impenetrable{NoSlip}, ::SWModel)

apply no slip boundary condition for velocity
set numerical flux to zero for U
"""
@inline function shallow_boundary_state!(
    ::NumericalFluxGradient,
    bc::Impenetrable{NoSlip},
    m::SWModel,
    ::ConstantViscosity,
    q⁺,
    α⁺,
    n⁻,
    q⁻,
    α⁻,
    t,
)
    FT = eltype(q⁺)
    q⁺.U = @SVector zeros(FT, 3)

    return nothing
end

"""
    shallow_boundary_state!(::NumericalFluxSecondOrder, ::Impenetrable{NoSlip}, ::SWModel)

apply no slip boundary condition for velocity
sets ghost point to have no numerical flux on the boundary for U
"""
@inline function shallow_boundary_state!(
    ::NumericalFluxSecondOrder,
    ::Impenetrable{NoSlip},
    m::SWModel,
    ::ConstantViscosity,
    q⁺,
    σ⁺,
    α⁺,
    n⁻,
    q⁻,
    σ⁻,
    α⁻,
    t,
)
    q⁺.U = -q⁻.U
    σ⁺.ν∇U = σ⁻.ν∇U

    return nothing
end
