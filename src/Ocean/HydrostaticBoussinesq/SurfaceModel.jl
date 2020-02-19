struct SurfaceModel{T} <: BalanceLaw
    cʰ::T
    νʰ::T
    κʰ::T
    function SurfaceModel{FT}(m::HydrostaticBoussinesqModel)
        return new{FT}(m.cʰ, m.νʰ, m.κʰ)
    end
end

function vars_state(m::SurfaceModel, T)
    @vars begin
        η::T
    end
end

function vars_aux(m::SurfaceModel, T)
    @vars begin
        wz0::T
    end
end

vars_gradient(m::SurfaceModel, T) = @vars()
vars_diffusive(m::SurfaceModel, T) = @vars()
vars_integrals(m::SurfaceModel, T) = @vars()

@inline flux_nondiffusive!(::SurfaceModel, ...) = nothing
@inline flux_diffusive!(::SurfaceModel, ...) = nothing

@inline function source!(m::SurfaceModel, S::Vars, Q::Vars, A::Vars, t::Real)
    @inbounds begin
        S.η += A.wz0
    end
end

@inline function do_integrals_and_pass_from_slow_to_fast!(Qfast, dQslow, fast::SurfaceModel, slow::HydrostaticBoussinesqModel)
    ### need to copy w at z=0 into wz0
end

@inline function reconcile_fast_to_slow(Qslow, Qfast, slow::HydrostaticBoussinesqModel, fast::SurfaceModel)
  ### need to copy η to aux for 3D

  # project w(z=0) down the stack
  # Need to be consistent with vars_aux
  # A[1] = w, A[5] = wz0
    copy_stack_field_down!(dg, m, A, 1, 5)
  ### need to change to jeremy's new function
end
