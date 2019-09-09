#### Moisture component in atmosphere model
abstract type MoistureModel end

vars_state(::MoistureModel, T) = @vars()
vars_gradient(::MoistureModel, T) = @vars()
vars_diffusive(::MoistureModel, T) = @vars()
vars_aux(::MoistureModel, T) = @vars()

function update_aux!(::MoistureModel, state::Vars, diffusive::Vars, aux::Vars, t::Real)
end
function diffusive!(::MoistureModel, diffusive, ∇transform, state, aux, t, ν, inv_Pr_turb)
end
function flux_diffusive!(::MoistureModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
end
function flux_nondiffusive!(::MoistureModel, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
end
function gradvariables!(::MoistureModel, transform::Vars, state::Vars, aux::Vars, t::Real)
end

@inline function internal_energy(m::MoistureModel, state::Vars, aux::Vars)
  MoistThermodynamics.internal_energy(state.ρ, state.ρe, state.ρu, aux.orientation.Φ)
end
@inline temperature(m::MoistureModel, state::Vars, aux::Vars) = air_temperature(thermo_state(m, state, aux))
@inline pressure(m::MoistureModel, state::Vars, aux::Vars) = air_pressure(thermo_state(m, state, aux))
@inline soundspeed(m::MoistureModel, state::Vars, aux::Vars) = soundspeed_air(thermo_state(m, state, aux))

"""
    DryModel

Assumes the moisture components is in the dry limit.
"""
struct DryModel <: MoistureModel
end

vars_aux(::DryModel,T) = @vars(θ_v::T)
@inline function update_aux!(m::DryModel, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  e_int = internal_energy(m, state, aux)
  TS = PhaseDry(e_int, state.ρ)
  aux.moisture.θ_v = virtual_pottemp(TS)
  nothing
end

thermo_state(m::DryModel, state::Vars, aux::Vars) = PhaseDry(internal_energy(m, state, aux), state.ρ)

"""
    EquilMoist

Assumes the moisture components are computed via thermodynamic equilibrium.
"""
struct EquilMoist <: MoistureModel
end
vars_state(::EquilMoist,T) = @vars(ρq_tot::T)
vars_gradient(::EquilMoist,T) = @vars(q_tot::T, h_tot::T)
vars_diffusive(::EquilMoist,T) = @vars(ρd_q_tot::SVector{3,T}, ρd_h_tot::SVector{3,T})
vars_aux(::EquilMoist,T) = @vars(temperature::T, θ_v::T, q_liq::T)

@inline function update_aux!(m::EquilMoist, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  e_int = internal_energy(m, state, aux)
  TS = PhaseEquil(e_int, state.moisture.ρq_tot/state.ρ, state.ρ)
  aux.moisture.temperature = air_temperature(TS)
  aux.moisture.θ_v = virtual_pottemp(TS)
  aux.moisture.q_liq = PhasePartition(TS).liq
  nothing
end

function thermo_state(m::EquilMoist, state::Vars, aux::Vars)
  e_int = internal_energy(m, state, aux)
  PhaseEquil(e_int, state.moisture.ρq_tot/state.ρ, state.ρ, aux.moisture.temperature)
end

function gradvariables!(m::EquilMoist, transform::Vars, state::Vars, aux::Vars, t::Real)
  ρinv = 1/state.ρ
  transform.moisture.q_tot = state.moisture.ρq_tot * ρinv
  phase = thermo_state(m, state, aux)
  R_m = gas_constant_air(phase)
  T = aux.moisture.temperature
  e_tot = state.ρe * ρinv
  transform.moisture.h_tot = e_tot + R_m*T
end

function diffusive!(m::EquilMoist, diffusive::Vars, ∇transform::Grad, state::Vars, aux::Vars, t::Real, ρν::Union{Real,AbstractMatrix}, inv_Pr_turb::Real)
  # turbulent Prandtl number
  diag_ρν = ρν isa Real ? ρν : diag(ρν) # either a scalar or matrix
  # Diffusivity Dₜ = ρν/Prandtl_turb
  ρD_T = diag_ρν * inv_Pr_turb
  # diffusive flux of q_tot
  diffusive.moisture.ρd_q_tot = (-ρD_T) .* ∇transform.moisture.q_tot
  # diffusive flux of total energy
  diffusive.moisture.ρd_h_tot = (-ρD_T) .* ∇transform.moisture.h_tot
end

function flux_diffusive!(m::EquilMoist, flux::Grad, state::Vars, diffusive::Vars, aux::Vars, t::Real)
  u = state.ρu / state.ρ
  flux.ρ += diffusive.moisture.ρd_q_tot
  flux.ρu += diffusive.moisture.ρd_q_tot .* u'
  flux.ρe += diffusive.moisture.ρd_h_tot
  flux.moisture.ρq_tot += diffusive.moisture.ρd_q_tot
end
