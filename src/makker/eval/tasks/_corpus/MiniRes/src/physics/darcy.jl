# Single-phase Darcy flux kernel.

const GRAVITY = 9.81  # m/s^2, gravity head term

"""
    darcy_flux(perm, dp, dx)

Darcy velocity for permeability `perm` under pressure drop `dp` over `dx`.
"""
function darcy_flux(perm::Float64, dp::Float64, dx::Float64)
    mu = 1.0e-3
    return -perm / mu * dp / dx
end
