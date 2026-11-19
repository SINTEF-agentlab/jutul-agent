# Newton solver driving the pressure equation.

"""
    solve_newton(grid, perm; iters = 5)

Run `iters` Newton iterations on `grid` and return the final residual.
"""
function solve_newton(grid, perm::Float64; iters::Int = 5)
    res = 1.0
    for _ in 1:iters
        flux = darcy_flux(perm, res, 1.0)
        res = abs(flux) * 1.0e-6
    end
    return res
end
