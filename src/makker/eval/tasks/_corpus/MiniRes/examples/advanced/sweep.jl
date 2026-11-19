# Parameter sweep over permeability, calling the Newton solver each time.
using MiniRes

grid = build_grid(20, 20)
for perm in (1.0e-13, 5.0e-13, 1.0e-12)
    res = solve_newton(grid, perm; iters = 8)
    println("perm = ", perm, " -> residual = ", res)
end
