# MiniRes API

- `build_grid(nx, ny)` - construct a Cartesian grid.
- `setup_well(grid, i, j)` - place a vertical well.
- `solve_newton(grid, perm; iters)` - run the Newton pressure solver.

Internal kernels such as the Darcy flux are not part of the public API.
