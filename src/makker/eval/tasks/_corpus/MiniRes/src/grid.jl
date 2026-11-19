"""
    build_grid(nx, ny)

Build a structured `nx` by `ny` Cartesian grid and return its cell count.
"""
function build_grid(nx::Int, ny::Int)
    return (nx = nx, ny = ny, ncells = nx * ny)
end
