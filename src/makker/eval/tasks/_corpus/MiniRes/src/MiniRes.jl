module MiniRes

# Toy reservoir simulator. Public API: build_grid / setup_well /
# solve_newton. The Darcy flux kernel is internal (not exported).

include("grid.jl")
include("physics/darcy.jl")
include("physics/wells.jl")
include("solver/newton.jl")

export build_grid, setup_well, solve_newton

end # module
