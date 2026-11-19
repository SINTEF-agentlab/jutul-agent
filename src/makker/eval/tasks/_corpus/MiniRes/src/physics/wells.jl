# Well models. Each well evaluates the Darcy flux at its perforation.

"""
    setup_well(grid, i, j)

Place a vertical well in cell (i, j) of `grid`.
"""
function setup_well(grid, i::Int, j::Int)
    return (cell = (i, j), index = (j - 1) * grid.nx + i)
end

function well_rate(well, perm, dp, dx)
    # The well rate is proportional to the local Darcy flux.
    return darcy_flux(perm, dp, dx) * 1.0
end
