# Minimal end-to-end example: build a grid, place a well, inspect it.
using MiniRes

grid = build_grid(10, 10)
well = setup_well(grid, 5, 5)
println("well at cell index ", well.index, " on a ", grid.ncells, " cell grid")
