---
name: jutuldarcy-geostatistics
description: Model spatially-correlated property fields with GeoStats.jl, condition them on well data, and feed them into JutulDarcy
---

# Geostatistics for JutulDarcy

## When to use

Use this skill when a property varies smoothly across the grid and you want to model that spatial structure instead of assigning a uniform or independent-random value per cell. It applies whenever the field must honor measured data, and whenever you need to quantify the uncertainty that remains away from the data.

Typical tasks:

- Model a continuous property such as porosity, log-permeability, or net-to-gross that varies over a spatial correlation length.
- Build a field that reproduces values measured at wells or along logs.
- Produce an ensemble of equally probable realizations that represents the geological uncertainty.

`GeoStats.jl` is a dependency of the JutulDarcy environment, so `using GeoStats` works directly. GeoStats builds the field. JutulDarcy can then use it as a per-cell array. See `jutuldarcy-overview` for the surrounding simulation workflow and `ensembles` for running realizations in parallel.

A common path is to select a variogram, build an estimate or an ensemble of realizations that honor the data, and quantify the resulting uncertainty. Propagating the field through a JutulDarcy simulation is one option, not a requirement. The sections below are independent, so use the ones your task needs.

## Data and grid

`georef` attaches values to coordinates. The coordinates must live in the same space as the target grid. On a unit-cell `CartesianGrid(nx, ny)`, logical cell `(i, j)` is centered at `(i-0.5, j-0.5)`.

```julia
using GeoStats

wells = [(10, 10), (40, 12), (25, 38)]       # logical (i, j) cells
phi   = [0.28, 0.21, 0.30]                   # a measured property at those cells
data  = georef((; porosity = phi), [(i - 0.5, j - 0.5) for (i, j) in wells])
grid  = CartesianGrid(50, 50)
```

A `CartesianGrid(nx, ny)` and a JutulDarcy `reservoir_mesh((nx, ny, 1), ...)` use the same column-major cell order, with `i` varying fastest. A field vector of length `nx*ny` therefore maps onto a per-cell array without reshuffling. Use `reshape(vec(field), nx, ny)` to view it as a map. Keep the grid dimensions identical on both sides. A wrong reshape silently scrambles the field.

## Variograms

A variogram is the model of spatial correlation. It is described by a `range`, which is the correlation length, a `sill`, which is the variance plateau, and a `nugget`, which is micro-scale variance or measurement noise.

```julia
SphericalVariogram(range = 15.0)                 # finite range
GaussianVariogram(range = 15.0, nugget = 0.001)  # very smooth fields
ExponentialVariogram(range = 15.0)               # rougher fields
```

A reasonable `range` is a fraction of the domain extent. When you have enough data, fit a variogram to an empirical estimate rather than guessing. `EmpiricalVariogram(data, :porosity; maxlag = 25.0)` computes the empirical estimate to fit.

The `sill` and the process mean set the scale of the field. Choose them on the scale of the property you are modeling, using the data statistics or a geological prior. A default `GaussianProcess(γ)` has mean 0 and sill 1. That is a standardized field, which is appropriate only when you model a normal-score variable and back-transform afterwards.

## Estimation and simulation

Estimation and simulation serve different purposes. Choose the one that matches your question.

Kriging estimation returns a predicted value at each cell together with a kriging variance, which is the formal estimation error. It is the best linear unbiased estimate, and it gives one smooth deterministic map. It smooths between the data and does not reproduce the true spatial variability, so it is a model of the trend rather than of the full uncertainty.

```julia
itp = data |> Interpolate(grid, model = Kriging(SphericalVariogram(range = 15.0)))
itp.porosity                                       # the estimated field
```

Conditional simulation returns many equally probable realizations. Each realization honors the data and reproduces the spatial variability described by the variogram. The realizations differ where the data does not constrain them. This is the basis for uncertainty quantification, which is usually the central goal of a geostatistical study. You analyze the spread across the realizations rather than any single one.

```julia
mu   = sum(phi) / length(phi)                      # mean on the property's scale
proc = GaussianProcess(SphericalVariogram(range = 15.0, sill = 0.04^2), mu)

real = rand(proc, grid, 100; data = data)          # conditional: honors phi at the wells
# rand(proc, grid, 100)                             # unconditional: ignores the data
```

`real[k]` is the k-th realization, which is a `GeoTable`. After conditioning, its column takes the data's name, which is `real[k].porosity` here. Without conditioning the column is named `:field`. Use `propertynames(real[1])` if you are unsure. Draw enough realizations that the ensemble statistics are stable. Confirming that the realizations honor the data says nothing about the field between the wells, so inspect `extrema(real[1].porosity)` to check that the values stay physical.

## Plotting a field with the wells on top

`reshape(vec(field), nx, ny)[i, j]` is the cell centered at `(i-0.5, j-0.5)`. There are two consistent ways to draw the field and overlay the data so the markers land on the cells they condition. Choose one and stay within it.

The first uses index space with a Makie `heatmap`. Plot the reshaped field and scatter the wells at their integer cell `(i, j)`.

```julia
M = reshape(vec(real[1].porosity), nx, ny)        # a realization as a map
heatmap(M)
scatter!([Float64(i) for (i, j) in wells], [Float64(j) for (i, j) in wells])
```

The second uses coordinate space with the GeoStats `viz` recipe. `viz(domain(real[1]), color = real[1].porosity)` draws each cell at its true coordinate. Scatter the wells at `(i-0.5, j-0.5)`, which are the coordinates you passed to `georef`.

Two mistakes put the data in the wrong place. Transposing the reshaped field with `M'` or `reshape(field, ny, nx)` scrambles it. Mixing the two spaces offsets the markers by half a cell, which happens if you scatter at `(i-0.5, j-0.5)` on an index-space heatmap or at `(i, j)` on a `viz` plot. As a check, confirm that `reshape(vec(field), nx, ny)[i, j]` equals the observation at well `(i, j)`. When you run as the agent, render through the `plot_julia` tool so the figures are saved with the session (`plotting-basics`).

## Quantifying uncertainty

Report uncertainty as a field rather than as a single number. The ensemble of realizations carries the information, so summarize it across the realizations at each cell.

The ensemble standard deviation is a map. Compute it with `std` or `var` over the realizations. It is near zero at the data locations and grows toward the sill where there is no data. Show this map. A domain-averaged value hides the effect, because conditioning reduces uncertainty only near the data. Quantile maps such as `quantile(real, 0.1)` and `quantile(real, 0.9)` give a low and a high scenario that meet at the wells and separate between them. These maps are usually the main product of the analysis.

For uncertainty, prefer the simulation ensemble over a kriging variance. In this GeoStats version the variance returned by `Interpolate(..., prob = true)` is unreliable, and the ensemble also captures the full spatial structure that a per-cell kriging variance does not.

## Using the field in JutulDarcy

Model the variable you have data for, then derive the property the simulation needs.

You can use the modeled field directly, for example a porosity field used as `porosity`. You can model a transformed variable when that is more natural. Permeability is positive and roughly log-normal, so a common choice is to model `log10(k)`, condition on well log-permeability, and back-transform with `10 .^ field`. You can also derive one property from another through a relation. A porosity-to-permeability transform such as Kozeny-Carman is one example, and `five_spot_ensemble.jl` shows a worked instance. Correlated properties can be co-simulated jointly.

Keep permeability in SI units of m^2 and porosity within 0 to 1. A light `clamp` guards the bounds. Run the case validation from `jutuldarcy-overview` before simulating, since it flags a property left in the wrong units.

To propagate uncertainty, turn each realization into a case and simulate it. Run the realizations with `run_ensemble` (`ensembles`) and validate every case before `simulate_reservoir`. How input uncertainty maps to an output depends on the model and the question, so measure the output distribution instead of assuming its shape. As one observed example, for a single injector and producer the data tends to re-center the forecast more than it narrows the spread, because the flow response integrates the whole path while the wells constrain only their neighborhoods.

## Discovery / probes

`GeoStats` is a meta-package that re-exports `GeoTables`, `Meshes`, `GeoStatsFunctions` for variograms, `GeoStatsModels` for kriging, and `GeoStatsProcesses` for simulation. Find APIs and read a worked example on disk.

```text
run_julia("@doc GaussianProcess")
read_file(joinpath(pkgdir(JutulDarcy), "examples/workflow/five_spot_ensemble.jl"))
```

That example builds an unconditional porosity ensemble. Conditioning on data with the `data=` keyword shown above is the step it leaves out. The online documentation covers kriging, anisotropy, 3D, indicator and categorical facies, and co-simulation at `https://juliaearth.github.io/GeoStatsDocs/stable`.
