using Random, Distributions, SparseArrays, GMRF

"""
    generateData(grid_params, nobs)

Generate fake observations for every grid cell from a given grid of parameters.

# Arguments
- `grid_params::Array{Float64, 3}`: Concatenated grids of values for each parameter of the GEV.
- `nobs::Integer`: Number of fake observations to generate.
"""
function generateData(grid_params::Array{Float64, 3}, nobs::Integer)

    Y = Vector{Float64}[]

    for i = 1:size(grid_params, 1)
        for j = 1:size(grid_params, 2)
            gev_params = grid_params[i, j, :];
            y = rand(GeneralizedExtremeValue(gev_params...), nobs);
            push!(Y, y);
        end
    end

    return Y
end


"""
    generateTargetGrid(F)

Create the grids of values for each parameter of the GEV.

For now the log-scale (ϕ) is fixed to 1 and form (ξ) to 0.

# Arguments
- `Fmu::iGMRF`: Prior of the location parameters.
- `Fphi::iGMRF`: Prior of the scale parameters.
"""
function generateTargetGrid(Fmu::iGMRF, Fphi::iGMRF)
    
    μ = generateParams(Fmu);
    ϕ = generateParams(Fphi);

    realxi = rand(Beta(6, 9)) - .5;
    ξ = fill(realxi, Fmu.G.gridSize...);
    
    return cat(μ, exp.(ϕ), ξ, dims=3)
end


"""
    generateParams(F::iGMRF)

Return a sample of a given iGMRF and reshape it to create a grid of true values for a given GEV parameter.

# Arguments
- `F::iGMRF`: The iGMRF to sample from.
"""
function generateParams(F::iGMRF)
    
    s = rand(F);
    
    return reshape(s, F.G.gridSize...)'
end