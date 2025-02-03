using Distributions, GMRF
using Distributions:loglikelihood

"""
    logposterior(θ; Fmu, Fphi, Fxi, data)
"""
function logposterior(θ::DenseVector; Fmu::iGMRF, Fphi::iGMRF, Fxi::iGMRF, data::Vector{Vector{Float64}})
    
    M = prod(Fmu.G.gridSize);
    μ = θ[1:M];
    ϕ = θ[M+1:2*M];
    ξ = θ[2*M+1:3*M];
    

    return (
        sum(loglikelihood.(GeneralizedExtremeValue.(μ, exp.(ϕ), ξ), data))
        + (prod(Fmu.G.gridSize) - Fmu.rankDeficiency)/2 * log(Fmu.κ) - Fmu.κ/2 * μ' * Fmu.G.W * μ
        + (prod(Fphi.G.gridSize) - Fphi.rankDeficiency)/2 * log(Fphi.κ) - Fphi.κ/2 * ϕ' * Fphi.G.W * ϕ
        + (prod(Fxi.G.gridSize) - Fxi.rankDeficiency)/2 * log(Fxi.κ) - Fxi.κ/2 * ξ' * Fxi.G.W * ξ
    )
end


"""
    logfullconditional(i, θi; μ̄i, ϕ̄i, ξ̄i, Fmu, Fphi, Fxi, data)

Define the full conditional of the cell i.

# Arguments
- `i::Integer`: Index of current cell.
- `θi::Vector{Float64}`: GEV parameters for cell i -> variables [μ, ϕ, ξ].
- `μ̄i::Real`: Neighbors influence for location (iGMRF).
- `ϕ̄i::Real`: Neighbors influence for log-scale (iGMRF).
- `ξ̄i::Real::Real`: Neighbors influence for form (iGMRF).
- `Fmu::iGMRF`: Spatial scheme for location.
- `Fphi::iGMRF`: Spatial scheme for log-scale.
- `Fxi::iGMRF`: Spatial scheme for form.
- `data::Vector{Vector{Float64}}`: Observations for each cell.
"""
function logfullconditional(
    i::Integer,
    θi::DenseVector;
    μ̄i::Real,
    ϕ̄i::Real,
    ξ̄i::Real,
    Fmu::iGMRF,
    Fphi::iGMRF,
    Fxi::iGMRF,
    data::Vector{Vector{Float64}},
)

    return (
        loglikelihood(GeneralizedExtremeValue(θi[1], exp(θi[2]), θi[3]), data[i])
        + logpdf(Normal(μ̄i, sqrt(1/Fmu.G.W[i, i]/Fmu.κ)), θi[1])
        + logpdf(Normal(ϕ̄i, sqrt(1/Fphi.G.W[i, i]/Fphi.κ)), θi[2])
        + logpdf(Normal(ξ̄i, sqrt(1/Fxi.G.W[i, i]/Fxi.κ)), θi[3])
    )
end


# """
#     mulogfullconditional(μ, cellIndex; ϕ, ξ, μ̄i, Fmu, data)

# Compute the log full conditional of μ parameter of cell cellIndex.

# # Arguments
# - `μ::Real`: Variable.
# - `cellIndex::Integer`: Index of the current cell.
# - `ϕ::Real`: Value of ϕ at this cell.
# - `ξ::Real`: Value of ξ at this cell.
# - `μ̄i::Real`: Neighbors influence over μ.
# - `Fmu::iGMRF`: Spatial scheme.
# - `data::Vector{Vector{Float64}}`: Observations.
# """
# function mulogfullconditional(
#     μ::Real,
#     cellIndex::Integer;
#     ϕ::Real,
#     ξ::Real,
#     μ̄i::Real,
#     Fmu::iGMRF,
#     data::Vector{Vector{Float64}},
# )
#     return (
#         loglikelihood(GeneralizedExtremeValue(μ, exp(ϕ), ξ), data[cellIndex])
#         + logpdf(Normal(μ̄i, sqrt(1/Fmu.G.W[cellIndex, cellIndex]/Fmu.κ)), μ)
#     )
# end


# """
#     philogfullconditional(ϕ, cellIndex; μ, ξ, ϕ̄i, Fphi, data)

# Compute the log full conditional of ϕ parameter of cell cellIndex.

# # Arguments
# - `ϕ::Real`: Variable.
# - `cellIndex::Integer`: Index of the current cell.
# - `μ::Real`: Value of μ at this cell.
# - `ξ::Real`: Value of ξ at this cell.
# - `ϕ̄i::Real`: Neighbors influence over ϕ.
# - `Fphi::iGMRF`: Spatial scheme.
# - `data::Vector{Vector{Float64}}`: Observations.
# """
# function philogfullconditional(
#     ϕ::Real,
#     cellIndex::Integer;
#     μ::Real,
#     ξ::Real,
#     ϕ̄i::Real,
#     Fphi::iGMRF,
#     data::Vector{Vector{Float64}},
# )
#     return (
#         loglikelihood(GeneralizedExtremeValue(μ, exp(ϕ), ξ), data[cellIndex])
#         + logpdf(Normal(ϕ̄i, sqrt(1/Fphi.G.W[cellIndex, cellIndex]/Fphi.κ)), ϕ)
#     )
# end


# """
#     xilogfullconditional(ξ, cellIndex; μ, ϕ, ξ̄i, Fxi, data)

# Compute the log full conditional of ξ parameter of cell cellIndex.

# # Arguments
# - `ξ::Real`: Variable.
# - `cellIndex::Integer`: Index of the current cell.
# - `μ::Real`: Value of μ at this cell.
# - `ϕ::Real`: Value of ϕ at this cell.
# - `ξ̄i::Real`: Neighbors influence over ξ.
# - `Fxi::iGMRF`: Spatial scheme.
# - `data::Vector{Vector{Float64}}`: Observations.
# """
# function xilogfullconditional(
#     ξ::Real,
#     cellIndex::Integer;
#     μ::Real;
#     ϕ::Real,
#     ξ̄i::Real,
#     Fxi::iGMRF,
#     data::Vector{Vector{Float64}},
# )
#     return (
#         loglikelihood(GeneralizedExtremeValue(μ, exp(ϕ), ξ), data[cellIndex])
#         + logpdf(Normal(ξ̄i, sqrt(1/Fxi.G.W[cellIndex, cellIndex]/Fxi.κ)), ξ)
#     )
# end


"""
    neighborsMean(cellIndex, θ, F)

Compute the iGMRF neighbors influence over cell i for parameter θ.

# Arguments
- `cellIndex::Integer`: Index of current cell.
- `θ::DenseVector`: Values of the given parameter for all cells.
- `F::iGMRF`: Spatial scheme.
"""
function neighborsMean(cellIndex::Integer, θ::DenseVector, F::iGMRF)
    return ((-F.G.W̄) * θ)[cellIndex] / F.G.W[cellIndex, cellIndex]
end


"""
    logapprox(θ, approxMarginals)

Return the log approximation density.
It is the sum of each log density of a cell.

# Arguments
- `θ::DenseVector`: Parameters [μ..., ϕ..., ξ...].
- `approxMarginals::Vector{<:Distribution}`: The marginal distribution of each cell.
"""
function logapprox(θ::DenseVector, approxMarginals::Vector{<:Distribution})

    M = length(θ) ÷ 3;

    return sum([
        logpdf(approxMarginals[i], [θ[i], θ[M+i], θ[2*M+i]])
        for i=1:M
    ]);
end