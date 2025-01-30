using Distributions
using Distributions:loglikelihood

"""
    logposterior(μ, ϕ, ξ; Fmu, Fphi, data)
"""
function logposterior(μ::DenseVector, ϕ::DenseVector, ξ::Real; Fmu::iGMRF, Fphi::iGMRF, data::Vector{Vector{Float64}})
    return (
        sum(loglikelihood.(GeneralizedExtremeValue.(μ, exp.(ϕ), ξ), data))
        + (prod(Fmu.G.gridSize) - Fmu.rankDeficiency)/2 * log(10) - 10/2 * μ' * Fmu.G.W * μ
        + (prod(Fphi.G.gridSize) - Fphi.rankDeficiency)/2 * log(100) - 100/2 * ϕ' * Fphi.G.W * ϕ
        + logpdf(Beta(6, 9), ξ + .5)
    )
end


"""
    logfullconditional(i, θi; μ, ϕ, μ̄i, ϕ̄i, Fmu, Fphi, data)

Define the full conditional of the cell i.

# Arguments
- `i::Integer`: Index of current cell.
- `θi::Vector{Float64}`: GEV parameters for cell i -> variables [μ, ϕ, ξ].
- `μ::DenseVector`: Location parameter values of every cell.
- `ϕ::DenseVector`: Log-scale parameter values of every cell.
- `μ̄i::Real`: Neighbpors iinfluence for location (iGMRF).
- `ϕ̄i::Real`: Neighbpors iinfluence for log-scale (iGMRF).
- `Fmu::iGMRF`: Spatial scheme for location.
- `Fphi::iGMRF`: Spatial scheme for log-scale.
- `data::Vector{Vector{Float64}}`: Observations for each cell.
"""
function logfullconditional(
    i::Integer,
    θi::DenseVector;
    μ::DenseVector,
    ϕ::DenseVector,
    μ̄i::Real,
    ϕ̄i::Real,
    Fmu::iGMRF,
    Fphi::iGMRF,
    data::Vector{Vector{Float64}},
)

    loglikegev = loglikelihood.(GeneralizedExtremeValue.(μ, exp.(ϕ), θi[3]), data);

    return (
        sum([loglikegev[1:i-1]..., loglikegev[i+1:end]...])
        + loglikelihood(GeneralizedExtremeValue(θi[1], exp(θi[2]), θi[3]), data[i])
        + logpdf(Normal(μ̄i, sqrt(1/Fmu.G.W[i, i]/Fmu.κ)), θi[1])
        + logpdf(Normal(ϕ̄i, sqrt(1/Fphi.G.W[i, i]/Fphi.κ)), θi[2])
        + logpdf(Beta(6, 9), θi[3] + .5)
    )
end


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
    xilogfullconditional(ξ; μ, ϕ, data)

Define the full conditional of the cell i.

# Arguments
- `ξ::Real`: Form parameter = variable.
- `μ::DenseVector`: Location parameter values of every cell.
- `ϕ::DenseVector`: Log-scale parameter values of every cell.
- `data::Vector{Vector{Float64}}`: Observations for each cell.
"""
function xilogfullconditional(
    ξ::Real;
    μ::DenseVector,
    ϕ::DenseVector,
    data::Vector{Vector{Float64}},
)

    return (
        sum(loglikelihood.(GeneralizedExtremeValue.(μ, exp.(ϕ), ξ), data))
        + logpdf(Beta(6, 9), ξ + .5)
    )
end