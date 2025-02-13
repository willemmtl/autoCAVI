using Distributions, GMRF
using Distributions:loglikelihood

"""
    logposterior(θ; Fmu, Fphi, Fxi, data)
"""
function logposterior(θ::DenseVector; Fmu::iGMRF, Fphi::iGMRF, data::Vector{Vector{Float64}})
    
    M = prod(Fmu.G.gridSize);
    μ = θ[1:M];
    ϕ = θ[M+1:2*M];
    ξ = θ[2*M+1];
    

    return (
        sum(loglikelihood.(GeneralizedExtremeValue.(μ, exp.(ϕ), ξ), data))
        + (prod(Fmu.G.gridSize) - Fmu.rankDeficiency)/2 * log(Fmu.κ) - Fmu.κ/2 * μ' * Fmu.G.W * μ
        + (prod(Fphi.G.gridSize) - Fphi.rankDeficiency)/2 * log(Fphi.κ) - Fphi.κ/2 * ϕ' * Fphi.G.W * ϕ
        + logpdf(Beta(6, 9), ξ + .5)
    )
end


"""
    celllogfullconditional(i, θi; ξ, μ̄i, ϕ̄i, Fmu, Fphi, data)

Log full conditional density of [μi, ϕi] knowing all other parameters.

# Arguments
- `i::Integer`: Numero of the cell.
- `θi::Vector{Float64}`: parameters for cell i -> variables [μi, ϕi].
- `ξ::Real`: Last updated shape parameter.
- `μ̄i::Real`: Neighbors influence for location (iGMRF).
- `ϕ̄i::Real`: Neighbors influence for log-scale (iGMRF).
- `Fmu::iGMRF`: Spatial scheme for location.
- `Fphi::iGMRF`: Spatial scheme for log-scale.
- `data::Vector{Float64}`: Observations for every cells.
"""
function celllogfullconditional(
    i::Integer,
    θi::DenseVector;
    ξ::Real,
    μ̄i::Real,
    ϕ̄i::Real,
    Fmu::iGMRF,
    Fphi::iGMRF,
    data::Vector{Vector{Float64}},
)

    return (
        loglikelihood(GeneralizedExtremeValue(θi[1], exp(θi[2]), ξ), data[i])
        + logpdf(Normal(μ̄i, sqrt(1/Fmu.G.W[i, i]/Fmu.κ)), θi[1])
        + logpdf(Normal(ϕ̄i, sqrt(1/Fphi.G.W[i, i]/Fphi.κ)), θi[2])
    )
end


"""
    xilogfullconditional(ξ; μ, ϕ, data)

Compute the log full conditional of ξ parameter of cell cellIndex.

# Arguments
- `ξ::DenseVector`: Variable.
- `μ::DenseVector`: Value of μ at this cell.
- `ϕ::DenseVector`: Value of ϕ at this cell.
- `data::Vector{Vector{Float64}}`: Observations.
"""
function xilogfullconditional(
    ξ::DenseVector;
    μ::DenseVector,
    ϕ::DenseVector,
    data::Vector{Vector{Float64}},
)
    return (
        sum(loglikelihood.(GeneralizedExtremeValue.(μ, exp.(ϕ), ξ[1]), data))
        + logpdf(Beta(6, 9), ξ[1] + .5)
    )
end


"""
    logfullconditional(i, θi; μ, ϕ, Fmu, Fphi, data)

Define the full conditional of the first cell.

# Arguments
- `i::Integer`: Numero of the cell.
- `θi::Vector{Float64}`: GEV parameters for cell i -> variables [μi, ϕi, ξ].
- `μ::DenseVector`: Up-to-date location parameters.
- `ϕ::DenseVector`: Up-to-date log-scale parameters.
- `Fmu::iGMRF`: Spatial scheme for location.
- `Fphi::iGMRF`: Spatial scheme for log-scale.
- `data::Vector{Vector{Float64}}`: Observations for each cell.
"""
function logfullconditional(
    i::Integer,
    θi::DenseVector;
    μ::DenseVector,
    ϕ::DenseVector,
    Fmu::iGMRF,
    Fphi::iGMRF,
    data::Vector{Vector{Float64}},
)

    μ̄i = neighborsMean(i, μ, Fmu);
    ϕ̄i = neighborsMean(i, ϕ, Fphi);

    llikegev = loglikelihood.(GeneralizedExtremeValue.(μ, exp.(ϕ), θi[3]), data);
    llikegev = vcat(llikegev[1:i-1], llikegev[i+1:end]);

    return (
        sum(llikegev)
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