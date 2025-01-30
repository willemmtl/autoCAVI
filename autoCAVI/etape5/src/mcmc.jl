using GMRF, ProgressMeter, LinearAlgebra, Mamba
using Distributions:loglikelihood

"""
    mcmc(datastructure, niter, initialvalues, stepsize)

Metropolis algorithm.
Use Markov asumption to parallelize calculations.

# Arguments
- `datastructure::Dict`: Data and spatial schemes.
- `niter::Integer`: Number of iterations.
- `initialvalues::Dict`: Initial value for each parameter.
- `stepsize::Dict`: Instrumental variance's step size for each parameter. 
"""
function mcmc(datastructure::Dict, niter::Integer, initialvalues::Dict, stepsize::Dict)

    Y = datastructure[:Y];
    Fmu = datastructure[:Fmu];
    Fphi = datastructure[:Fphi];
    Fxi = datastructure[:Fxi];

    M = prod(Fmu.G.gridSize);

    μ = Array{Float64}(undef, M, niter);
    ϕ = Array{Float64}(undef, M, niter);
    ξ = Array{Float64}(undef, M, niter);

    # Initialisation

    μ[:, 1] = initialvalues[:μ];
    ϕ[:, 1] = initialvalues[:ϕ];
    ξ[:, 1] = initialvalues[:ξ];

    @showprogress for j=2:niter
        
        μ[:, j] = μ[:, j-1];
        ϕ[:, j] = ϕ[:, j-1];
        ξ[:, j] = ξ[:, j-1];

        δ = randn(M);

        # Pour μ
        μ̃ = μ[:, j] .+ stepsize[:μ] .* δ;
        logL = datalevel_loglike.(Y, μ̃, ϕ[:, j], ξ[:, j]) - datalevel_loglike.(Y, μ[:, j], ϕ[:, j], ξ[:, j]);
        μ[:, j] = vertices_update(μ[:, j], Fmu, μ̃, logL);

        # Pour ϕ
        ϕ̃ = ϕ[:, j] .+ stepsize[:ϕ] .* δ;
        logL = datalevel_loglike.(Y, μ[:, j], ϕ̃, ξ[:, j]) - datalevel_loglike.(Y, μ[:, j], ϕ[:, j], ξ[:, j]);
        ϕ[:, j] = vertices_update(ϕ[:, j], Fphi, ϕ̃, logL);
        
        # Pour ξ
        ξ̃ = ξ[:, j] .+ stepsize[:ξ] .* δ;
        logL = datalevel_loglike.(Y, μ[:, j], ϕ[:, j], ξ̃) - datalevel_loglike.(Y, μ[:, j], ϕ[:, j], ξ[:, j]);
        ξ[:, j] = vertices_update(ξ[:, j], Fxi, ξ̃, logL);

    end

    return createChain(M, μ, ϕ, ξ)

end


"""
    datalevel_loglike(Y, μ, ϕ, ξ)

Compute the likelihood of the GEV parameters μ, ϕ and ξ as function of the data Y.

# Arguments
- `Y` : Vector of data.
- `μ` : GEV location parameter.
- `ϕ` : GEV log-scale parameter.
- `ξ` : GEV shape parameter.
"""
function datalevel_loglike(Y::DenseVector, μ::Real, ϕ::Real, ξ::Real)
    return loglikelihood(GeneralizedExtremeValue(μ, exp(ϕ), ξ), Y)
end


"""
    vertices_update(θ, F, θ̃, logL)

Update the vertices using the proposed candidates with data.

# Arguments
- `θ::DenseVector`: The current state.
- `F::iGMRF`: Spatial scheme.
- `θ̃::DenseVector`: Proposed candidates.
- `logL::DenseVector`: The log-likelihood difference between the proposed and the current states at every vertice.
"""
function vertices_update(θ::DenseVector, F::iGMRF, θ̃::DenseVector, logL::DenseVector)

    for indices in F.G.condIndSubset
        vertices_update!(θ, F, θ̃, logL, indices);
    end

    return θ

end


"""
    vertices_update!(θ, F, θ̃, logL, ind)

Update the vertices within a conditional independant subset using the proposed candidates with data.
The vertices are assumed to be conditionally independant, so they can be updated in parallel.

# Arguments
- `θ::DenseVector`: The current state.
- `F::iGMRF`: Spatial scheme.
- `θ̃::DenseVector`: Proposed candidates.
- `logL::DenseVector`: The log-likelihood difference between the proposed and the current states at every vertice.
- `ind::Vector{Integer}`: Indices of the current conditional independant subset.
"""
function vertices_update!(θ::DenseVector, F::iGMRF, θ̃::DenseVector, logL::DenseVector, ind::Vector{Int64})
    
    pds = fullconditionalsIGMRF(F, θ)[ind];
    lf = logpdf.(pds, θ̃[ind]) .- logpdf.(pds, θ[ind]);
    lr = logL[ind] .+ lf;

    logu = log.(rand(length(ind)));

    accepted = lr .> logu;
    
    setindex!(θ, θ̃[ind][accepted], ind[accepted]);

end


"""
    fullconditionalsIGMRF(F, θ)

Compute the probability density of the full conditional function of the GEV's location parameter due to the iGMRF.

# Arguments

- `F::iGMRF`: Spatial scheme.
- `θ::Vector{<:Real}`: Last updated parameters.
"""
function fullconditionalsIGMRF(F::iGMRF, θ::Vector{<:Real})

    Q = F.κ * Array(diag(F.G.W))
    b = -F.κ * (F.G.W̄ * θ)

    return NormalCanon.(b, Q)

end


"""
    createChain(M, μ, ϕ, ξ)

Create an object of type Mamba.Chains from the output of the mcmc algorithm.

# Arguments
- `M::Integer`: Number of cells.
- `μ::Matrix{<:Real}`: Traces of location parameters.
- `ϕ::Matrix{<:Real}`: Traces of log-scale parameters.
- `ξ::Matrix{<:Real}`: Traces of shape parameters.
"""
function createChain(M::Integer, μ::Matrix{<:Real}, ϕ::Matrix{<:Real}, ξ::Matrix{<:Real})
    
    paramnames = vcat(["μ$i" for i=1:M], ["ϕ$i" for i=1:M], ["ξ$i" for i=1:M]);
    res = vcat(μ, ϕ, ξ);
    
    return Mamba.Chains(collect(res'), names=paramnames)
end