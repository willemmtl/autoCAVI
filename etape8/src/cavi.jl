include("utils.jl");
include("model.jl");
include("monteCarlo.jl");

struct CAVIres
    MCKL::DenseVector
    approxMarginals::Vector{<:Distribution}
    traces::Dict
end


"""
CAVI algorithm.
"""
function runCAVI(nEpoch::Integer, epochSize::Integer, initialValues::Dict{Symbol, Any}, spatialScheme::Dict{Symbol, Any})
    
    M = spatialScheme[:M];

    traces = Dict(
        :muMean => zeros(M, nEpoch*epochSize),
        :phiMean => zeros(M, nEpoch*epochSize),
        :xiMean => zeros(nEpoch*epochSize),
        :cellVar => zeros(M, nEpoch, 4),
        :xiVar => zeros(M),
        :kappaUparams => zeros(2, nEpoch*epochSize),
        :kappaVparams => zeros(2, nEpoch*epochSize),
    )

    approxMarginals = Vector{Distribution}(undef, M+3);

    MCKL = zeros(nEpoch);

    # Initialisation

    traces[:muMean][:, 1] = initialValues[:μ];
    traces[:phiMean][:, 1] = initialValues[:ϕ];
    traces[:xiMean][1] = initialValues[:ξ];
    traces[:kappaUparams][1, :] = fill((M - 1)/2 + 1, nEpoch*epochSize);
    traces[:kappaUparams][2, 1] = initialValues[:κᵤ];
    traces[:kappaVparams][1, :] = fill((M - 1)/2 + 1, nEpoch*epochSize);
    traces[:kappaVparams][2, 1] = initialValues[:κᵥ];

    # CAVI

    caviCounter = Dict(
        :iter => 1,
        :epoch => 1,
        :numCell => 1,
    )

    compApproxMarginals!(approxMarginals, traces, caviCounter=caviCounter, spatialScheme=spatialScheme);
    compMCKL!(MCKL, approxMarginals, caviCounter=caviCounter, spatialScheme=spatialScheme);

    for k = 1:nEpoch
        caviCounter[:epoch] = k;
        runEpoch!(traces, MCKL, approxMarginals, caviCounter=caviCounter, epochSize=epochSize, spatialScheme=spatialScheme);
    end

    return CAVIres(
        MCKL,
        approxMarginals,
        traces,
    )
end


"""
    runEpoch!(traces, MCKL, approxMarginals; caviCounter, epochSize, spatialScheme)

Perform one epoch of the CAVI algorithm.

# Arguments
- `traces::Dict`: Traces of all variational parameters.
- `MCKL::DenseVector`: Trace of KL divergence values.
- `approxMarginals::Vector{<:Distribution}`: Current approximation marginals.
- `caviCounter::Dict`: Counters of the CAVI algorithm.
- `epochSize::Integer`: Number of iterations within the epoch.
- `spatialScheme::Dict`: Spatial structures and data.
"""
function runEpoch!(traces::Dict, MCKL::DenseVector, approxMarginals::Vector{<:Distribution}; caviCounter::Dict, epochSize::Integer, spatialScheme::Dict)
    
    for j = 1:epochSize

        caviCounter[:iter] = epochSize * (caviCounter[:epoch] - 1) + j;

        if (caviCounter[:iter] > 1)
            runIter!(traces, caviCounter=caviCounter, spatialScheme=spatialScheme);
        end
        
    end

    compApproxMarginals!(approxMarginals, traces, caviCounter=caviCounter, spatialScheme=spatialScheme);
    compMCKL!(MCKL, approxMarginals, caviCounter=caviCounter, spatialScheme=spatialScheme);
end


"""
    compMCKL!(MCKL, approxMarginals; caviCounter, spatialScheme)

Compute KL divergence between target and approximation densities using Monte Carlo.

# Arguments
- `MCKL::DenseVector`: Trace of KL divergence values.
- `approxMarginals::Vector{<:Distribution}`: Current approximation marginals.
- `caviCounter::Dict`: Counters of the CAVI algorithm.
- `spatialScheme::Dict`: Spatial structures and data.
"""
function compMCKL!(MCKL::DenseVector, approxMarginals::Vector{<:Distribution}; caviCounter::Dict, spatialScheme::Dict)

    Fmu = spatialScheme[:Fmu];
    Fphi = spatialScheme[:Fphi];
    data = spatialScheme[:data];

    logtarget(θ::DenseVector) = logposterior(θ, Fmu=Fmu, Fphi=Fphi, data=data);
    MCKL[caviCounter[:epoch]] =  MonteCarloKL(logtarget, approxMarginals);
end


"""
    compApproxMarginals!(approxMarginals, traces; caviCounter, spatialScheme)

Build the approximation marginals thanks to the current values of variational parameters.

# Arguments
- `approxMarginals::Vector{<:Distribution}`: Current approximation marginals.
- `traces::Dict`: Traces of all variational parameters.
- `caviCounter::Dict`: Counters of the CAVI algorithm.
- `spatialScheme::Dict`: Spatial structures and data.
"""
function compApproxMarginals!(approxMarginals::Vector{<:Distribution}, traces::Dict; caviCounter::Dict, spatialScheme::Dict)

    iter = caviCounter[:iter];
    epoch = caviCounter[:epoch];
    M = spatialScheme[:M];

    for i = 1:M

        m_i = [
            traces[:muMean][i, iter],
            traces[:phiMean][i, iter],
        ];
        
        
        cellVar = fisherVar(θi -> clfc(θi, caviCounter, traces, spatialScheme), m_i);
        approxMarginals[i] = MvNormal(m_i, round.(cellVar, digits = 12));
        
        caviCounter[:numCell] = i;
        traces[:cellVar][i, epoch, :] = flatten(cellVar);

    end

    xiVar = fisherVar(ξ -> xilfc(ξ, caviCounter, traces, spatialScheme), traces[:xiMean][iter]);
    approxMarginals[M+1] = Normal(traces[:xiMean][iter], sqrt(xiVar));
    approxMarginals[M+2] = Gamma(traces[:kappaUparams][1, iter], 1/traces[:kappaUparams][2, iter]);
    approxMarginals[M+3] = Gamma(traces[:kappaVparams][1, iter], 1/traces[:kappaVparams][2, iter]);

end


"""
    runIter!(traces; caviCounter, spatialScheme)

Perform one iteration of the CAVI algorithm.

# Arguments
- `traces::Dict`: Traces of all variational parameters.
- `caviCounter::Dict`: Counters of the CAVI algorithm.
- `spatialScheme::Dict`: Spatial structures and data.
"""
function runIter!(traces::Dict; caviCounter::Dict, spatialScheme::Dict)

    iter = caviCounter[:iter];

    traces[:muMean][:, iter] = traces[:muMean][:, iter-1];
    traces[:phiMean][:, iter] = traces[:phiMean][:, iter-1];
    traces[:xiMean][iter] = traces[:xiMean][iter-1];
    traces[:kappaUparams][2, iter] = traces[:kappaUparams][2, iter-1];
    traces[:kappaVparams][2, iter] = traces[:kappaVparams][2, iter-1];

    for i = 1:spatialScheme[:M]

        caviCounter[:numCell] = i;

        (
            traces[:muMean][i, iter],
            traces[:phiMean][i, iter],
        ) = findMode(
            θi -> clfc(θi, caviCounter, traces, spatialScheme),
            [
                traces[:muMean][i, iter],
                traces[:phiMean][i, iter],
            ],
        );
    end

    traces[:xiMean][iter] = findMode(ξ -> xilfc(ξ, caviCounter, traces, spatialScheme), traces[:xiMean][iter])[1];
    traces[:kappaUparams][2, iter] = compKappaParam(traces[:muMean][:, iter], Fmu);
    traces[:kappaVparams][2, iter] = compKappaParam(traces[:phiMean][:, iter], Fphi);


end


"""
Log full conditional density of [μi, ϕi] knowing all other parameters.
"""
function clfc(θi::DenseVector, caviCounter::Dict, traces::Dict, spatialScheme::Dict)

    numCell = caviCounter[:numCell];
    iter = caviCounter[:iter];

    Fmu = spatialScheme[:Fmu];
    Fphi = spatialScheme[:Fphi];
    data = spatialScheme[:data];

    μ = traces[:muMean][:, iter];
    ϕ = traces[:phiMean][:, iter];
    ξ = traces[:xiMean][iter];
    
    μ̄i = neighborsMean(numCell, μ, Fmu);
    ϕ̄i = neighborsMean(numCell, ϕ, Fphi);
    κ̂ᵤ =  estimateKappa("u", traces=traces, caviCounter=caviCounter);
    κ̂ᵥ =  estimateKappa("v", traces=traces, caviCounter=caviCounter);

    return celllogfullconditional(
        numCell,
        θi,
        ξ=ξ,
        μ̄i=μ̄i,
        ϕ̄i=ϕ̄i,
        κ̂ᵤ=κ̂ᵤ,
        κ̂ᵥ=κ̂ᵥ,
        Fmu=Fmu,
        Fphi=Fphi,
        data=data,
    )

end


"""
Log full conditional density of ξ knowing all other variables.
"""
function xilfc(ξ::Real, caviCounter::Dict, traces::Dict, spatialScheme::Dict)
    
    iter = caviCounter[:iter];

    data = spatialScheme[:data];

    μ = traces[:muMean][:, iter];
    ϕ = traces[:phiMean][:, iter];

    return xilogfullconditional(
        ξ,
        μ=μ,
        ϕ=ϕ,
        data=data,
    )

end


"""
    estimateKappa(variant; traces, caviCounter)

Compute kappa estimate based on the parameters of its law (Gamma).

# Arguments
- `variant::String`: Which kappa to compute, 'u' or 'v'.
- `traces::Dict`: Traces of all variational parameters.
- `caviCounter::Dict`: Counters of the CAVI algorithm.
"""
function estimateKappa(variant::String; traces::Dict, caviCounter::Dict)
    
    iter = caviCounter[:iter];

    if (variant == "u")
        return traces[:kappaUparams][1, iter] / traces[:kappaUparams][2, iter];
    elseif (variant == "v")
        return traces[:kappaVparams][1, iter] / traces[:kappaVparams][2, iter];
    end
end