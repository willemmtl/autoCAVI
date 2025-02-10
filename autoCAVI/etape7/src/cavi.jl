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
    )

    approxMarginals = Vector{Distribution}(undef, M+1);

    MCKL = zeros(nEpoch);

    # Initialisation

    traces[:muMean][:, 1] = initialValues[:μ];
    traces[:phiMean][:, 1] = initialValues[:ϕ];
    traces[:xiMean][1] = initialValues[:ξ];

    # CAVI

    caviCounter = Dict(
        :iter => 1,
        :epoch => 1,
        :numCell => 1,
    )

    computeMCKL!(MCKL, approxMarginals, traces=traces, caviCounter=caviCounter, spatialScheme=spatialScheme);

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
"""
function runEpoch!(traces::Dict, MCKL::DenseVector, approxMarginals::Vector{<:Distribution}; caviCounter::Dict, epochSize::Integer, spatialScheme::Dict)
    
    for j = 1:epochSize

        caviCounter[:iter] = epochSize * (caviCounter[:epoch] - 1) + j;

        if (caviCounter[:iter] > 1)
            runIter!(traces, caviCounter=caviCounter, spatialScheme=spatialScheme);
        end
        
    end

    computeMCKL!(MCKL, approxMarginals, traces=traces, caviCounter=caviCounter, spatialScheme=spatialScheme);
end


"""
"""
function computeMCKL!(MCKL::DenseVector, approxMarginals::Vector{<:Distribution}; traces::Dict, caviCounter::Dict, spatialScheme::Dict)
    
    computeApproxMarginals!(approxMarginals, traces, caviCounter=caviCounter, spatialScheme=spatialScheme);

    Fmu = spatialScheme[:Fmu];
    Fphi = spatialScheme[:Fphi];
    data = spatialScheme[:data];

    logtarget(θ::DenseVector) = logposterior(θ, Fmu=Fmu, Fphi=Fphi, data=data);
    MCKL[caviCounter[:epoch]] =  MonteCarloKL(logtarget, approxMarginals);

end


"""
"""
function computeApproxMarginals!(approxMarginals::Vector{<:Distribution}, traces::Dict; caviCounter::Dict, spatialScheme::Dict)

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

end


"""
"""
function runIter!(traces::Dict; caviCounter::Dict, spatialScheme::Dict)

    iter = caviCounter[:iter];

    traces[:muMean][:, iter] = traces[:muMean][:, iter-1];
    traces[:phiMean][:, iter] = traces[:phiMean][:, iter-1];
    traces[:xiMean][iter] = traces[:xiMean][iter-1];

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

    return celllogfullconditional(
        numCell,
        θi,
        ξ=ξ,
        μ̄i=μ̄i,
        ϕ̄i=ϕ̄i,
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