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
function runCAVI(nEpoch::Integer, epochSize::Integer, initialValues::Dict{Symbol, Vector{Float64}}, spatialScheme::Dict{Symbol, Any})
    
    M = spatialScheme[:M];

    traces = Dict(
        :muMean => zeros(M, nEpoch*epochSize),
        :phiMean => zeros(M, nEpoch*epochSize),
        :xiMean => zeros(M, nEpoch*epochSize),
        :var => zeros(M, nEpoch, 9),
    )

    approxMarginals = Vector{Distribution}(undef, M);

    MCKL = zeros(nEpoch);

    # Initialisation

    traces[:muMean][:, 1] = initialValues[:μ];
    traces[:phiMean][:, 1] = initialValues[:ϕ];
    traces[:xiMean][:, 1] = initialValues[:ξ];

    # CAVI

    caviCounter = Dict(
        :iter => 1,
        :epoch => 1,
        :numCell => 1,
    )

    computeMCKL!(MCKL, approxMarginals, traces=traces, caviCounter=caviCounter, spatialScheme=spatialScheme);

    for k = 1:nEpoch
        caviCounter[:epoch] = k;
        runEpoch!(traces, approxMarginals, MCKL, caviCounter=caviCounter, epochSize=epochSize, spatialScheme=spatialScheme);
    end

    return CAVIres(
        MCKL,
        approxMarginals,
        traces,
    )
end


"""
"""
function runEpoch!(traces::Dict, approxMarginals::Vector{<:Distribution}, MCKL::DenseVector; caviCounter::Dict, epochSize::Integer, spatialScheme::Dict)
    
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
    Fxi = spatialScheme[:Fxi];
    data = spatialScheme[:data];

    logtarget(θ::DenseVector) = logposterior(θ, Fmu=Fmu, Fphi=Fphi, Fxi=Fxi, data=data);
    MCKL[caviCounter[:epoch]] =  MonteCarloKL(logtarget, approxMarginals);

end


"""
"""
function computeApproxMarginals!(approxMarginals::Vector{<:Distribution}, traces::Dict; caviCounter::Dict, spatialScheme::Dict)

    iter = caviCounter[:iter];
    epoch = caviCounter[:epoch];

    for i = 1:spatialScheme[:M]

        mode = [
            traces[:muMean][i, iter],
            traces[:phiMean][i, iter],
            traces[:xiMean][i, iter],
        ];

        caviCounter[:numCell] = i;

        var = fisherVar(θi -> lfc(θi, caviCounter, traces, spatialScheme), mode);
            
        approxMarginals[i] = MvNormal(mode, round.(var, digits = 7));

        traces[:var][i, epoch, :] = flatten(var);

    end

end


"""
"""
function runIter!(traces::Dict; caviCounter::Dict, spatialScheme::Dict)

    iter = caviCounter[:iter];

    traces[:muMean][:, iter] = traces[:muMean][:, iter-1];
    traces[:phiMean][:, iter] = traces[:phiMean][:, iter-1];
    traces[:xiMean][:, iter] = traces[:xiMean][:, iter-1];

    for i = 1:spatialScheme[:M]

        caviCounter[:numCell] = i;

        (
            traces[:muMean][i, iter],
            traces[:phiMean][i, iter],
            traces[:xiMean][i, iter],
        ) = findMode(
            θi -> lfc(θi, caviCounter, traces, spatialScheme),
            [
                traces[:muMean][i, iter],
                traces[:phiMean][i, iter],
                traces[:xiMean][i, iter],
            ],
        );
    end
end


"""
"""
function lfc(θi::DenseVector, caviCounter::Dict, traces::Dict, spatialScheme::Dict)

    numCell = caviCounter[:numCell];
    iter = caviCounter[:iter];

    Fmu = spatialScheme[:Fmu];
    Fphi = spatialScheme[:Fphi];
    Fxi = spatialScheme[:Fxi];
    data = spatialScheme[:data];

    μ̄i = neighborsMean(numCell, traces[:muMean][:, iter], Fmu);
    ϕ̄i = neighborsMean(numCell, traces[:phiMean][:, iter], Fphi);
    ξ̄i = neighborsMean(numCell, traces[:xiMean][:, iter], Fxi);

    return logfullconditional(
            numCell,
            θi,
            μ̄i=μ̄i,
            ϕ̄i=ϕ̄i,
            ξ̄i=ξ̄i,
            Fmu=Fmu,
            Fphi=Fphi,
            Fxi=Fxi,
            data=data,
    )

end