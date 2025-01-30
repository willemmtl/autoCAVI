"""
    MonteCarloKL(logTargetDensity, approxMarginals)

Compute the convergence criterion with current hyper-parameters.

# Arguments
TBD
"""
function MonteCarloKL(logTargetDensity::Function, approxMarginals::Vector{<:Distribution})
    
    N = 1000;
    supp = generateApproxSample(approxMarginals, N);

    logApproxDensity(θ::DenseVector) = sum(
        logpdf(approxMarginals[1], θ[1]) 
        + logpdf(approxMarginals[2], θ[2]) 
        + logpdf(approxMarginals[3], θ[3])
    )

    logTarget = evaluateLogMvDensity(x -> logTargetDensity(x), supp);
    logApprox = evaluateLogMvDensity(x -> logApproxDensity(x), supp);
    
    return sum(logApprox .- logTarget) / N
end;


"""
    generateApproxSample(approxMarginals, N)

Draw samples from the approximating distribution with current hyper-parameters.

Use the mean-field approximation by generating each variable independantly.
Will be used to compute KL divergence.

# Arguments :
- `approxMarginals::Vector{Distribution}`: The marginal distribution of each parameter.
- `N::Integer`: Sample size (the same for all variables).
"""
function generateApproxSample(approxMarginals::Vector{<:Distribution}, N::Integer)
    
    supp = zeros(3, N, 9);

    for (k, approxMarginal) in enumerate(approxMarginals)
        supp[k, :] = rand(approxMarginal, N);
    end
    
    return supp
end


function tensToMat(tensor::Array{Float64, 3})
    
    G, N, M = size(tensor)
    
    return reshape(tensor)
end