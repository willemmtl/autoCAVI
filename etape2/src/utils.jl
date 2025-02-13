using Distributions, Optim, ForwardDiff

"""
    findMode(f, x0)

Find the mode of a given density f from starting point x0.

# Arguments
- `f::Function`: Functional form of the density.
- `x0::DenseVector`: Starting point (must be a vector).
"""
function findMode(f::Function, x0::DenseVector)
    F(x::DenseVector) = -f(x);
    mode = optimize(F, x0);
    return Optim.minimizer(mode)
end;


"""
    getSecondDerivative(f, x0)
"""
function getSecondDerivative(f::Function, x0::Real)
    return ForwardDiff.derivative(x -> ForwardDiff.derivative(f, x), x0)
end


"""
    MonteCarloKL(logTargetDensity, approxMarginals)

Compute the convergence criterion with current hyper-parameters.

# Arguments
- `logTargetDensity::Function`: The density we're trying to approximate (log).
- `approxMarginals::Vector{Distribution}`: The marginal distribution of each parameter.
"""
function MonteCarloKL(logTargetDensity::Function, approxMarginals::Vector{<:Distribution})
    
    N = 1000;
    supp = generateApproxSample(approxMarginals, N);

    logApproxDensity(θ::DenseVector) = (
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
    
    supp = zeros(3, N);

    for (k, approxMarginal) in enumerate(approxMarginals)
        supp[k, :] = rand(approxMarginal, N);
    end
    
    return supp
end


"""
    evaluateLogMvDensity(f, supp)

Evaluate a log multivariate density over a given set of vectors.

# Arguments :
- `f::Function`: Log multivariate density function to evaluate.
- `supp::Matrix{<:Real}`: Set of n p-arrays stored in a (p x n) matrix.
"""
function evaluateLogMvDensity(f::Function, supp::Matrix{<:Real})
    return vec(mapslices(f, supp, dims=1))
end