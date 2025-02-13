using Distributions:loglikelihood

function logposterior(θ::DenseVector, data::DenseVector)
    return loglikelihood(GeneralizedExtremeValue(θ[1], exp(θ[2]), θ[3]), data)
end