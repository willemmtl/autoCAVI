using Distributions:loglikelihood

function logposterior(θ::DenseVector, data::DenseVector)
    return (
        loglikelihood(GeneralizedExtremeValue(θ[1], exp(θ[2]), θ[3]), data)
        + logpdf(Normal(50, 100), θ[1])
        + logpdf(Normal(8, 10), θ[2])
        + logpdf(Normal(-.3, 1), θ[3])
    )
end