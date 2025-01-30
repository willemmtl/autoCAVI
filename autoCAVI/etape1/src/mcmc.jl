using Distributions, LinearAlgebra
using Distributions:loglikelihood

"""
    mcmc(niter, δ, y0, data)

Metropolis algorithm to sample from the posterior.

# Arguments
- `niter::Integer`: Number of iterations.
- `δ::DenseVector`: Instrumental variances.
- `y0::DenseVector`: Initial values.
- `data::DenseVector`: Observations for the log-posterior density.
"""
function mcmc(niter::Integer, δ::DenseVector, y0::DenseVector, data::DenseVector)

    θ = zeros(3, niter);
    acc = zeros(3);

    θ[:, 1] = y0;

    for j = 2:niter
        for i = 1:3
            cand = rand(Normal(θ[i, j-1], sqrt(δ[i])));
            y = θ[:, j-1];
            y[i] = cand;
            lr = logposterior(y, data) - logposterior(θ[:, j-1], data);

            if lr > log(rand())
                θ[i, j] = cand;
                acc[i] += 1;
            else
                θ[i, j] = θ[i, j-1];
            end
        end
    end

    acc = acc .* 100 ./ (niter - 1);
    println("Taux d'acceptation (%) : ", acc);

    return θ
end


"""
    logposterior(θ, x)

Log-posterior density.

# Arguments
- `θ::DenseVector`: Parameters [μ, ϕ, ξ].
- `x::DenseVector`: Observations.
"""
function logposterior(θ::DenseVector, x::DenseVector)
    return loglikelihood(GeneralizedExtremeValue(θ[1], exp(θ[2]), θ[3]), x)
end