using Optim, ForwardDiff, LinearAlgebra


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
    fisherVar(f, x)

Compute the covariance matrix derived from the Fisher information.

# Arguments
- `logf::Function`: Log-functional form of the density.
- `x::DenseVector`: Where to compute the Fisher info (usually the mode).
"""
function fisherVar(logf::Function, x::DenseVector)
    return - ForwardDiff.hessian(logf, x) \ I
end