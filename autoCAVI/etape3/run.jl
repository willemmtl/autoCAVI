using Distributions, Gadfly

include("src/model.jl");
include("src/utils.jl");
include("src/mcmc.jl");
include("src/plotting.jl");

# ---- GENERATING DATA ----

N = 100;
realmu = 10.0;
realphi = 2.0;
realxi = .3;

data = rand(GeneralizedExtremeValue(realmu, exp(realphi), realxi), N);

# ---- LAPLACE APPROX ----

m = findMode(θ -> logposterior(θ, data), [0.0, 0.0, 0.0]);

s² = fisherVar(θ -> logposterior(θ, data), m);

# ---- MCMC ----

niter = 100000;
δ = [10, .15, .15];
y0 = [0.0, 0.0, 0.0];

θ = mcmc(niter, δ, y0, data);

# ---- PLOTS ----

warmingSize = 1000;

approxMarginals = [
    Normal(m[1], sqrt(s²[1, 1])),
    Normal(m[2], sqrt(s²[2, 2])),
    Normal(m[3], sqrt(s²[3, 3])),    
];

plotApproxVSMCMC(
    approxMarginals,
    θ,
    warmingSize,
    "plots/etape3/approx_vs_mcmc.png",
);