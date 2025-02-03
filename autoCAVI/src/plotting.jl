using Gadfly, Cairo, Fontconfig, Distributions, Mamba

"""
    plotConvergenceCriterion(MCKL)

Plot evolution of the KL divergence over CAVI epochs.

# Arguments
- `MCKL::DenseVector`: Values of the KL divergence for each epoch.
"""
function plotConvergenceCriterion(MCKL::DenseVector)
    
    set_default_plot_size(15cm ,10cm)

    n_mckl = length(MCKL);

    plot(
        layer(x=1:n_mckl, y=MCKL, Geom.line),
        layer(x=1:n_mckl, y=MCKL, Geom.point, shape=[Shape.cross], Theme(default_color="red")),
        Theme(background_color="white"),
        Guide.title("Convergence criterion"),
        Guide.xlabel("Epoch"),
        Guide.ylabel("KL divergence"),
    )
end


"""
    plotTraceCAVI(trace, name)

Plot evolution of the KL divergence over CAVI epochs.

# Arguments
- `trace::DenseVector`: Trace of the parameter.
- `name::String`: Name of the parameter.
"""
function plotTraceCAVI(trace::DenseVector, name::String)
    
    set_default_plot_size(15cm ,10cm)

    n_trace = length(trace);

    plot(
        layer(x=1:n_trace, y=trace, Geom.line),
        layer(x=1:n_trace, y=trace, Geom.point, shape=[Shape.cross], Theme(default_color="red")),
        Theme(background_color="white"),
        Guide.title("CAVI trace of $name"),
        Guide.xlabel("Iteration"),
        Guide.ylabel("Value"),
    )
end


"""
    plotTraceMCMC(chain, name)

Plot evolution of the KL divergence over CAVI epochs.

# Arguments
- `chain::Mamba.Chains`: Traces of all parameters.
- `name::String`: Name of the parameter.
"""
function plotTraceMCMC(chain::Mamba.Chains, name::String)
    
    set_default_plot_size(15cm ,10cm)

    trace = chain[:, name, 1].value;
    n_trace = length(trace);

    plot(
        layer(x=1:n_trace, y=trace, Geom.line),
        Theme(background_color="white"),
        Guide.title("MCMC trace of $name"),
        Guide.xlabel("Iteration"),
        Guide.ylabel("Value"),
    )
end


"""
    plotCAVIvsMCMC(numCell; caviRes, mcmcChain, warmingSize)

Plot approx marginals and histogram of MCMC samples for each parameter.

# Arguments
TBD
"""
function plotCAVIvsMCMC(
    numCell::Integer;
    caviRes::CAVIres,
    mcmcChain::Mamba.Chains, 
    warmingSize::Integer,
)

    set_default_plot_size(15cm ,21cm)

    x = 7:.001:13;

    marginal = buildCAVImarginal(numCell, 1, caviRes=caviRes);
    mcmcSample = mcmcChain[:, "μ$numCell", 1].value[warmingSize:end];

    p1 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true)),
        Theme(background_color="white"),
        Guide.title("Approx vs MCMC for mu"),
        Guide.xlabel("mu"),
        Guide.ylabel("Density"),
    );

    x = .5:.001:1.5;

    marginal = buildCAVImarginal(numCell, 2, caviRes=caviRes);
    mcmcSample = mcmcChain[:, "ϕ$numCell", 1].value[warmingSize:end];

    p2 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true)),
        Theme(background_color="white"),
        Guide.title("Approx vs MCMC for phi"),
        Guide.xlabel("phi"),
        Guide.ylabel("Density"),
    );

    x = 0:.001:.5;

    marginal = buildCAVImarginal(numCell, 3, caviRes=caviRes);
    mcmcSample = mcmcChain[:, "ξ$numCell", 1].value[warmingSize:end];

    p3 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true)),
        Theme(background_color="white"),
        Guide.title("Approx vs MCMC for xi"),
        Guide.xlabel("xi"),
        Guide.ylabel("Density"),
    );

    vstack(p1, p2, p3)
end


"""
    plotCAVIvsMCMC(approxMarginals, mcmcSamples, warmingSize, path)

Plot approx marginals and histogram of MCMC samples for each parameter.
Draw the plot at the given folder path.

# Arguments
TBD
"""
function plotCAVIvsMCMC(
    numCell::Integer;
    caviRes::CAVIres,
    mcmcChain::Mamba.Chains, 
    warmingSize::Integer,
)
    
    set_default_plot_size(15cm ,21cm)

    x = 7:.001:13;

    marginal = buildCAVImarginal(numCell, 1, caviRes=caviRes);
    mcmcSample = mcmcChain[:, "μ$numCell", 1].value[warmingSize:end];

    p1 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true)),
        Theme(background_color="white"),
        Guide.title("Approx vs MCMC for mu"),
        Guide.xlabel("mu"),
        Guide.ylabel("Density"),
    );

    x = .5:.001:1.5;

    marginal = buildCAVImarginal(numCell, 2, caviRes=caviRes);
    mcmcSample = mcmcChain[:, "ϕ$numCell", 1].value[warmingSize:end];

    p2 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true)),
        Theme(background_color="white"),
        Guide.title("Approx vs MCMC for phi"),
        Guide.xlabel("phi"),
        Guide.ylabel("Density"),
    );

    x = 0:.001:.5;

    marginal = buildCAVImarginal(numCell, 3, caviRes=caviRes);
    mcmcSample = mcmcChain[:, "ξ$numCell", 1].value[warmingSize:end];

    p3 = plot(
        layer(x=x, y=pdf.(marginal, x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSample, Geom.histogram(density=true)),
        Theme(background_color="white"),
        Guide.title("Approx vs MCMC for xi"),
        Guide.xlabel("xi"),
        Guide.ylabel("Density"),
    );

    Gadfly.draw(PNG(path, dpi=300), vstack(p1, p2, p3))
end


"""
    buildCAVImarginal(numCell, paramNum; caviRes)

Build the marginal density of a parameter of a given cell from the result of the CAVI algorithm.

# Arguments
TBD
"""
function buildCAVImarginal(numCell::Integer, paramNum::Integer; caviRes::CAVIres)
    return Normal(
        params(caviRes.approxMarginals[numCell])[1][paramNum],
        sqrt(diag(params(caviRes.approxMarginals[numCell])[2])[paramNum])
    )
end