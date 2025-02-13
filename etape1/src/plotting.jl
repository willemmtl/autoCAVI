using Gadfly, Cairo, Fontconfig, Distributions

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
    plotApproxVSMCMC(approxMarginals, mcmcSamples, warmingSize)

Plot approx marginals and histogram of MCMC samples for each parameter.

# Arguments
TBD
"""
function plotApproxVSMCMC(
    approxMarginals::Vector{<:Distribution}, 
    mcmcSamples::Matrix{<:Float64},
    warmingSize::Integer,
)
    
    set_default_plot_size(15cm ,21cm)

    x = 9:.001:11;
    p1 = plot(
        layer(x=x, y=pdf.(approxMarginals[1], x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSamples[1, warmingSize:end], Geom.histogram(density=true)),
        Theme(background_color="white"),
        Guide.title("Approx vs MCMC for mu"),
        Guide.xlabel("mu"),
        Guide.ylabel("Density"),
    );

    x = 1.5:.001:2.5;
    p2 = plot(
        layer(x=x, y=pdf.(approxMarginals[2], x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSamples[2, warmingSize:end], Geom.histogram(density=true)),
        Theme(background_color="white"),
        Guide.title("Approx vs MCMC for phi"),
        Guide.xlabel("phi"),
        Guide.ylabel("Density"),
    );

    x = 0:.001:.5;
    p3 = plot(
        layer(x=x, y=pdf.(approxMarginals[3], x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSamples[3, warmingSize:end], Geom.histogram(density=true)),
        Theme(background_color="white"),
        Guide.title("Approx vs MCMC for xi"),
        Guide.xlabel("xi"),
        Guide.ylabel("Density"),
    );

    vstack(p1, p2, p3)
end


"""
    plotApproxVSMCMC(approxMarginals, mcmcSamples, warmingSize)

Plot approx marginals and histogram of MCMC samples for each parameter.

# Arguments
TBD
"""
function plotApproxVSMCMC(
    approxMarginals::Vector{<:Distribution}, 
    mcmcSamples::Matrix{<:Float64},
    warmingSize::Integer,
    path::String,
)
    
    set_default_plot_size(15cm ,21cm)

    x = 9:.001:11;
    p1 = plot(
        layer(x=x, y=pdf.(approxMarginals[1], x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSamples[1, warmingSize:end], Geom.histogram(density=true)),
        Theme(background_color="white"),
        Guide.title("Approx vs MCMC for mu"),
        Guide.xlabel("mu"),
        Guide.ylabel("Density"),
    );

    x = 1.5:.001:2.5;
    p2 = plot(
        layer(x=x, y=pdf.(approxMarginals[2], x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSamples[2, warmingSize:end], Geom.histogram(density=true)),
        Theme(background_color="white"),
        Guide.title("Approx vs MCMC for phi"),
        Guide.xlabel("phi"),
        Guide.ylabel("Density"),
    );

    x = 0:.001:.5;
    p3 = plot(
        layer(x=x, y=pdf.(approxMarginals[3], x), Geom.line, Theme(default_color="red")),
        layer(x=mcmcSamples[3, warmingSize:end], Geom.histogram(density=true)),
        Theme(background_color="white"),
        Guide.title("Approx vs MCMC for xi"),
        Guide.xlabel("xi"),
        Guide.ylabel("Density"),
    );

    Gadfly.draw(PNG(path, dpi=300), vstack(p1, p2, p3))
end