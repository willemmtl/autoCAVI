using Gadfly, Cairo, Fontconfig, Distributions


"""
    plotApproxVSMCMC(approxMarginals, mcmcSamples, warmingSize)

Plot approx marginals and histogram of MCMC samples for each parameter.

# Arguments
TBD
"""
function plotApproxVSMCMC(
    approxMarginals::Vector{<:Distribution}, 
    mcmcSamples::Matrix{<:Real}, 
    warmingSize::Integer,
)
    
    set_default_plot_size(15cm ,21cm)

    x = 5:.001:13;
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

    x = -.3:.001:.8;
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
    plotApproxVSMCMC(approxMarginals, mcmcSamples, warmingSize, path)

Plot approx marginals and histogram of MCMC samples for each parameter.
Draw the plot at the given folder path.

# Arguments
TBD
"""
function plotApproxVSMCMC(
    approxMarginals::Vector{<:Distribution}, 
    mcmcSamples::Matrix{<:Real}, 
    warmingSize::Integer,
    path::String,
)
    
    set_default_plot_size(15cm ,21cm)

    x = 5:.001:15;
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

    x = -.3:.001:.8;
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